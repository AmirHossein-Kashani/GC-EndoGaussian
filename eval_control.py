"""Control-from-tracks controllability metric (Leg 2 — the centerpiece).

Given a few ground-truth tracked tissue points used as CONTROL HANDLES (we set the control nodes
nearest them to the handles' OBSERVED motion), how well does the control graph PREDICT the motion of
the remaining HELD-OUT tracked points? This is a GT-backed *controllability* number — given sparse
control, predict the dense deformation — and needs no robot kinematics.

Reported: median reprojection error (px) on held-out points, a controllability curve over K handles,
leave-groups-out CV with bootstrap CIs, vs. classical control baselines (rigid / nearest / TPS) and
the gnn_layers=0 ablation (run separately on that model).

HONEST SCOPE: the control input is GT *tissue* motion, not a robot command — a
deformation-prediction-under-sparse-observation metric, not closed-loop controllability.

Usage: python eval_control.py --model_path output/endonerf/super_match \
         --configs arguments/endonerf/super_graph_match.py \
         --tracks data/super/v2_data/trial_3/rgb/trial_3_l_pts.npy --pad_v 16 --iteration 3000
"""
import os, sys, math, json
import numpy as np
import torch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from scene import Scene, GaussianModel
from eval_tracking import deformed_xyz, project, bootstrap_ci


# ---------- geometry helpers (static SuPer camera: c2w = I, centered intrinsics) ----------
def cam_focal_cxcy(cam, W, H):
    fx = (W / 2.0) / math.tan(cam.FoVx * 0.5)
    return fx, W / 2.0, H / 2.0

def depth_at(cam, u, v, W, H):
    """Median depth in a 3x3 window at pixel (u,v); 0 if invalid."""
    d = cam.original_depth
    d = d.squeeze().detach().cpu().numpy() if torch.is_tensor(d) else np.asarray(d).squeeze()
    ui, vi = int(round(u)), int(round(v))
    if ui < 1 or vi < 1 or ui >= W - 1 or vi >= H - 1:
        return 0.0
    patch = d[vi - 1:vi + 2, ui - 1:ui + 2]
    patch = patch[patch > 0]
    return float(np.median(patch)) if patch.size else 0.0

def backproject(u, v, Z, fx, cx, cy):
    return np.array([(u - cx) / fx * Z, (v - cy) / fx * Z, Z], dtype=np.float64)


# ---------- thin-plate-spline RBF (3D->3D) baseline ----------
def tps_fit(P, V, lam=1e-3):
    K = P.shape[0]
    r = np.linalg.norm(P[:, None] - P[None], axis=-1)
    Phi = r ** 2 * np.log(r + 1e-9); np.fill_diagonal(Phi, 0.0); Phi += lam * np.eye(K)
    Pp = np.hstack([np.ones((K, 1)), P])
    A = np.block([[Phi, Pp], [Pp.T, np.zeros((4, 4))]])
    b = np.vstack([V, np.zeros((4, 3))])
    return np.linalg.lstsq(A, b, rcond=None)[0]

def tps_eval(P, w, Q):
    K = P.shape[0]
    r = np.linalg.norm(Q[:, None] - P[None], axis=-1)
    Phi = r ** 2 * np.log(r + 1e-9)
    Qp = np.hstack([np.ones((Q.shape[0], 1)), Q])
    return Phi @ w[:K] + Qp @ w[K:]


def farthest_point_idx(pts, k):
    """FPS over 2D/3D points -> k indices covering the set."""
    n = pts.shape[0]; k = min(k, n)
    sel = [0]; d = np.full(n, np.inf)
    for _ in range(1, k):
        d = np.minimum(d, np.linalg.norm(pts - pts[sel[-1]], axis=1))
        sel.append(int(np.argmax(d)))
    return np.array(sel)


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=3000, type=int)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--tracks", type=str, required=True)
    parser.add_argument("--pad_v", type=int, default=16)
    parser.add_argument("--splat_k", type=int, default=8)   # nodes each handle drives
    args = get_combined_args(parser)
    if args.configs:
        from utils.config_loader import load_config
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, load_config(args.configs))
    dataset, pipe, hyper = model.extract(args), pipeline.extract(args), hp.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyper)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        if not getattr(gaussians, "use_node_graph", False) or not gaussians.node_seeded:
            print("ERROR: control metric requires a seeded node graph."); sys.exit(1)
        nd = gaussians._deformation.deformation_net.node_deform
        node_xyz = nd.node_xyz.detach().cpu().numpy()
        cams = scene.getVideoCameras()
        N = len(cams); W, H = int(cams[0].image_width), int(cams[0].image_height)
        fx, cx, cy = cam_focal_cxcy(cams[0], W, H)
        dev = nd.node_xyz.device

        gt = np.load(args.tracks, allow_pickle=True).tolist()
        gt = gt["gt"] if isinstance(gt, dict) and "gt" in gt else gt
        keys = sorted(gt.keys()); T = min(len(keys), N)
        tracks = np.stack([np.asarray(gt[keys[f]], np.float64) for f in range(T)], 0)  # (T,P,3)
        tracks[:, :, 1] += args.pad_v
        P = tracks.shape[1]

        # frame-0 world positions of all points (for FPS + baseline anchors)
        x0 = np.full((P, 3), np.nan)
        for p in range(P):
            u, v, vis = tracks[0, p]
            if vis and u > 0 and v > 0:
                Z = depth_at(cams[0], u, v, W, H)
                if Z > 0:
                    x0[p] = backproject(u, v, Z, fx, cx, cy)
        valid = ~np.isnan(x0).any(1)

        # anchor each point to nearest front-most Gaussian at frame 0 (for the GRAPH prediction)
        xyz0 = deformed_xyz(gaussians, cams[0].time)
        uv0, z0 = project(xyz0, cams[0], W, H)
        anchor = np.full(P, -1, np.int64)
        for p in range(P):
            if not valid[p]:
                continue
            d2 = (uv0[:, 0] - tracks[0, p, 0]) ** 2 + (uv0[:, 1] - tracks[0, p, 1]) ** 2 + (z0 < 0).float() * 1e9
            near = torch.topk(d2, min(15, d2.numel()), largest=False).indices
            anchor[p] = int(near[torch.argmin(z0[near])].item())

        node_xyz_t = nd.node_xyz                                    # (M,3) cuda
        results = {}

        def predict_and_score(handles, targets, methods):
            """Per (method): list of held-out reprojection errors over all frames."""
            errs = {m: [] for m in methods}
            for f in range(1, T):
                # handle displacements at frame f (world)
                hd, hx0, hxn = [], [], []
                for p in handles:
                    u, v, vis = tracks[f, p]
                    if not (valid[p] and vis and u > 0 and v > 0):
                        continue
                    Z = depth_at(cams[f], u, v, W, H)
                    if Z <= 0:
                        continue
                    xf = backproject(u, v, Z, fx, cx, cy)
                    hd.append(xf - x0[p]); hx0.append(x0[p]); hxn.append(xf)
                if len(hd) < 1:
                    continue
                hd = np.array(hd); hx0 = np.array(hx0)

                # ----- GRAPH: drive nodes near handles, control_only, deform anchored targets -----
                if "graph" in methods:
                    edit = torch.zeros_like(node_xyz_t)
                    wsum = torch.zeros(node_xyz_t.shape[0], 1, device=dev)
                    for j in range(len(hd)):
                        h0 = torch.tensor(hx0[j], device=dev, dtype=node_xyz_t.dtype)
                        dist2 = ((node_xyz_t - h0) ** 2).sum(1)
                        kk = torch.topk(dist2, min(args.splat_k, dist2.numel()), largest=False)
                        ww = torch.softmax(-kk.values / (kk.values[:1] + 1e-8), 0).unsqueeze(1)
                        edit[kk.indices] += ww * torch.tensor(hd[j], device=dev, dtype=node_xyz_t.dtype)
                        wsum[kk.indices] += ww
                    edit = edit / wsum.clamp_min(1e-8)
                    nd.edit_translation = edit
                    nd.control_only = True
                    xyz = deformed_xyz(gaussians, cams[f].time)
                    nd.control_only = False
                    nd.edit_translation = torch.zeros_like(node_xyz_t)
                    uv, _ = project(xyz, cams[f], W, H)

                # ----- baselines: predict target world pos via interpolation of handle disp -----
                tw = tps_fit(hx0, hd) if ("tps" in methods and len(hd) >= 4) else None
                for p in targets:
                    if not valid[p]:
                        continue
                    gu, gv, vis = tracks[f, p]
                    if not (vis and gu > 0 and gv > 0):
                        continue
                    if "graph" in methods and anchor[p] >= 0:
                        pu, pv = uv[anchor[p]].tolist()
                        errs["graph"].append(float(np.hypot(pu - gu, pv - gv)))
                    base = x0[p]
                    if "rigid" in methods:
                        q = base + hd.mean(0); errs["rigid"].append(reproj_err(q, gu, gv, fx, cx, cy))
                    if "nearest" in methods:
                        nn = int(np.argmin(np.linalg.norm(hx0 - base, axis=1)))
                        q = base + hd[nn]; errs["nearest"].append(reproj_err(q, gu, gv, fx, cx, cy))
                    if "tps" in methods and tw is not None:
                        q = base + tps_eval(hx0, tw, base[None])[0]
                        errs["tps"].append(reproj_err(q, gu, gv, fx, cx, cy))
            return errs

        # ---- controllability curve + 4-fold CV ----
        vpts = np.where(valid)[0]
        methods = ["graph", "rigid", "nearest", "tps"]
        for K in (4, 8, 16):
            order = vpts[farthest_point_idx(x0[vpts], len(vpts))]   # FPS ordering of valid pts
            fold_meds = {m: [] for m in methods}
            for fold in range(4):
                # rotate which points are handles across folds (FPS-ordered, strided)
                handles = order[fold::4][:K] if len(order[fold::4]) >= K else order[:K]
                targets = [p for p in vpts if p not in set(handles.tolist())]
                e = predict_and_score(handles.tolist(), targets, methods)
                for m in methods:
                    if e[m]:
                        fold_meds[m].append(float(np.median(e[m])))
            results[f"K={K}"] = {m: (float(np.mean(fold_meds[m])) if fold_meds[m] else float("nan"),
                                     float(np.std(fold_meds[m])) if fold_meds[m] else float("nan"))
                                 for m in methods}

        print(f"\n=== CONTROLLABILITY (control-from-tracks) [{args.model_path}] ===")
        print(f"  held-out reprojection error (px), median ± std over 4 folds:")
        print(f"  {'K':>4} | {'graph':>12} | {'rigid':>12} | {'nearest':>12} | {'tps':>12}")
        for K in (4, 8, 16):
            r = results[f"K={K}"]
            print(f"  {K:>4} | " + " | ".join(f"{r[m][0]:6.2f}±{r[m][1]:4.2f}" for m in methods))
        with open(os.path.join(args.model_path, "control_results.json"), "w") as fp:
            json.dump(results, fp, indent=1)


def reproj_err(q_world, gu, gv, fx, cx, cy):
    """Project a world point (static identity cam) and return px error vs (gu,gv)."""
    if q_world[2] <= 1e-6:
        return float("nan")
    u = q_world[0] / q_world[2] * fx + cx
    v = q_world[1] / q_world[2] * fx + cy
    return float(np.hypot(u - gu, v - gv))


if __name__ == "__main__":
    main()
