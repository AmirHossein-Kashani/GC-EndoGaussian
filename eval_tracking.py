"""Tracking-fidelity evaluation on SuPer: reprojection error of GT-annotated tissue points.

For each ground-truth tracked point we anchor it (at frame 0) to the nearest front-most canonical
Gaussian, then deform that Gaussian to every frame's time, project it, and compare to the GT 2D
track. Reports mean/median reprojection error (px) + a drift curve. Computed identically for vanilla
and graph models -> a fair head-to-head on how faithfully each tracks real tissue motion.

HONEST SCOPE: this measures *tracking fidelity* (reproducing observed motion), which is GT-backed,
NOT *controllability* under a user/robot control input (that needs the dVRK kinematics; see plan).

Usage: python eval_tracking.py --model_path output/endonerf/super_match \
         --configs arguments/endonerf/super_graph_match.py \
         --tracks data/super/v2_data/trial_3/rgb/trial_3_l_pts.npy --pad_v 16 --iteration 3000
"""
import os, sys
import numpy as np
import torch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render  # noqa: F401  (ensures CUDA ext import parity)


def bootstrap_ci(x, n=10000, alpha=0.05, stat=np.median, seed=0):
    """Percentile bootstrap CI of a statistic (default median) over a 1-D error population."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = np.array([stat(rng.choice(x, x.size, replace=True)) for _ in range(n)])
    return (float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2))))


@torch.no_grad()
def deformed_xyz(pc, t):
    """Replicate render()'s deform path to get per-Gaussian deformed world positions at time t."""
    means3D = pc.get_xyz
    N = means3D.shape[0]
    time = torch.full((N, 1), float(t), device=means3D.device)
    mask = pc._deformation_table
    scales, rots, opacity = pc._scaling, pc._rotation, pc._opacity
    binding_idx = binding_w = None
    if getattr(pc, "use_node_graph", False) and pc.node_seeded:
        if pc._binding_idx.shape[0] != N:
            pc.compute_node_bindings()
        binding_idx, binding_w = pc._binding_idx[mask], pc._binding_w[mask]
    out = means3D.clone()
    if mask.any():
        dxyz = pc._deformation(means3D[mask], scales[mask], rots[mask], opacity[mask],
                               time[mask], binding_idx, binding_w)[0]
        out[mask] = dxyz
    return out


@torch.no_grad()
def project(xyz, cam, W, H):
    """World xyz (N,3) -> pixel (u,v) (N,2) and camera-space depth (N,), via the camera matrices."""
    ph = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)          # (N,4)
    clip = ph @ cam.full_proj_transform.to(xyz.device)                 # (N,4) NDC homogeneous
    ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-7)
    u = (ndc[:, 0] * 0.5 + 0.5) * W
    v = (ndc[:, 1] * 0.5 + 0.5) * H
    z = (ph @ cam.world_view_transform.to(xyz.device))[:, 2]
    return torch.stack([u, v], dim=1), z


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=3000, type=int)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--tracks", type=str, required=True)
    parser.add_argument("--pad_v", type=int, default=16)              # v offset from 480->512 padding
    args = get_combined_args(parser)
    if args.configs:
        from utils.config_loader import load_config
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, load_config(args.configs))
    dataset, pipe, hyper = model.extract(args), pipeline.extract(args), hp.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyper)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        cams = scene.getVideoCameras()                                # all frames, temporal order
        N = len(cams)
        W, H = int(cams[0].image_width), int(cams[0].image_height)

        gt = np.load(args.tracks, allow_pickle=True).tolist()
        gt = gt["gt"] if isinstance(gt, dict) and "gt" in gt else gt
        keys = sorted(gt.keys())
        # align track frames to camera frames by index order
        T = min(len(keys), N)
        tracks = np.stack([np.asarray(gt[keys[f]], dtype=np.float64) for f in range(T)], 0)  # (T,P,3)
        tracks[:, :, 1] += args.pad_v                                  # padding offset on v
        P = tracks.shape[1]

        # --- anchor each track to a canonical Gaussian at frame 0 ---
        xyz0 = deformed_xyz(gaussians, cams[0].time)
        uv0, z0 = project(xyz0, cams[0], W, H)
        anchors = np.full(P, -1, dtype=np.int64)
        for p in range(P):
            gu, gv, vis = tracks[0, p]
            if vis == 0 or gu <= 0 or gv <= 0:
                continue
            d2 = (uv0[:, 0] - gu) ** 2 + (uv0[:, 1] - gv) ** 2
            d2 = d2 + (z0 < 0).float() * 1e9                          # prefer in-front Gaussians
            near = torch.topk(d2, k=min(15, d2.numel()), largest=False).indices  # 15 closest in 2D
            anchors[p] = int(near[torch.argmin(z0[near])].item())     # of those, the front-most

        # --- deform+project anchors per frame; reprojection error vs GT ---
        per_frame, all_err, frame0_err = [], [], []
        per_point = {p: [] for p in range(P)}                         # for paired stats across models
        for f in range(T):
            xyz = deformed_xyz(gaussians, cams[f].time)
            uv, _ = project(xyz, cams[f], W, H)
            errs = []
            for p in range(P):
                gu, gv, vis = tracks[f, p]
                if anchors[p] < 0 or vis == 0 or gu <= 0 or gv <= 0:
                    continue
                pu, pv = uv[anchors[p]].tolist()
                e = float(np.hypot(pu - gu, pv - gv))
                errs.append(e); all_err.append(e); per_point[p].append(e)
                if f == 0:
                    frame0_err.append(e)
            per_frame.append((f, float(np.mean(errs)) if errs else float("nan")))

        all_err = np.array(all_err)
        lo, hi = bootstrap_ci(all_err, stat=np.median)
        per_point_mean = [float(np.mean(per_point[p])) for p in range(P) if per_point[p]]
        res = {
            "RPE_mean_px": float(all_err.mean()),
            "RPE_median_px": float(np.median(all_err)),
            "RPE_median_CI95": [lo, hi],
            "RPE_frame0_px": float(np.mean(frame0_err)) if frame0_err else float("nan"),
            "n_points": int((anchors >= 0).sum()),
            "n_frames": int(T),
        }
        print(f"\n=== TRACKING FIDELITY [{args.model_path}] ===")
        print(f"  RPE_mean_px:   {res['RPE_mean_px']:.4f}")
        print(f"  RPE_median_px: {res['RPE_median_px']:.4f}  (95% CI [{lo:.3f}, {hi:.3f}])")
        print(f"  RPE_frame0_px: {res['RPE_frame0_px']:.4f}   n_points={res['n_points']} n_frames={res['n_frames']}")
        print("  drift (RPE px @ 0,T/4,T/2,3T/4,T-1): " +
              ", ".join(f"{per_frame[i][1]:.2f}" for i in [0, T // 4, T // 2, 3 * T // 4, T - 1]))
        import json
        with open(os.path.join(args.model_path, "tracking_results.json"), "w") as fp:
            json.dump({"summary": res, "per_frame": per_frame, "per_point_mean": per_point_mean}, fp, indent=1)


if __name__ == "__main__":
    main()
