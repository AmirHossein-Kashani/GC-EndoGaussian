"""Quantitative edit evaluation for GC-EndoGaussian (paper §"edit metrics").

The paper's editing interface is only demonstrated qualitatively; this script measures it.
For a trained model with a seeded control graph it reports, over a sweep of handle draws,
edit axes, and edit magnitudes:

  1. handle_fidelity   — achieved displacement of strongly-bound Gaussians / commanded
                         magnitude (1.0 = tissue follows the handle exactly).
  2. locality curve    — median displacement (fraction of commanded) binned by 3D distance
                         to the nearest handle node, in units of the node-cloud extent.
  3. leakage_3d        — mean displacement (fraction of commanded) of Gaussians farther
                         than 2x the handle radius from the handle set.
  4. leakage_px        — fraction of per-pixel image change that falls OUTSIDE the dilated
                         projected footprint of the handle-bound Gaussians.
  5. foldover_rate     — fraction of Gaussian KNN pairs whose relative orientation inverts
                         under the edit (local self-intersection proxy).
  6. strain_p95        — 95th-percentile relative stretch |d_i - d_j| / rest_ij over KNN
                         pairs (edit-smoothness proxy).
  7. latency_ms        — wall-clock to re-pose + re-render one 640x512 frame after a new
                         edit vector is set (interactivity of the edit loop).
  8. energy split      — over video timestamps: fraction of learned motion carried by the
                         node field vs the dense per-Gaussian residual (node_hybrid models).

Usage:
  python eval_edit.py --model_path output/endonerf/pulling_match3k \
      --configs arguments/endonerf/pulling_graph_match_3k.py --iteration 3000 \
      --out_json output/endonerf/pulling_match3k/edit_metrics.json
"""
import os, sys, json, time
import numpy as np
import torch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render
from eval_tracking import deformed_xyz


def knn_pairs(pts, k=8, sample=20000, seed=0):
    """Subsampled symmetric KNN pairs (i, j) over deformed positions."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = pts.shape[0]
    idx = torch.randperm(n, generator=g)[: min(sample, n)].to(pts.device)
    sub = pts[idx]                                           # (S,3)
    d = torch.cdist(sub, sub)                                # (S,S)
    d.fill_diagonal_(float("inf"))
    nbr = d.topk(k, largest=False).indices                   # (S,k)
    src = torch.arange(sub.shape[0], device=pts.device).unsqueeze(1).expand(-1, k)
    return idx, src.reshape(-1), nbr.reshape(-1)


@torch.no_grad()
def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=3000, type=int)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--n_handles", type=int, default=8)      # handle draws
    parser.add_argument("--radius_frac", type=float, default=0.10)
    parser.add_argument("--mags", type=float, nargs="+", default=[0.01, 0.02, 0.04, 0.08])
    parser.add_argument("--latency_reps", type=int, default=30)
    parser.add_argument("--times", type=float, nargs="*", default=[],
                        help="extra timestamps for the temporal sweep (fidelity/leak_px/foldover per t)")
    args = get_combined_args(parser)
    if args.configs:
        from utils.config_loader import load_config
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, load_config(args.configs))

    dataset, pipe, hyper = model.extract(args), pipeline.extract(args), hp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    if not getattr(gaussians, "use_node_graph", False) or not gaussians.node_seeded:
        print("ERROR: model has no seeded control graph."); sys.exit(1)

    nd = gaussians._deformation.deformation_net.node_deform
    dnet = gaussians._deformation.deformation_net
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    cam = scene.getTestCameras()[0]
    t = cam.time
    W, H = cam.image_width, cam.image_height

    nx = nd.node_xyz                                          # (M,3)
    M = nx.shape[0]
    ext = float((nx.max(0).values - nx.min(0).values).norm())
    handle_radius = args.radius_frac * ext

    # baseline pose at the eval timestamp
    nd.edit_translation = torch.zeros(M, 3, device=nx.device)
    p0 = deformed_xyz(gaussians, t)                           # (N,3)
    base_img = render(cam, gaussians, pipe, bg, stage="fine")["render"]
    N = p0.shape[0]

    # Gaussian->handle geometry needs bindings on the CURRENT gaussian set
    bidx, bw = gaussians._binding_idx, gaussians._binding_w   # (N,K), (N,K)

    # KNN pairs on the unedited pose (shared across all edits)
    sub_idx, src, nbr = knn_pairs(p0)
    rest_vec = p0[sub_idx][src] - p0[sub_idx][nbr]
    rest_len = rest_vec.norm(dim=1).clamp_min(1e-6)

    rng = np.random.default_rng(0)
    centers = rng.choice(M, size=args.n_handles, replace=False)
    axes = torch.eye(3, device=nx.device)

    per_edit = []
    curve_bins = np.linspace(0.0, 0.6, 13)                    # distance bins (units of extent)
    curve_acc = [[] for _ in range(len(curve_bins) - 1)]

    for hi, c in enumerate(centers):
        sel = (nx - nx[int(c)]).norm(dim=1) < handle_radius   # (M,) handle node set
        if int(sel.sum()) < 3:
            continue
        # per-Gaussian: summed binding weight to handle nodes, distance to nearest handle node
        handle_ids = torch.nonzero(sel).squeeze(1)
        w_handle = torch.zeros(N, device=nx.device)
        for kk in range(bidx.shape[1]):
            w_handle += bw[:, kk] * torch.isin(bidx[:, kk], handle_ids).float()
        dist_h = torch.cdist(p0, nx[sel]).min(dim=1).values / ext
        core = w_handle > 0.5                                 # strongly-bound Gaussians
        far = dist_h > 2.0 * args.radius_frac                 # outside 2x handle radius

        # projected ROI for pixel leakage: core Gaussians' footprint, dilated
        uv_h = None
        if core.any():
            ph = torch.cat([p0[core], torch.ones_like(p0[core][:, :1])], dim=1)
            clip = ph @ cam.full_proj_transform.to(p0.device)
            ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-7)
            uv_h = torch.stack([(ndc[:, 0] * .5 + .5) * W, (ndc[:, 1] * .5 + .5) * H], 1)

        axis = axes[hi % 3]
        for mag_frac in args.mags:
            mag = mag_frac * ext
            disp = torch.zeros(M, 3, device=nx.device)
            disp[sel] = axis * mag
            nd.edit_translation = disp
            p1 = deformed_xyz(gaussians, t)
            d = (p1 - p0).norm(dim=1)                         # (N,) achieved displacement

            fidelity = float((d[core] / mag).mean()) if core.any() else float("nan")
            leak3d = float((d[far] / mag).mean()) if far.any() else 0.0

            # locality curve (normalized displacement vs distance)
            dh = dist_h.cpu().numpy(); dn = (d / mag).cpu().numpy()
            for b in range(len(curve_bins) - 1):
                m = (dh >= curve_bins[b]) & (dh < curve_bins[b + 1])
                if m.any():
                    curve_acc[b].append(float(np.median(dn[m])))

            # foldover + strain over KNN pairs
            dvec = (p1 - p0)[sub_idx]
            new_vec = rest_vec + (dvec[src] - dvec[nbr])
            fold = float((torch.einsum("ij,ij->i", new_vec, rest_vec) < 0).float().mean())
            strain = ((dvec[src] - dvec[nbr]).norm(dim=1) / rest_len)
            strain_p95 = float(torch.quantile(strain, 0.95))

            # pixel leakage at the largest magnitude only (rendering is the slow part)
            leak_px = None
            if mag_frac == args.mags[-1] and uv_h is not None:
                img = render(cam, gaussians, pipe, bg, stage="fine")["render"]
                diff = (img - base_img).abs().sum(0)          # (H,W)
                roi = torch.zeros(H, W, device=diff.device, dtype=torch.bool)
                u = uv_h[:, 0].round().long().clamp(0, W - 1)
                v = uv_h[:, 1].round().long().clamp(0, H - 1)
                roi[v, u] = True
                pad = 15                                       # dilate footprint
                roi = torch.nn.functional.max_pool2d(roi[None, None].float(),
                                                     2 * pad + 1, 1, pad)[0, 0].bool()
                total = float(diff.sum())
                leak_px = float(diff[~roi].sum() / total) if total > 0 else 0.0

            per_edit.append(dict(handle=int(c), mag_frac=mag_frac, fidelity=fidelity,
                                 leak3d=leak3d, foldover=fold, strain_p95=strain_p95,
                                 leak_px=leak_px, n_core=int(core.sum())))

    nd.edit_translation = torch.zeros(M, 3, device=nx.device)

    # ---- edit-update latency: set new edit vector -> re-render one frame ----
    disp = torch.zeros(M, 3, device=nx.device)
    sel = (nx - nx[int(centers[0])]).norm(dim=1) < handle_radius
    torch.cuda.synchronize()
    times = []
    for r in range(args.latency_reps):
        disp[sel] = axes[r % 3] * (0.02 + 0.001 * r) * ext    # a fresh edit each rep
        t0 = time.perf_counter()
        nd.edit_translation = disp
        _ = render(cam, gaussians, pipe, bg, stage="fine")["render"]
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    latency_ms = float(np.median(times[5:]))                  # skip warmup
    nd.edit_translation = torch.zeros(M, 3, device=nx.device)

    # ---- node-field vs residual energy split over video timestamps ----
    energy = None
    if getattr(dnet.args, "node_hybrid", False):
        x0 = deformed_xyz(gaussians, 0.0)
        ts = np.linspace(0.0, 1.0, 20)
        e_node, e_res = 0.0, 0.0
        for tt in ts:
            p_full = deformed_xyz(gaussians, float(tt))
            dnet.args.node_hybrid = False                      # residual off
            p_node = deformed_xyz(gaussians, float(tt))
            dnet.args.node_hybrid = True
            e_node += float((p_node - x0).pow(2).sum())        # motion rel. to t=0 pose
            e_res += float((p_full - p_node).pow(2).sum())
        energy = dict(node_frac=e_node / max(e_node + e_res, 1e-12),
                      residual_frac=e_res / max(e_node + e_res, 1e-12))

    # ---- temporal sweep: same interface metrics at multiple timestamps ----
    # (reviewer request: does edit behavior hold across the deforming sequence?)
    per_time = []
    req_times = list(getattr(args, "times", []) or [])
    if req_times:
        vcams = scene.getVideoCameras()
        mag_frac_t = 0.04                                     # one mid magnitude per timestamp
        for tt in req_times:
            cam_t = min(vcams, key=lambda c: abs(c.time - tt))
            t_used = cam_t.time
            nd.edit_translation = torch.zeros(M, 3, device=nx.device)
            p0t = deformed_xyz(gaussians, t_used)
            base_t = render(cam_t, gaussians, pipe, bg, stage="fine")["render"]
            sub_i, src_t, nbr_t = knn_pairs(p0t)
            rest_t = p0t[sub_i][src_t] - p0t[sub_i][nbr_t]
            fids, leaks, folds = [], [], []
            for hi, c in enumerate(centers[:4]):              # 4 handle draws per timestamp
                sel = (nx - nx[int(c)]).norm(dim=1) < handle_radius
                if int(sel.sum()) < 3:
                    continue
                handle_ids = torch.nonzero(sel).squeeze(1)
                w_handle = torch.zeros(N, device=nx.device)
                for kk in range(bidx.shape[1]):
                    w_handle += bw[:, kk] * torch.isin(bidx[:, kk], handle_ids).float()
                core = w_handle > 0.5
                mag = mag_frac_t * ext
                disp = torch.zeros(M, 3, device=nx.device)
                disp[sel] = axes[hi % 3] * mag
                nd.edit_translation = disp
                p1t = deformed_xyz(gaussians, t_used)
                d = (p1t - p0t).norm(dim=1)
                if core.any():
                    fids.append(float((d[core] / mag).mean()))
                dvec = (p1t - p0t)[sub_i]
                new_vec = rest_t + (dvec[src_t] - dvec[nbr_t])
                folds.append(float((torch.einsum("ij,ij->i", new_vec, rest_t) < 0).float().mean()))
                if core.any():
                    ph = torch.cat([p0t[core], torch.ones_like(p0t[core][:, :1])], dim=1)
                    clip = ph @ cam_t.full_proj_transform.to(p0t.device)
                    ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-7)
                    u = ((ndc[:, 0] * .5 + .5) * W).round().long().clamp(0, W - 1)
                    v = ((ndc[:, 1] * .5 + .5) * H).round().long().clamp(0, H - 1)
                    img = render(cam_t, gaussians, pipe, bg, stage="fine")["render"]
                    diff = (img - base_t).abs().sum(0)
                    roi = torch.zeros(H, W, device=diff.device, dtype=torch.bool)
                    roi[v, u] = True
                    pad = 15
                    roi = torch.nn.functional.max_pool2d(roi[None, None].float(),
                                                         2 * pad + 1, 1, pad)[0, 0].bool()
                    total = float(diff.sum())
                    leaks.append(float(diff[~roi].sum() / total) if total > 0 else 0.0)
            nd.edit_translation = torch.zeros(M, 3, device=nx.device)
            per_time.append(dict(t=float(t_used),
                                 fidelity=float(np.median(fids)) if fids else None,
                                 leak_px=float(np.median(leaks)) if leaks else None,
                                 foldover=float(np.median(folds)) if folds else None,
                                 n=len(fids)))
            print(f"[t={t_used:.2f}] fidelity={per_time[-1]['fidelity']} "
                  f"leak_px={per_time[-1]['leak_px']} foldover={per_time[-1]['foldover']}")

    # ---- aggregate ----
    def agg(key, mask_fn=lambda e: True):
        vals = [e[key] for e in per_edit if e[key] is not None and mask_fn(e) and np.isfinite(e[key])]
        return dict(median=float(np.median(vals)), p25=float(np.percentile(vals, 25)),
                    p75=float(np.percentile(vals, 75)), n=len(vals)) if vals else None

    curve = [dict(bin_lo=float(curve_bins[b]), bin_hi=float(curve_bins[b + 1]),
                  median=float(np.median(curve_acc[b])) if curve_acc[b] else None,
                  n=len(curve_acc[b])) for b in range(len(curve_bins) - 1)]

    out = dict(model_path=args.model_path, iteration=args.iteration,
               n_nodes=M, n_gaussians=N, extent=ext, radius_frac=args.radius_frac,
               mags=args.mags, per_edit=per_edit, locality_curve=curve,
               handle_fidelity=agg("fidelity"), leak3d=agg("leak3d"),
               leak_px=agg("leak_px"), foldover=agg("foldover"),
               strain_p95=agg("strain_p95"), latency_ms=latency_ms, energy_split=energy,
               per_time=per_time)

    print("\n================ EDIT METRICS ================")
    print(f"model: {args.model_path} @ {args.iteration} | nodes={M} gaussians={N}")
    print(f"handle_fidelity (median [IQR]): {out['handle_fidelity']}")
    print(f"leak3d  beyond 2x radius:       {out['leak3d']}")
    print(f"leak_px outside ROI:            {out['leak_px']}")
    print(f"foldover_rate:                  {out['foldover']}")
    print(f"strain_p95:                     {out['strain_p95']}")
    print(f"edit->render latency (ms):      {latency_ms:.2f}")
    print(f"energy_split:                   {energy}")
    print("==============================================\n")

    # get_combined_args drops cmdline args left at their None default, so read defensively
    out_json = getattr(args, "out_json", None) or os.path.join(args.model_path, "edit_metrics.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", out_json)


if __name__ == "__main__":
    main()
