"""Find where the reconstruction actually has TWO depth-separated layers, then peel the best ones.

Motivation: hand-picking a region to remove gives weak results, because most of an endoscopic
reconstruction is a single surface -- removing the front of it just exposes background. A convincing
"remove this tissue and see what is behind" example needs a place where the model genuinely holds a
second layer behind the first. This scans for that instead of guessing.

Method: deform all Gaussians to timestamp t, project them, and slide a disc over the image. Inside
each disc, split the Gaussians' view-space depths into two clusters (1-D 2-means). Score the disc by

    separation = (mean_back - mean_front) / (spread_front + spread_back)

requiring both clusters to hold a real share of the mass (--min_mass) so a handful of floaters cannot
win. High separation = a distinct back layer sitting behind a distinct front layer. The top regions
are then peeled (front cluster removed) and written as before/after panels.

  python3 tools/find_layered_regions.py --model_path output/endonerf/pulling_match3k \
      --configs arguments/endonerf/pulling_graph_match_3k.py --iteration 3000 \
      --frame 30 --out output/endonerf/pulling_match3k/layered
"""
import os, sys
from argparse import ArgumentParser
import torch
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render
from tools.endoscapes_peel import deformed_xyz, to8, label, dashed_circle


def two_means(z, iters=25):
    """1-D 2-means; returns (front_mean, back_mean, front_std, back_std, front_mask)."""
    lo, hi = z.min(), z.max()
    c0, c1 = lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)
    for _ in range(iters):
        m = z < 0.5 * (c0 + c1)
        if m.sum() == 0 or (~m).sum() == 0:
            break
        c0, c1 = z[m].mean(), z[~m].mean()
    m = z < 0.5 * (c0 + c1)
    if m.sum() < 2 or (~m).sum() < 2:
        return None
    return float(c0), float(c1), float(z[m].std()), float(z[~m].std()), m


@torch.no_grad()
def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=3000, type=int)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--frame", type=int, default=30)
    parser.add_argument("--radius", type=float, default=70.0)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--min_pts", type=int, default=400)
    parser.add_argument("--min_mass", type=float, default=0.18, help="min share in each layer")
    parser.add_argument("--topk", type=int, default=5)
    args = get_combined_args(parser)
    for k in ("rect", "px", "py"):
        if not hasattr(args, k):
            setattr(args, k, None)
    if args.configs:
        from utils.config_loader import load_config
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, load_config(args.configs))

    dataset, pipe, hyper = model.extract(args), pipeline.extract(args), hp.extract(args)
    os.makedirs(args.out, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree, hyper)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    vcams = scene.getVideoCameras()
    cam = vcams[min(args.frame, len(vcams) - 1)]
    W, H = cam.image_width, cam.image_height

    xyz = deformed_xyz(gaussians, cam)
    ph = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
    clip = ph @ cam.full_proj_transform.to(xyz.device)
    ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-7)
    sx = (ndc[:, 0] * .5 + .5) * W
    sy = (ndc[:, 1] * .5 + .5) * H
    vz = (ph @ cam.world_view_transform.to(xyz.device))[:, 2]

    cands = []
    for cy in range(args.stride, H - args.stride, args.stride):
        for cx in range(args.stride, W - args.stride, args.stride):
            sel = torch.hypot(sx - cx, sy - cy) < args.radius
            n = int(sel.sum())
            if n < args.min_pts:
                continue
            z = vz[sel]
            r = two_means(z)
            if r is None:
                continue
            c0, c1, s0, s1, fm = r
            mass = float(fm.float().mean())
            if mass < args.min_mass or mass > 1 - args.min_mass:
                continue
            sep = (c1 - c0) / max(s0 + s1, 1e-6)
            cands.append((sep, cx, cy, n, mass, c0, c1))

    cands.sort(reverse=True)
    print(f"[scan] {len(cands)} candidate regions with two layers (frame {args.frame})")
    for sep, cx, cy, n, mass, c0, c1 in cands[:12]:
        print(f"   ({cx:3d},{cy:3d})  sep={sep:5.2f}  n={n:6d}  front={mass*100:4.1f}%  "
              f"z {c0:.3f} -> {c1:.3f}")

    # peel the top-K, suppressing near-duplicates
    chosen = []
    for c in cands:
        if all((c[1] - o[1]) ** 2 + (c[2] - o[2]) ** 2 > (1.2 * args.radius) ** 2 for o in chosen):
            chosen.append(c)
        if len(chosen) >= args.topk:
            break

    orig = gaussians._opacity.clone()
    before = to8(render(cam, gaussians, pipe, bg, stage="fine")["render"])
    gt = (cam.original_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    for rank, (sep, cx, cy, n, mass, c0, c1) in enumerate(chosen):
        sel = (torch.hypot(sx - cx, sy - cy) < args.radius) & (vz < 0.5 * (c0 + c1))
        gaussians._opacity[sel] = -50.0
        after = to8(render(cam, gaussians, pipe, bg, stage="fine")["render"])
        gaussians._opacity.copy_(orig)
        d = np.abs(after.astype(np.float32) - before.astype(np.float32)).mean(-1)
        heat = cv2.applyColorMap(np.clip(d / max(d.max(), 1e-6) * 255, 0, 255).astype(np.uint8),
                                 cv2.COLORMAP_INFERNO)[:, :, ::-1].copy()
        panel = np.hstack([
            label(dashed_circle(before.copy(), cx, cy, args.radius), f"before (frame {args.frame})"),
            label(after, f"front layer removed ({int(sel.sum())} gaussians, sep={sep:.2f})"),
            label(heat, "what changed"),
            label(gt.copy(), f"ground truth frame {args.frame}", (60, 255, 60)),
        ])
        p = os.path.join(args.out, f"layer{rank}_x{cx}y{cy}.png")
        cv2.imwrite(p, panel[:, :, ::-1])
        print(f"[peel] rank {rank}: ({cx},{cy}) sep={sep:.2f} removed {int(sel.sum())} "
              f"changed {(d > 8).mean()*100:.1f}% -> {p}")


if __name__ == "__main__":
    main()
