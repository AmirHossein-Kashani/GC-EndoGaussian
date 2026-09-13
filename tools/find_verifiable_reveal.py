"""Find where removing tissue exposes content the camera GENUINELY SAW at another moment.

This is the case that matters for an editing claim, and it is different from both
(a) the baseline's tool masking -- there the occluder is painted out of training, no edit involved,
and (b) peeling into a single-surface shell -- there nothing is behind, so you get background.

Here we ask, per Gaussian: is it *hidden* at the timestamp we want to edit, but *directly visible* at
some other timestamp because the tissue moved? Those Gaussians are observed content sitting behind
the current surface -- removing the surface re-shows them, and the frame where they were visible is
the ground-truth check.

Visibility test: render the depth map at frame s, project every Gaussian, and call it visible if its
own view-space depth agrees with the rendered surface depth at its pixel (within --tol, relative).

Score for a disc: number of Gaussians hidden at t but visible at >= 1 other sampled frame. The top
discs are peeled (their front layer removed) and written next to the frame that verifies them.

  python3 tools/find_verifiable_reveal.py --model_path output/endonerf/pulling_match3k \
      --configs arguments/endonerf/pulling_graph_match_3k.py --iteration 3000 \
      --frame 30 --out output/endonerf/pulling_match3k/verifiable
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


@torch.no_grad()
def project(pc, cam):
    xyz = deformed_xyz(pc, cam)
    ph = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
    clip = ph @ cam.full_proj_transform.to(xyz.device)
    ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-7)
    sx = (ndc[:, 0] * .5 + .5) * cam.image_width
    sy = (ndc[:, 1] * .5 + .5) * cam.image_height
    vz = (ph @ cam.world_view_transform.to(xyz.device))[:, 2]
    return sx, sy, vz


@torch.no_grad()
def visible_mask(pc, cam, pipe, bg, tol):
    """True for Gaussians lying on the rendered visible surface at this camera/time."""
    out = render(cam, gaussians_ref[0], pipe, bg, stage="fine")
    depth = out["depth"][0] if out["depth"].dim() == 3 else out["depth"]
    H, W = depth.shape[-2], depth.shape[-1]
    sx, sy, vz = project(pc, cam)
    xi = sx.long().clamp(0, W - 1)
    yi = sy.long().clamp(0, H - 1)
    surf = depth[yi, xi]
    inside = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H) & (surf > 0)
    return inside & ((vz - surf).abs() <= tol * surf.clamp_min(1e-6)), sx, sy, vz


gaussians_ref = [None]


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
    parser.add_argument("--sample", type=int, default=4, help="check every Nth frame for visibility")
    parser.add_argument("--tol", type=float, default=0.03, help="relative depth tolerance")
    parser.add_argument("--radius", type=float, default=70.0)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--topk", type=int, default=4)
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
    gaussians_ref[0] = gaussians
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    vcams = scene.getVideoCameras()
    N = len(vcams)
    t = min(args.frame, N - 1)
    cam = vcams[t]
    W, H = cam.image_width, cam.image_height

    vis_t, sx, sy, vz = visible_mask(gaussians, cam, pipe, bg, args.tol)
    hidden = ~vis_t
    seen_at = torch.zeros_like(vis_t, dtype=torch.int32)
    best_frame = torch.full_like(seen_at, -1)
    for s in range(0, N, args.sample):
        if s == t:
            continue
        v, _, _, _ = visible_mask(gaussians, vcams[s], pipe, bg, args.tol)
        newly = v & (best_frame < 0)
        best_frame[newly] = s
        seen_at += v.int()
    target = hidden & (seen_at > 0)
    print(f"[scan] frame {t}: {int(hidden.sum())} hidden gaussians, "
          f"{int(target.sum())} of them directly observed at another frame "
          f"({100*float(target.sum())/max(int(hidden.sum()),1):.1f}%)")

    cands = []
    for cy in range(args.stride, H - args.stride, args.stride):
        for cx in range(args.stride, W - args.stride, args.stride):
            disc = torch.hypot(sx - cx, sy - cy) < args.radius
            n_t = int((disc & target).sum())
            if n_t < 100:
                continue
            cands.append((n_t, cx, cy, int((disc & vis_t).sum())))
    cands.sort(reverse=True)
    print(f"[scan] {len(cands)} discs contain observed-but-hidden content; top:")
    for n_t, cx, cy, n_v in cands[:10]:
        print(f"   ({cx:3d},{cy:3d})  observed-hidden={n_t:5d}  front-surface={n_v:5d}")

    chosen = []
    for c in cands:
        if all((c[1] - o[1]) ** 2 + (c[2] - o[2]) ** 2 > (1.2 * args.radius) ** 2 for o in chosen):
            chosen.append(c)
        if len(chosen) >= args.topk:
            break

    orig = gaussians._opacity.clone()
    before = to8(render(cam, gaussians, pipe, bg, stage="fine")["render"])
    for rank, (n_t, cx, cy, n_v) in enumerate(chosen):
        disc = torch.hypot(sx - cx, sy - cy) < args.radius
        sel = disc & vis_t                                   # remove exactly the visible front layer
        gaussians._opacity[sel] = -50.0
        after = to8(render(cam, gaussians, pipe, bg, stage="fine")["render"])
        gaussians._opacity.copy_(orig)
        # the frame that best verifies this region
        fr = best_frame[disc & target]
        fr = fr[fr >= 0]
        vfr = int(torch.mode(fr).values) if fr.numel() else 0
        gtv = (vcams[vfr].original_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        d = np.abs(after.astype(np.float32) - before.astype(np.float32)).mean(-1)
        panel = np.hstack([
            label(dashed_circle(before.copy(), cx, cy, args.radius), f"before (frame {t})"),
            label(after, f"front tissue removed ({int(sel.sum())} gaussians)"),
            label(gtv.copy(), f"GT frame {vfr} - this content was really seen here", (60, 255, 60)),
        ])
        p = os.path.join(args.out, f"rev{rank}_x{cx}y{cy}.png")
        cv2.imwrite(p, panel[:, :, ::-1])
        print(f"[peel] rank {rank}: ({cx},{cy}) removed {int(sel.sum())} "
              f"observed-hidden={n_t} verify-frame={vfr} changed {(d>8).mean()*100:.1f}% -> {p}")


if __name__ == "__main__":
    main()
