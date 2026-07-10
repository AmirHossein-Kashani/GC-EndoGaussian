"""Anonymized demo video for the workshop submission (feedback #7).

Three segments, composed into one MP4 (no names, paths, or identifying text in frames):
  1. Dynamic reconstruction replay: ground truth | rendered reconstruction.
  2. Interactive drag at a fixed timestamp: a local node group is dragged with a ramping
     magnitude along two axes; right panel shows the per-pixel change heatmap (locality).
  3. Temporal consistency: the same edit held fixed while time sweeps the whole sequence;
     unedited | edited side by side.

Run inside a GPU job:
  python tools/make_demo_video.py --model_path output/endonerf/pulling_match3k \
      --configs arguments/endonerf/pulling_graph_match_3k.py --iteration 3000 \
      --out docs/supplementary/demo_pulling.mp4
"""
import os, sys, math
import numpy as np
import torch
import cv2
import imageio.v2 as iio
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render


def to8(img):
    return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()


def label(im, txt):
    cv2.rectangle(im, (0, 0), (im.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(im, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return im


def heat(diff, vmax):
    d = (diff / max(vmax, 1e-6)).clamp(0, 1).cpu().numpy()
    return cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1].copy()


@torch.no_grad()
def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=3000, type=int)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--edit_mag", type=float, default=0.06)   # peak, fraction of extent
    parser.add_argument("--radius_frac", type=float, default=0.10)
    parser.add_argument("--fps", type=int, default=15)
    args = get_combined_args(parser)
    if args.configs:
        from utils.config_loader import load_config
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, load_config(args.configs))

    dataset, pipe, hyper = model.extract(args), pipeline.extract(args), hp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    nd = gaussians._deformation.deformation_net.node_deform
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    vcams = scene.getVideoCameras()
    out_path = getattr(args, "out", None) or "docs/supplementary/demo.mp4"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    nx = nd.node_xyz
    M = nx.shape[0]
    ext = float((nx.max(0).values - nx.min(0).values).norm())
    center = nx.median(0).values
    sel = (nx - center).norm(dim=1) < args.radius_frac * ext
    axes = torch.eye(3, device=nx.device)
    zero = torch.zeros(M, 3, device=nx.device)
    frames = []

    # ---- segment 1: reconstruction replay (GT | render) ----
    for cam in vcams:
        nd.edit_translation = zero
        r = to8(render(cam, gaussians, pipe, bg, stage="fine")["render"])
        g = to8(cam.original_image.cuda())
        fr = np.hstack([g, r])
        frames.append(label(fr, "1/3  Dynamic reconstruction replay:  ground truth (L) | ours (R)"))

    # ---- segment 2: interactive drag at fixed t, magnitude ramp, two axes ----
    cam = vcams[len(vcams) // 3]
    nd.edit_translation = zero
    base = render(cam, gaussians, pipe, bg, stage="fine")["render"]
    vmax = 0.5
    STEPS = 60
    for phase, ax in [(0, 1), (1, 0)]:                       # y then x
        for k in range(STEPS):
            a = args.edit_mag * ext * 0.5 * (1 - math.cos(2 * math.pi * k / STEPS))
            disp = zero.clone(); disp[sel] = axes[ax] * a
            nd.edit_translation = disp
            img = render(cam, gaussians, pipe, bg, stage="fine")["render"]
            diff = (img - base).abs().sum(0)
            fr = np.hstack([to8(img), heat(diff, vmax)])
            frames.append(label(fr, f"2/3  Interactive drag ({'vertical' if ax==1 else 'lateral'}) | change heatmap (locality)"))

    # ---- segment 3: fixed edit held while time sweeps the sequence ----
    hold = zero.clone(); hold[sel] = axes[1] * (args.edit_mag * ext * 0.7)
    for cam in vcams:
        nd.edit_translation = zero
        un = to8(render(cam, gaussians, pipe, bg, stage="fine")["render"])
        nd.edit_translation = hold
        ed = to8(render(cam, gaussians, pipe, bg, stage="fine")["render"])
        fr = np.hstack([un, ed])
        frames.append(label(fr, "3/3  Same edit held over time:  unedited (L) | edited (R)"))
    nd.edit_translation = zero

    iio.mimwrite(out_path, frames, fps=args.fps, quality=8)
    print(f"wrote {out_path}: {len(frames)} frames @ {args.fps} fps "
          f"({len(frames)/args.fps:.0f}s), {frames[0].shape[1]}x{frames[0].shape[0]}")


if __name__ == "__main__":
    main()
