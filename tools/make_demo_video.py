"""Anonymized demo video for the workshop submission — retraction-reveal focused.

Reproduces the paper's tissue-retraction edit (cutting scene, median region,
radius 0.06 x extent, lateral -x drag, peak magnitude 0.16 x extent — the same
parameters as figures/edit_reveal_cutting_*) as a video:

  1. Dynamic reconstruction replay: ground truth | rendered reconstruction.
  2. Tissue retraction: the circled node group is dragged laterally with a smooth
     ramp-hold-release, revealing the tissue behind it; right panel shows the
     per-pixel change heatmap (locality).
  3. Temporal consistency: the retraction held fixed while time sweeps the whole
     sequence; unedited | edited side by side.

Run inside a GPU job:
  python tools/make_demo_video.py --model_path output/endonerf/cutting_match \
      --configs arguments/endonerf/cutting_graph_match.py --iteration 6000 \
      --out docs/supplementary/demo_reveal.mp4
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


def project_px(pts, cam, W, H):
    ph = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
    clip = ph @ cam.full_proj_transform.to(pts.device)
    ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-7)
    return (ndc[:, 0] * .5 + .5) * W, (ndc[:, 1] * .5 + .5) * H


def dashed_circle(im, cx, cy, r, color=(60, 220, 60), seg=18):
    for k in range(seg):
        if k % 2 == 0:
            a0, a1 = 360.0 * k / seg, 360.0 * (k + 0.7) / seg
            cv2.ellipse(im, (int(cx), int(cy)), (int(r), int(r)), 0, a0, a1, color, 2, cv2.LINE_AA)
    return im


def ease(u):
    return 0.5 * (1 - math.cos(math.pi * u))          # smooth 0->1


@torch.no_grad()
def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=6000, type=int)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--peak_mag", type=float, default=0.16)   # figure's retraction magnitude
    parser.add_argument("--radius_frac", type=float, default=0.06)
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
    cam0 = scene.getTrainCameras()[0]                 # same view as the paper figure
    W, H = cam0.image_width, cam0.image_height
    out_path = getattr(args, "out", None) or "docs/supplementary/demo_reveal.mp4"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # handle selection identical to the reveal figure: median region, radius 0.06
    nx = nd.node_xyz
    M = nx.shape[0]
    ext = float((nx.max(0).values - nx.min(0).values).norm())
    center = nx.median(0).values
    sel = (nx - center).norm(dim=1) < args.radius_frac * ext
    direction = torch.tensor([-1.0, 0.0, 0.0], device=nx.device)   # lateral -x retraction
    zero = torch.zeros(M, 3, device=nx.device)

    # circle overlay geometry (projected grabbed nodes)
    px, py = project_px(nx[sel], cam0, W, H)
    ccx, ccy = float(px.mean()), float(py.mean())
    crad = float(torch.sqrt((px - px.mean())**2 + (py - py.mean())**2).max()) + 14

    frames = []

    # ---- segment 1: reconstruction replay (GT | render), every 2nd frame ----
    for cam in vcams[::2]:
        nd.edit_translation = zero
        r = to8(render(cam, gaussians, pipe, bg, stage="fine")["render"])
        g = to8(cam.original_image.cuda())
        frames.append(label(np.hstack([g, r]),
                            "1/3  Dynamic reconstruction replay:  ground truth (L) | ours (R)"))

    # ---- segment 2: retraction reveal, ramp-hold-release at the figure's view ----
    nd.edit_translation = zero
    base = render(cam0, gaussians, pipe, bg, stage="fine")["render"]
    vmax = 0.5
    RAMP, HOLD, REL = 45, 20, 30
    profile = [ease(k / RAMP) for k in range(RAMP)] + [1.0] * HOLD + \
              [1.0 - ease(k / REL) for k in range(REL)]
    for a in profile:
        disp = zero.clone(); disp[sel] = direction * (a * args.peak_mag * ext)
        nd.edit_translation = disp
        img = render(cam0, gaussians, pipe, bg, stage="fine")["render"]
        diff = (img - base).abs().sum(0)
        left = dashed_circle(to8(img), ccx, ccy, crad)
        frames.append(label(np.hstack([left, heat(diff, vmax)]),
                            "2/3  Retracting the circled tissue reveals the region behind it | change heatmap"))

    # ---- segment 3: retraction held while time sweeps the sequence ----
    hold = zero.clone(); hold[sel] = direction * (args.peak_mag * ext)
    for cam in vcams[::2]:
        nd.edit_translation = zero
        un = to8(render(cam, gaussians, pipe, bg, stage="fine")["render"])
        nd.edit_translation = hold
        ed = dashed_circle(to8(render(cam, gaussians, pipe, bg, stage="fine")["render"]), ccx, ccy, crad)
        frames.append(label(np.hstack([un, ed]),
                            "3/3  Retraction held over time:  unedited (L) | edited (R)"))
    nd.edit_translation = zero

    iio.mimwrite(out_path, frames, fps=args.fps, quality=8)
    print(f"wrote {out_path}: {len(frames)} frames @ {args.fps} fps "
          f"({len(frames)/args.fps:.0f}s), {frames[0].shape[1]}x{frames[0].shape[0]} | "
          f"{int(sel.sum())}/{M} nodes grabbed")


if __name__ == "__main__":
    main()
