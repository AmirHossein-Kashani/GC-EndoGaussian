"""Retract an organ at inference time and compare what is revealed against a real observation.

Endoscapes clip 83 gives something the EndoNeRF figures could not: a *ground-truth reference* for an
edit. Late in the clip (frame index 69 = 83_65775) the gallbladder is retracted and the dissected
space beneath it is exposed; in earlier frames that same region is covered. So we can retract the
organ at an early timestamp and check the revealed content against a frame where it was genuinely
observed.

Handles are picked in IMAGE space -- you name a pixel on the organ and a radius, and the control
nodes whose projection lands in that disc are grabbed. That is far easier to aim than the 3D median
cluster used by tools/make_edit_reveal_seq.py.

  python3 tools/endoscapes_reveal.py --model_path output/endoscapes/seq83 \
      --configs arguments/endoscapes/seq83.py --iteration 3000 \
      --frame 20 --ref_frame 69 --px 300 --py 200 --pr 110 --drag -y --peak 0.18 \
      --out output/endoscapes/seq83/reveal
"""
import os, sys, math
from argparse import ArgumentParser
import torch
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render

AXIS = {"x": 0, "y": 1, "z": 2}


def to8(img):
    return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()


def project(pts, cam):
    ph = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
    clip = ph @ cam.full_proj_transform.to(pts.device)
    ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-7)
    return ((ndc[:, 0] * .5 + .5) * cam.image_width,
            (ndc[:, 1] * .5 + .5) * cam.image_height)


def label(im, txt, color=(255, 255, 255)):
    cv2.rectangle(im, (0, 0), (im.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(im, txt, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    return im


def dashed_circle(im, cx, cy, r, color=(60, 220, 60), seg=20, w=2):
    for k in range(seg):
        if k % 2 == 0:
            cv2.ellipse(im, (int(cx), int(cy)), (int(r), int(r)), 0,
                        360.0 * k / seg, 360.0 * (k + 0.72) / seg, color, w, cv2.LINE_AA)
    return im


@torch.no_grad()
def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=3000, type=int)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--frame", type=int, default=20, help="timestamp to edit (organ covering)")
    parser.add_argument("--ref_frame", type=int, default=69, help="frame where the region is exposed")
    parser.add_argument("--px", type=float, required=True, help="handle center x in pixels")
    parser.add_argument("--py", type=float, required=True, help="handle center y in pixels")
    parser.add_argument("--pr", type=float, default=110.0, help="handle radius in pixels")
    parser.add_argument("--drag", type=str, default="-y")
    parser.add_argument("--peak", type=float, default=0.18, help="fraction of node-cloud extent")
    parser.add_argument("--steps", type=int, default=1, help=">1 renders a ramp for a video")
    args = get_combined_args(parser)
    if args.configs:
        from utils.config_loader import load_config
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, load_config(args.configs))

    dataset, pipe, hyper = model.extract(args), pipeline.extract(args), hp.extract(args)
    os.makedirs(args.out, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree, hyper)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    if not getattr(gaussians, "use_node_graph", False) or not gaussians.node_seeded:
        sys.exit("ERROR: model has no seeded control graph; editing requires use_node_graph.")
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    nd = gaussians._deformation.deformation_net.node_deform
    vcams = scene.getVideoCameras()
    cam = vcams[min(args.frame, len(vcams) - 1)]
    ref = vcams[min(args.ref_frame, len(vcams) - 1)]

    nx = nd.node_xyz
    M = nx.shape[0]
    ext = float((nx.max(0).values - nx.min(0).values).norm())

    # --- select the handle by projected pixel position ---
    sx, sy = project(nx, cam)
    sel = (torch.hypot(sx - args.px, sy - args.py) < args.pr)
    n_sel = int(sel.sum())
    if n_sel == 0:
        sys.exit(f"ERROR: no nodes project inside the disc at ({args.px},{args.py}) r={args.pr}")

    sign = -1.0 if args.drag[0] == "-" else 1.0
    direction = torch.zeros(3, device=nx.device); direction[AXIS[args.drag[-1]]] = sign
    zero = torch.zeros(M, 3, device=nx.device)

    # --- render: unedited, edited, and the real observation of the exposed region ---
    nd.edit_translation = zero
    before = to8(render(cam, gaussians, pipe, bg, stage="fine")["render"])

    edited = []
    for s in range(1, args.steps + 1):
        mag = args.peak * ext * (s / args.steps)
        disp = zero.clone(); disp[sel] = direction * mag
        nd.edit_translation = disp
        edited.append(to8(render(cam, gaussians, pipe, bg, stage="fine")["render"]))
    nd.edit_translation = zero
    after = edited[-1]

    gt_ref = (ref.original_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()

    # difference map: what the edit actually changed
    diff = np.abs(after.astype(np.float32) - before.astype(np.float32)).mean(-1)
    heat = cv2.applyColorMap(np.clip(diff / max(diff.max(), 1e-6) * 255, 0, 255).astype(np.uint8),
                             cv2.COLORMAP_INFERNO)[:, :, ::-1].copy()

    b = dashed_circle(before.copy(), args.px, args.py, args.pr)
    panel = np.hstack([
        label(b, f"before  (frame {args.frame}, organ covering)"),
        label(after.copy(), f"after retraction  (drag {args.drag}, {args.peak:.2f} extent)"),
        label(heat, "edit magnitude"),
        label(gt_ref, f"REAL observation (frame {args.ref_frame})", (60, 255, 60)),
    ])
    out_png = os.path.join(args.out, f"reveal_f{args.frame}_ref{args.ref_frame}.png")
    cv2.imwrite(out_png, panel[:, :, ::-1])

    if args.steps > 1:
        fdir = os.path.join(args.out, "frames"); os.makedirs(fdir, exist_ok=True)
        for i, im in enumerate([before] + edited):
            cv2.imwrite(os.path.join(fdir, f"f{i:04d}.png"), im[:, :, ::-1])

    print(f"[reveal] {n_sel}/{M} nodes grabbed at px=({args.px:.0f},{args.py:.0f}) r={args.pr:.0f}")
    print(f"[reveal] edit changed {(diff > 8).mean()*100:.1f}% of pixels -> {out_png}")


if __name__ == "__main__":
    main()
