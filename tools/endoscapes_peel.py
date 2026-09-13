"""Select a region, REMOVE the tissue in front, and see what is underneath.

This is a different edit from the drag-retraction in tools/endoscapes_reveal.py: instead of pushing
an organ aside, we delete the front-most layer of Gaussians inside a screen-space disc, so the
rasterizer composites whatever sits behind them. It answers "what is under this section?" directly.

How the selection works:
  1. Deform every Gaussian to the requested timestamp (same call the renderer makes).
  2. Project to the image and keep those landing inside the disc (--px,--py,--pr).
  3. Within that disc, sort by view-space depth and remove the front --front_frac of the depth
     *range* -- not all of them, otherwise you punch through to background instead of revealing
     the next layer.
  4. Removal = set _opacity to -50 (sigmoid -> 0). Note the renderer activates pc._opacity directly
     (gaussian_renderer/__init__.py:108), so this is exactly what the rasterizer sees. Restored
     afterwards; nothing on disk changes.

--steps > 1 peels progressively (a depth sweep), which also makes a good short video.

  python3 tools/endoscapes_peel.py --model_path output/endoscapes/seq83 \
      --configs arguments/endoscapes/seq83.py --iteration 3000 \
      --frame 20 --ref_frame 69 --px 300 --py 270 --pr 120 --front_frac 0.45 --steps 6 \
      --out output/endoscapes/seq83/peel
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


def to8(img):
    return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()


def label(im, txt, color=(255, 255, 255)):
    im = im.copy()
    cv2.rectangle(im, (0, 0), (im.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(im, txt, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return im


def dashed_circle(im, cx, cy, r, color=(60, 220, 60), seg=22, w=2):
    for k in range(seg):
        if k % 2 == 0:
            cv2.ellipse(im, (int(cx), int(cy)), (int(r), int(r)), 0,
                        360.0 * k / seg, 360.0 * (k + 0.7) / seg, color, w, cv2.LINE_AA)
    return im


def mark_region(im, args, color=(60, 220, 60)):
    im = im.copy()
    if getattr(args, "mask", None):
        mk = cv2.imread(args.mask, 0)
        if mk is not None:
            mk = cv2.resize(mk, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
            c, _ = cv2.findContours((mk > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im, c, -1, color, 2)
        return im
    if args.rect is not None:
        x0, y0, x1, y1 = [int(v) for v in args.rect]
        cv2.rectangle(im, (x0, y0), (x1, y1), color, 2)
        return im
    return dashed_circle(im, args.px, args.py, args.pr, color)


@torch.no_grad()
def deformed_xyz(pc, cam):
    """Positions at the camera's timestamp -- mirrors gaussian_renderer.render()."""
    means3D = pc.get_xyz
    time = torch.tensor(cam.time).to(means3D.device).repeat(means3D.shape[0], 1)
    dp = pc._deformation_table
    binding_idx = binding_w = None
    if getattr(pc, "use_node_graph", False) and pc.node_seeded:
        # Bindings are not persisted in the checkpoint -- render() rebuilds them on its first call
        # (gaussian_renderer/__init__.py:78). We may run before that, so rebuild here if needed.
        if pc._binding_idx.shape[0] != means3D.shape[0]:
            pc.compute_node_bindings()
        dev = means3D.device
        binding_idx = pc._binding_idx.to(dev)[dp.to(dev)]
        binding_w = pc._binding_w.to(dev)[dp.to(dev)]
    out = pc._deformation(means3D[dp], pc._scaling[dp], pc._rotation[dp], pc._opacity[dp],
                          time[dp], binding_idx, binding_w)
    final = means3D.clone()
    final[dp] = out[0]
    return final


@torch.no_grad()
def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=3000, type=int)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--frame", type=int, default=20)
    parser.add_argument("--ref_frame", type=int, default=69)
    parser.add_argument("--px", type=float, default=None, help="disc center x (circle mode)")
    parser.add_argument("--py", type=float, default=None, help="disc center y (circle mode)")
    parser.add_argument("--pr", type=float, default=120.0, help="disc radius (circle mode)")
    parser.add_argument("--rect", type=float, nargs=4, default=None,
                        metavar=("X0", "Y0", "X1", "Y1"),
                        help="select a rectangle instead of a disc; overrides --px/--py/--pr")
    parser.add_argument("--mask", type=str, default=None,
                        help="select by a mask PNG (white = the region to remove); overrides rect/disc. "
                             "Used only to define the selection at inference -- not during training.")
    parser.add_argument("--front_frac", type=float, default=0.45,
                        help="fraction of the in-disc depth RANGE removed from the front")
    parser.add_argument("--steps", type=int, default=1, help=">1 peels progressively")
    parser.add_argument("--tag", type=str, default="")
    args = get_combined_args(parser)
    # get_combined_args drops cmdline keys whose value is None when merging the saved cfg_args,
    # so optional selectors can vanish entirely -- restore them before use.
    for k in ("rect", "px", "py", "mask"):
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
    ref = vcams[min(args.ref_frame, len(vcams) - 1)]

    xyz = deformed_xyz(gaussians, cam)
    ph = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
    clip = ph @ cam.full_proj_transform.to(xyz.device)
    ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-7)
    sx = (ndc[:, 0] * .5 + .5) * cam.image_width
    sy = (ndc[:, 1] * .5 + .5) * cam.image_height
    view_z = (ph @ cam.world_view_transform.to(xyz.device))[:, 2]

    if getattr(args, "mask", None):
        mk = cv2.imread(args.mask, 0)
        if mk is None:
            sys.exit(f"ERROR: could not read mask {args.mask}")
        mk = cv2.resize(mk, (cam.image_width, cam.image_height), interpolation=cv2.INTER_NEAREST)
        mt = torch.from_numpy((mk > 127).astype(np.uint8)).to(sx.device)
        xi = sx.long().clamp(0, cam.image_width - 1)
        yi = sy.long().clamp(0, cam.image_height - 1)
        in_disc = mt[yi, xi] > 0
        region_txt = f"mask({os.path.basename(args.mask)})"
    elif args.rect is not None:
        x0, y0, x1, y1 = args.rect
        in_disc = (sx >= x0) & (sx <= x1) & (sy >= y0) & (sy <= y1)
        region_txt = f"rect({int(x0)},{int(y0)})-({int(x1)},{int(y1)})"
    else:
        if args.px is None or args.py is None:
            sys.exit("ERROR: give either --rect X0 Y0 X1 Y1, or --px and --py")
        in_disc = torch.hypot(sx - args.px, sy - args.py) < args.pr
        region_txt = f"disc({int(args.px)},{int(args.py)}) r={int(args.pr)}"
    n_disc = int(in_disc.sum())
    if n_disc == 0:
        sys.exit(f"ERROR: no Gaussians project inside {region_txt}")
    zd = view_z[in_disc]
    z0, z1 = float(torch.quantile(zd, 0.01)), float(torch.quantile(zd, 0.99))

    orig = gaussians._opacity.clone()
    before = to8(render(cam, gaussians, pipe, bg, stage="fine")["render"])

    peels, counts = [], []
    for s in range(1, args.steps + 1):
        frac = args.front_frac * s / args.steps
        zcut = z0 + (z1 - z0) * frac
        sel = in_disc & (view_z <= zcut)
        gaussians._opacity[sel] = -50.0                     # sigmoid -> 0 == removed
        peels.append(to8(render(cam, gaussians, pipe, bg, stage="fine")["render"]))
        counts.append(int(sel.sum()))
        gaussians._opacity.copy_(orig)
    after = peels[-1]

    gt_now = (cam.original_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gt_ref = (ref.original_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    diff = np.abs(after.astype(np.float32) - before.astype(np.float32)).mean(-1)
    changed = float((diff > 8).mean() * 100)

    panel = np.hstack([
        label(mark_region(before, args), f"before  (frame {args.frame})"),
        label(after, f"after removing front {args.front_frac:.0%}  ({counts[-1]} gaussians)"),
        label(gt_now, f"GT frame {args.frame} (region covered)"),
        label(gt_ref, f"GT frame {args.ref_frame} = what is really under", (60, 255, 60)),
    ])
    tag = args.tag or f"f{args.frame}_{region_txt.replace(' ','')}"
    out_png = os.path.join(args.out, f"peel_{tag}.png")
    cv2.imwrite(out_png, panel[:, :, ::-1])

    if args.steps > 1:
        fdir = os.path.join(args.out, f"frames_{tag}"); os.makedirs(fdir, exist_ok=True)
        for i, im in enumerate([before] + peels):
            cv2.imwrite(os.path.join(fdir, f"f{i:03d}.png"), im[:, :, ::-1])

    print(f"[peel] {region_txt} holds {n_disc} gaussians | removed {counts[-1]} "
          f"| changed {changed:.1f}% of pixels -> {out_png}")


if __name__ == "__main__":
    main()
