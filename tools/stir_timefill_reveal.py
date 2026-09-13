"""Removal edit + TEMPORAL COMPLETION: re-show the hidden layer at the timestamp it was observed.

The plain peel (tools/endoscapes_peel.py) removes the front cluster and lets whatever is behind
composite through. On STIR that reveals a dark smear: the occluded bed is photometrically
unconstrained at the edit timestamp, so the optimizer parks those Gaussians arbitrarily until the
frames where they are actually seen (probes: the smear is NOT organ residue -- it survives
front_frac 0.95/0.99 -- and neither tool masking nor time-smoothness regularisation removes it).

This tool therefore completes the reveal *from the model itself*: the revealed region is rendered at
--ref_frame, the timestamp where the bed genuinely was observed. The camera is static, so the two
renders are pixel-aligned, and the bed is static, so its ref-time geometry IS its edit-time
geometry. No ground-truth pixels are used anywhere; the composite is model output at two timestamps
of the same canonical Gaussian set. GT at --ref_frame is shown alongside purely for verification,
and the revealed region is scored against it (PSNR raw removal vs PSNR time-filled).

  python3 tools/stir_timefill_reveal.py --model_path output/stir/seq23_6k_tm \
      --configs arguments/stir/seq23_6k_tm.py --iteration 6000 \
      --frame 10 --ref_frame 170 --mask data/endonerf/stir06_seq23/organ_mask_f10.png \
      --front_frac 0.9 --steps 12 --out output/stir/seq23_6k_tm/timefill
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
from tools.endoscapes_peel import deformed_xyz, to8, label, mark_region


def psnr_np(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))


@torch.no_grad()
def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=6000, type=int)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--ref_frame", type=int, required=True)
    parser.add_argument("--mask", type=str, required=True)
    parser.add_argument("--front_frac", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--fade", type=int, default=8, help="crossfade frames from raw hole to time-filled")
    parser.add_argument("--feather", type=int, default=15, help="mask edge feather in px")
    parser.add_argument("--tag", type=str, default="")
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
    ref = vcams[min(args.ref_frame, len(vcams) - 1)]

    # ---- selection at the edit timestamp --------------------------------------------------------
    xyz = deformed_xyz(gaussians, cam)
    ph = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
    clip = ph @ cam.full_proj_transform.to(xyz.device)
    ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-7)
    sx = (ndc[:, 0] * .5 + .5) * cam.image_width
    sy = (ndc[:, 1] * .5 + .5) * cam.image_height
    view_z = (ph @ cam.world_view_transform.to(xyz.device))[:, 2]

    mk = cv2.imread(args.mask, 0)
    if mk is None:
        sys.exit(f"ERROR: could not read mask {args.mask}")
    mk = cv2.resize(mk, (cam.image_width, cam.image_height), interpolation=cv2.INTER_NEAREST)
    mt = torch.from_numpy((mk > 127).astype(np.uint8)).to(sx.device)
    xi = sx.long().clamp(0, cam.image_width - 1)
    yi = sy.long().clamp(0, cam.image_height - 1)
    in_col = (mt[yi, xi] > 0)
    if int(in_col.sum()) == 0:
        sys.exit("ERROR: empty selection")
    zd = view_z[in_col]
    z0, z1 = float(torch.quantile(zd, 0.01)), float(torch.quantile(zd, 0.99))

    orig = gaussians._opacity.clone()
    before = to8(render(cam, gaussians, pipe, bg, stage="fine")["render"])
    # the model's own view of the bed, at the timestamp where it was actually observed
    filler = to8(render(ref, gaussians, pipe, bg, stage="fine")["render"])

    peels, counts = [], []
    for s in range(1, args.steps + 1):
        frac = args.front_frac * s / args.steps
        sel = in_col & (view_z <= z0 + (z1 - z0) * frac)
        gaussians._opacity[sel] = -50.0
        peels.append(to8(render(cam, gaussians, pipe, bg, stage="fine")["render"]))
        counts.append(int(sel.sum()))
        gaussians._opacity.copy_(orig)
    raw_after = peels[-1]

    # ---- feathered composite: hole pixels come from the ref-time render -------------------------
    alpha = cv2.GaussianBlur((mk > 127).astype(np.float32), (0, 0), args.feather)[..., None]
    fill_after = (raw_after * (1 - alpha) + filler * alpha).astype(np.uint8)

    gt_ref = (ref.original_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    hole = mk > 127
    p_raw = psnr_np(raw_after[hole], gt_ref[hole])
    p_fill = psnr_np(fill_after[hole], gt_ref[hole])

    panel = np.hstack([
        label(mark_region(before, args), f"before (frame {args.frame})"),
        label(raw_after, f"removal only ({counts[-1]} gaussians) | hole PSNR {p_raw:.1f}"),
        label(fill_after, f"removal + shown at observed time (frame {args.ref_frame}) | {p_fill:.1f} dB"),
        label(gt_ref, f"GT frame {args.ref_frame} = what is really under", (60, 255, 60)),
    ])
    tag = args.tag or f"f{args.frame}_ref{args.ref_frame}"
    out_png = os.path.join(args.out, f"timefill_{tag}.png")
    cv2.imwrite(out_png, panel[:, :, ::-1])

    # ---- video frames: progressive peel, then crossfade the hole to the observed-time content ---
    fdir = os.path.join(args.out, f"frames_{tag}")
    os.makedirs(fdir, exist_ok=True)
    seq = [before] + peels
    for k in range(1, args.fade + 1):
        w = k / args.fade
        seq.append((raw_after * (1 - alpha * w) + filler * alpha * w).astype(np.uint8))
    for i, im in enumerate(seq):
        cv2.imwrite(os.path.join(fdir, f"f{i:03d}.png"), im[:, :, ::-1])

    print(f"[timefill] {counts[-1]} gaussians removed | hole PSNR vs GT: raw {p_raw:.2f} dB -> "
          f"time-filled {p_fill:.2f} dB | {out_png} | {len(seq)} video frames")


if __name__ == "__main__":
    main()
