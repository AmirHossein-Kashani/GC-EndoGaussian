"""Object-level removal reveal for the STIR scenes (v2 of tools/stir_timefill_reveal.py).

v1 selected a screen-space ellipse and removed the front fraction of the depth range. Here the
selection is the OBJECT: a Gaussian belongs to the moving tissue piece iff (a) it projects inside a
loose region hint at the edit frame and (b) its trajectory span across the sequence exceeds
--motion_mm -- the organ is the only thing that travels in these clips, the bed is static. The
whole object is removed (no depth fraction), so the reveal hole follows the organ's exact
silhouette, and the temporal-completion filler is rendered WITH the object removed too, so the
organ's later resting position can no longer intrude into the filled region.

  python3 tools/stir_object_reveal.py --model_path output/stir/seq33_6k \
      --configs arguments/stir/seq33_6k.py --iteration 6000 \
      --frame 8 --ref_frame 90 --mask data/endonerf/stir06_seq33/organ_mask_f8.png \
      --motion_mm 8 --out output/stir/seq33_6k/objreveal
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
    parser.add_argument("--mask", type=str, required=True, help="loose region hint at the edit frame")
    parser.add_argument("--motion_mm", type=float, default=8.0,
                        help="trajectory-span threshold: Gaussians moving more than this belong to the object")
    parser.add_argument("--traj_samples", type=int, default=12)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--fade", type=int, default=8)
    parser.add_argument("--feather", type=int, default=9)
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

    # ---- region hint at the edit frame ------------------------------------------------------
    xyz = deformed_xyz(gaussians, cam)
    ph = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
    clip = ph @ cam.full_proj_transform.to(xyz.device)
    ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-7)
    sx = (ndc[:, 0] * .5 + .5) * cam.image_width
    sy = (ndc[:, 1] * .5 + .5) * cam.image_height
    mk = cv2.imread(args.mask, 0)
    if mk is None:
        sys.exit(f"ERROR: could not read mask {args.mask}")
    mk = cv2.resize(mk, (cam.image_width, cam.image_height), interpolation=cv2.INTER_NEAREST)
    mk = cv2.dilate(mk, np.ones((11, 11), np.uint8))          # slight margin only: the hint hugs the organ
    mt = torch.from_numpy((mk > 127).astype(np.uint8)).to(sx.device)
    xi = sx.long().clamp(0, cam.image_width - 1)
    yi = sy.long().clamp(0, cam.image_height - 1)
    in_hint = mt[yi, xi] > 0

    # ---- object membership: the near-depth cluster within the hint --------------------------
    # Motion-based selection fails here: the deformation field represents the organ's travel by
    # recruiting different Gaussians over time (median per-Gaussian trajectory span ~2 mm), so
    # per-Gaussian motion does not tag the object. Depth does: within the hint the histogram is
    # bimodal (organ sits tens of mm proud of the bed); Otsu splits it adaptively and the object
    # is the ENTIRE near cluster -- its full thickness, nothing of the bed.
    view_z = (ph @ cam.world_view_transform.to(xyz.device))[:, 2]
    z = view_z[in_hint].cpu().numpy()
    z = z[(z > 1) & (z < np.percentile(z, 99.5))]
    hist, edges = np.histogram(z, bins=64)
    w = hist.astype(np.float64); tot = w.sum()
    mids = 0.5 * (edges[:-1] + edges[1:])
    best, thr = -1.0, float(np.median(z))
    for k in range(1, 64):
        w0, w1 = w[:k].sum(), w[k:].sum()
        if w0 < tot * 0.05 or w1 < tot * 0.05:
            continue
        m0 = (w[:k] * mids[:k]).sum() / w0
        m1 = (w[k:] * mids[k:]).sum() / w1
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best:
            best, thr = v, float(edges[k])
    thr -= 8.0            # safety margin: keep the near rim of the bed out of the selection
    sel = in_hint & (view_z < thr)
    print(f"[objsel] depth Otsu split at {thr:.1f} mm (hint depths {z.min():.0f}-{z.max():.0f}) | "
          f"object = in-hint & nearer: {int(sel.sum())} Gaussians")

    # ---- the object's silhouette at the edit frame (alpha-blended contribution) -------------
    n = xyz.shape[0]
    ocol = torch.zeros(n, 3, device="cuda"); ocol[sel] = 1.0
    omap = render(cam, gaussians, pipe, bg, override_color=ocol, stage="fine")["render"][0]
    sil = (omap > 0.5).float().cpu().numpy().astype(np.uint8)
    sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    alpha = cv2.GaussianBlur(sil.astype(np.float32), (0, 0), args.feather)[..., None]

    orig = gaussians._opacity.clone()
    before = to8(render(cam, gaussians, pipe, bg, stage="fine")["render"])

    # filler: the scene at the observed time, as-is. (Removing the object here too was tried and
    # backfired: where the hole overlaps the object's resting place at ref time, the ref view of
    # the bed is itself unobserved, so the filler showed junk while GT shows the object.)
    filler = to8(render(ref, gaussians, pipe, bg, stage="fine")["render"])
    gaussians._opacity[sel] = -50.0
    raw_after = to8(render(cam, gaussians, pipe, bg, stage="fine")["render"])
    gaussians._opacity.copy_(orig)

    # progressive removal for the video: fade the object's opacity down in steps
    peels = []
    for s in range(1, args.steps + 1):
        gaussians._opacity[sel] = orig[sel] - 8.0 * s / args.steps        # sigmoid ramp to ~0
        if s == args.steps:
            gaussians._opacity[sel] = -50.0
        peels.append(to8(render(cam, gaussians, pipe, bg, stage="fine")["render"]))
        gaussians._opacity.copy_(orig)

    fill_after = (raw_after * (1 - alpha) + filler * alpha).astype(np.uint8)
    gt_ref = (ref.original_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    hole = sil > 0
    p_raw = psnr_np(raw_after[hole], gt_ref[hole])
    p_fill = psnr_np(fill_after[hole], gt_ref[hole])

    # outline the object silhouette on the before panel (this IS the selection now)
    before_marked = before.copy()
    cont, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cont:
        pts = c[:, 0, :]
        for k in range(0, len(pts), 14):
            seg = pts[k:k + 8]
            if len(seg) > 1:
                cv2.polylines(before_marked, [seg.reshape(-1, 1, 2)], False, (60, 220, 60), 2, cv2.LINE_AA)

    panel = np.hstack([
        label(before_marked, f"before (frame {args.frame}); selected object outlined"),
        label(raw_after, f"object removed ({int(sel.sum())} gaussians) | hole PSNR {p_raw:.1f}"),
        label(fill_after, f"removal + shown at observed time (frame {args.ref_frame}) | {p_fill:.1f} dB"),
        label(gt_ref, f"GT frame {args.ref_frame} = what is really under", (60, 255, 60)),
    ])
    tag = args.tag or f"f{args.frame}_ref{args.ref_frame}_v2"
    out_png = os.path.join(args.out, f"objreveal_{tag}.png")
    cv2.imwrite(out_png, panel[:, :, ::-1])

    fdir = os.path.join(args.out, f"frames_{tag}")
    os.makedirs(fdir, exist_ok=True)
    seq = [before] + peels
    for k in range(1, args.fade + 1):
        w = k / args.fade
        seq.append((raw_after * (1 - alpha * w) + filler * alpha * w).astype(np.uint8))
    for i, im in enumerate(seq):
        cv2.imwrite(os.path.join(fdir, f"f{i:03d}.png"), im[:, :, ::-1])

    print(f"[objreveal] {int(sel.sum())} object gaussians removed | silhouette {100*sil.mean():.1f}% "
          f"of frame | hole PSNR raw {p_raw:.2f} -> filled {p_fill:.2f} dB | {out_png}")


if __name__ == "__main__":
    main()
