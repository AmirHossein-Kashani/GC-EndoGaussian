"""Representative demo video for the workshop submission (anonymized).

Tells the paper's actual story in three segments on a fixed 1280x1024 canvas:

  1. GENERALITY  -- synchronized GT | reconstruction replays on four real da Vinci
     tissue sequences (SuPer trials 3/4/8/9), then the two EndoNeRF scenes.
     Pure image composition from existing video renders.
  2. THE FINDING -- the residual-isolation ablation over time on pulling:
     top row = SC-GS-style (no residual) | GC-EndoGaussian renders,
     bottom row = their tissue-masked error heatmaps vs GT (shared scale).
     The no-residual panel visibly glows/flickers; ours stays dark.
  3. EDITABILITY -- brief: the paper-figure drag on pulling (median region,
     r=0.14) ramped and held, with change heatmap; then held across time.

Run inside a GPU job AFTER rendering the pulling_scgs video split:
  python tools/make_demo_video.py --out docs/supplementary/demo.mp4
(paths to models/renders are fixed below; only segment 3 needs the GPU)
"""
import os, sys, math, glob
import numpy as np
import torch
import cv2
import imageio.v2 as iio
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render

W, H = 640, 512
CANVAS = (2 * H, 2 * W)          # 1024 x 1280


def imread(p):
    return np.asarray(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))


def frames_of(d):
    return sorted(glob.glob(os.path.join(d, "*.png")))


def label(im, txt, y=24):
    cv2.rectangle(im, (0, y - 24), (im.shape[1], y + 10), (0, 0, 0), -1)
    cv2.putText(im, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return im


def tag(im, txt):
    cv2.putText(im, txt, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(im, txt, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def grid(tl, tr, bl, br, title):
    fr = np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])])
    return label(fr, title)


def heat_rgb(err, vmax):
    d = np.clip(err / max(vmax, 1e-6), 0, 1)
    return cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1].copy()


def to8(img):
    return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()


def dashed_circle(im, cx, cy, r, color=(60, 220, 60), seg=18):
    for k in range(seg):
        if k % 2 == 0:
            a0, a1 = 360.0 * k / seg, 360.0 * (k + 0.7) / seg
            cv2.ellipse(im, (int(cx), int(cy)), (int(r), int(r)), 0, a0, a1, color, 2, cv2.LINE_AA)
    return im


def replay_pair(base_dir, name, step=2):
    """(gt, render, tag) frame triples from a video render dir."""
    g, r = frames_of(f"{base_dir}/gt"), frames_of(f"{base_dir}/renders")
    n = min(len(g), len(r))
    return [(imread(g[i]), tag(imread(r[i]), name)) for i in range(0, n, step)]


@torch.no_grad()
def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=6000, type=int)
    parser.add_argument("--configs", type=str, default="arguments/endonerf/pulling_graph_match.py")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--fps", type=int, default=15)
    args = get_combined_args(parser)
    if args.configs:
        from utils.config_loader import load_config
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, load_config(args.configs))
    out_path = getattr(args, "out", None) or "docs/supplementary/demo.mp4"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frames = []

    # ================= segment 1: generality on real surgical sequences ==========
    t1 = "1/3  4D reconstruction on real da Vinci sequences:  ground truth (L) | ours (R)"
    a = replay_pair("output/endonerf/super_match/video/ours_3000",   "trial 3")
    b = replay_pair("output/endonerf/super_t4_match/video/ours_3000", "trial 4")
    for (g1, r1), (g2, r2) in zip(a, b):
        frames.append(grid(g1, r1, g2, r2, t1))
    a = replay_pair("output/endonerf/super_t8_match/video/ours_3000", "trial 8")
    b = replay_pair("output/endonerf/super_t9_match/video/ours_3000", "trial 9")
    for (g1, r1), (g2, r2) in zip(a, b):
        frames.append(grid(g1, r1, g2, r2, t1))
    t1b = "1/3  4D reconstruction (EndoNeRF):  ground truth (L) | ours (R)"
    a = replay_pair("output/endonerf/pulling_match3k/video/ours_3000", "pulling", step=1)
    b = replay_pair("output/endonerf/cutting_match/video/ours_6000",   "cutting", step=2)
    for (g1, r1), (g2, r2) in zip(a, b):
        frames.append(grid(g1, r1, g2, r2, t1b))

    # ================= segment 2: the finding — residual ablation over time ======
    t2 = "2/3  Why the dense residual matters:  no residual (L) | ours (R);  bottom: error vs GT"
    scgs = "output/endonerf/pulling_scgs/video/ours_3000"
    ours = "output/endonerf/pulling_match3k/video/ours_3000"
    g_f, s_f, o_f = frames_of(f"{ours}/gt"), frames_of(f"{scgs}/renders"), frames_of(f"{ours}/renders")
    m_f = sorted(glob.glob("data/endonerf/pulling/masks/*.png"))
    n = min(len(g_f), len(s_f), len(o_f), len(m_f))
    # shared error scale from a few sample frames
    samples = []
    for i in range(0, n, max(n // 6, 1)):
        gt = imread(g_f[i]).astype(np.float32) / 255
        tissue = (np.asarray(cv2.imread(m_f[i], 0)) < 127).astype(np.float32)
        for f in (s_f, o_f):
            samples.append((np.abs(imread(f[i]).astype(np.float32) / 255 - gt).mean(-1) * tissue))
    vmax = float(np.percentile(np.stack(samples), 99.7))
    for loop in range(2):                                     # play the comparison twice
        for i in range(n):
            gt = imread(g_f[i]).astype(np.float32) / 255
            tissue = (np.asarray(cv2.imread(m_f[i], 0)) < 127).astype(np.float32)
            s = imread(s_f[i]); o = imread(o_f[i])
            es = heat_rgb(np.abs(s.astype(np.float32) / 255 - gt).mean(-1) * tissue, vmax)
            eo = heat_rgb(np.abs(o.astype(np.float32) / 255 - gt).mean(-1) * tissue, vmax)
            frames.append(grid(tag(s.copy(), "SC-GS-style, no residual"),
                               tag(o.copy(), "GC-EndoGaussian (ours)"),
                               es, eo, t2))

    # ================= segment 3: editability (paper-figure drag, brief) =========
    dataset, pipe, hyper = model.extract(args), pipeline.extract(args), hp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    nd = gaussians._deformation.deformation_net.node_deform
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    vcams = scene.getVideoCameras()
    cam0 = scene.getTrainCameras()[0]
    nx = nd.node_xyz; M = nx.shape[0]
    ext = float((nx.max(0).values - nx.min(0).values).norm())
    center = nx.median(0).values
    sel = (nx - center).norm(dim=1) < 0.14 * ext              # figure-2 handle
    direction = torch.zeros(3, device=nx.device); direction[1] = 1.0   # figure-2 "drag down"
    zero = torch.zeros(M, 3, device=nx.device)
    ph = torch.cat([nx[sel], torch.ones_like(nx[sel][:, :1])], dim=1)
    clip = ph @ cam0.full_proj_transform.to(nx.device)
    ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-7)
    px, py = (ndc[:, 0] * .5 + .5) * W, (ndc[:, 1] * .5 + .5) * H
    ccx, ccy = float(px.mean()), float(py.mean())
    crad = float(torch.sqrt((px - px.mean())**2 + (py - py.mean())**2).max()) + 12

    nd.edit_translation = zero
    base = render(cam0, gaussians, pipe, bg, stage="fine")["render"]
    base8 = dashed_circle(to8(base), ccx, ccy, crad)
    t3 = "3/3  Drag-to-edit at inference time (no retraining):  edited | change heatmap;  bottom: held over time"
    RAMP, HOLD = 40, 15
    prof = [0.5 * (1 - math.cos(math.pi * k / RAMP)) for k in range(RAMP)] + [1.0] * HOLD
    PEAK = 0.16 * ext
    hold = zero.clone(); hold[sel] = direction * PEAK
    n_bot = len(prof)
    bot_cams = [vcams[int(i * (len(vcams) - 1) / max(n_bot - 1, 1))] for i in range(n_bot)]
    for k, aa in enumerate(prof):
        disp = zero.clone(); disp[sel] = direction * (aa * PEAK)
        nd.edit_translation = disp
        img = render(cam0, gaussians, pipe, bg, stage="fine")["render"]
        diff = (img - base).abs().sum(0)
        tl = dashed_circle(to8(img), ccx, ccy, crad)
        tr = heat_rgb(diff.cpu().numpy(), 0.5)
        camb = bot_cams[k]
        nd.edit_translation = zero
        bl = tag(to8(render(camb, gaussians, pipe, bg, stage="fine")["render"]), "unedited, t sweeping")
        nd.edit_translation = hold
        br = tag(dashed_circle(to8(render(camb, gaussians, pipe, bg, stage="fine")["render"]),
                               ccx, ccy, crad), "edit held, t sweeping")
        frames.append(grid(tl, tr, bl, br, t3))
    nd.edit_translation = zero

    iio.mimwrite(out_path, frames, fps=args.fps, quality=8)
    print(f"wrote {out_path}: {len(frames)} frames @ {args.fps} fps "
          f"({len(frames)/args.fps:.0f}s), {frames[0].shape[1]}x{frames[0].shape[0]}")


if __name__ == "__main__":
    main()
