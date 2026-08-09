"""Re-render the demo 'core' (generality + residual-finding segments) with banners
that match the new presentation video — i.e. WITHOUT the old '1/3 … 2/3 …' numbering
(which no longer fits, now that title/what/edit/results cards frame the footage) and
using 'GC-EndoGaussian' consistently instead of 'ours'.

This is a CPU-only re-composition of the existing per-frame renders (segments 1-2 of
tools/make_demo_video.py); segment 3 was already trimmed off demoF1. Writes PNG frames
to <out>/frames for ffmpeg. Run with the opencv module loaded (cv2 + numpy).
"""
import os, glob, argparse
import numpy as np
import cv2

W, H = 640, 512

# Updated banners (no fraction numbering; GC-EndoGaussian instead of 'ours')
T1  = "4D reconstruction on real da Vinci tissue:  ground truth (L) | GC-EndoGaussian (R)"
T1B = "4D reconstruction on EndoNeRF scenes:  ground truth (L) | GC-EndoGaussian (R)"
T2  = "Why the dense residual matters:  no residual (L) | GC-EndoGaussian (R);  bottom: error vs GT"


def imread(p):
    return np.asarray(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))


def frames_of(d):
    return sorted(glob.glob(os.path.join(d, "*.png")))


def label(im, txt, y=26):
    cv2.rectangle(im, (0, y - 26), (im.shape[1], y + 12), (0, 0, 0), -1)
    cv2.putText(im, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def tag(im, txt):
    cv2.putText(im, txt, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(im, txt, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def grid(tl, tr, bl, br, title):
    return label(np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])]), title)


def heat_rgb(err, vmax):
    d = np.clip(err / max(vmax, 1e-6), 0, 1)
    return cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1].copy()


def replay_pair(base_dir, name, step=2):
    g, r = frames_of(f"{base_dir}/gt"), frames_of(f"{base_dir}/renders")
    n = min(len(g), len(r))
    return [(imread(g[i]), tag(imread(r[i]), name)) for i in range(0, n, step)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/endonerf/demo_core")
    args = ap.parse_args()
    fdir = os.path.join(args.out, "frames")
    os.makedirs(fdir, exist_ok=True)
    for f in os.listdir(fdir):
        if f.endswith(".png"):
            os.remove(os.path.join(fdir, f))

    frames = []
    # ---- segment 1: generality on real surgical + EndoNeRF sequences ----
    # All four da Vinci trials, now retrained with DILATED tool masks (*_tooldil) so the
    # tool region reconstructs cleanly. Lead with trial 9 (cleanest), then 8 / 3 / 4.
    a = replay_pair("output/endonerf/super_t9_tooldil/video/ours_3000", "trial 9")
    b = replay_pair("output/endonerf/super_t8_tooldil/video/ours_3000", "trial 8")
    for (g1, r1), (g2, r2) in zip(a, b):
        frames.append(grid(g1, r1, g2, r2, T1))
    a = replay_pair("output/endonerf/super_t3_tooldil/video/ours_3000", "trial 3")
    b = replay_pair("output/endonerf/super_t4_tooldil/video/ours_3000", "trial 4")
    for (g1, r1), (g2, r2) in zip(a, b):
        frames.append(grid(g1, r1, g2, r2, T1))
    a = replay_pair("output/endonerf/pulling_match3k/video/ours_3000", "pulling", step=1)
    b = replay_pair("output/endonerf/cutting_match/video/ours_6000", "cutting", step=2)
    for (g1, r1), (g2, r2) in zip(a, b):
        frames.append(grid(g1, r1, g2, r2, T1B))

    # ---- segment 2: the residual finding, error-vs-GT over time ----
    scgs = "output/endonerf/pulling_scgs/video/ours_3000"
    ours = "output/endonerf/pulling_match3k/video/ours_3000"
    g_f = frames_of(f"{ours}/gt"); s_f = frames_of(f"{scgs}/renders"); o_f = frames_of(f"{ours}/renders")
    m_f = sorted(glob.glob("data/endonerf/pulling/masks/*.png"))
    n = min(len(g_f), len(s_f), len(o_f), len(m_f))
    samples = []
    for i in range(0, n, max(n // 6, 1)):
        gt = imread(g_f[i]).astype(np.float32) / 255
        tissue = (np.asarray(cv2.imread(m_f[i], 0)) < 127).astype(np.float32)
        for f in (s_f, o_f):
            samples.append(np.abs(imread(f[i]).astype(np.float32) / 255 - gt).mean(-1) * tissue)
    vmax = float(np.percentile(np.stack(samples), 99.7))
    for _ in range(2):
        for i in range(n):
            gt = imread(g_f[i]).astype(np.float32) / 255
            tissue = (np.asarray(cv2.imread(m_f[i], 0)) < 127).astype(np.float32)
            s = imread(s_f[i]); o = imread(o_f[i])
            es = heat_rgb(np.abs(s.astype(np.float32) / 255 - gt).mean(-1) * tissue, vmax)
            eo = heat_rgb(np.abs(o.astype(np.float32) / 255 - gt).mean(-1) * tissue, vmax)
            frames.append(grid(tag(s.copy(), "SC-GS-style, no residual"),
                               tag(o.copy(), "GC-EndoGaussian (ours)"), es, eo, T2))

    for i, fr in enumerate(frames):
        cv2.imwrite(os.path.join(fdir, f"f{i:05d}.png"), fr[:, :, ::-1])
    print(f"[demo_core] wrote {len(frames)} frames -> {fdir}")


if __name__ == "__main__":
    main()
