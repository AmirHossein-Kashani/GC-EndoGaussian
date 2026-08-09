"""Short standalone clip of the Figure-2 cutting lateral-retraction edit.

Crossfade-interpolates the *real* edit_figure.py renders
(before.png + after_mx_{0.08,0.12,0.16}.png in an edit_fig dir) into a smooth
ramp: a local group of control nodes is dragged laterally at inference time,
retracting the overlying tissue to reveal the region behind, held briefly, then
released. A green dashed circle marks the grabbed handle region (matches the
paper figure, drawn from meta.json). CPU only (PIL) -- no GPU / no numpy.

  python3 tools/make_edit_reveal_video.py \
      --dir output/endonerf/cutting_match/edit_fig_r06 \
      --frames_out /tmp/edit_reveal_frames --fps 30
  # then encode frames_out/*.png with ffmpeg (see run_edit_reveal_video usage).
"""
import os, json, math, argparse
from PIL import Image, ImageDraw

GREEN = (57, 255, 20)          # #39ff14, matches compose_panels handle outline


def smooth(u):                 # ease-in-out (cosine)
    return 0.5 * (1.0 - math.cos(math.pi * max(0.0, min(1.0, u))))


def percentile(vals, q):
    s = sorted(vals)
    if not s:
        return 0.0
    idx = q / 100.0 * (len(s) - 1)
    lo = int(math.floor(idx)); hi = int(math.ceil(idx))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def dashed_circle(im, cx, cy, r, color=GREEN, width=2, seg=20):
    d = ImageDraw.Draw(im)
    box = [cx - r, cy - r, cx + r, cy + r]
    for k in range(seg):
        if k % 2 == 0:
            a0 = 360.0 * k / seg
            a1 = 360.0 * (k + 0.72) / seg
            d.arc(box, a0, a1, fill=color, width=width)
    d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=color)   # centre dot
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="edit_fig dir (before.png + after_mx_*.png + meta.json)")
    ap.add_argument("--dirtag", default="mx", help="drag direction tag rendered by edit_figure.py")
    ap.add_argument("--mags", default="0.08,0.12,0.16", help="ascending magnitudes present as after_<tag>_<m>.png")
    ap.add_argument("--frames_out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no_circle", action="store_true")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.dir, "meta.json")))
    cpx, cpy = meta["centroid_px"], meta["centroid_py"]
    rp = percentile([math.hypot(x - cpx, y - cpy) for x, y in zip(meta["sel_px"], meta["sel_py"])], 80)

    def load(name):
        return Image.open(os.path.join(args.dir, name)).convert("RGB")

    mags = [0.0] + [float(m) for m in args.mags.split(",") if m.strip()]
    imgs = [load("before.png")] + [load(f"after_{args.dirtag}_{m:.2f}.png") for m in mags[1:]]
    keys = list(zip(mags, imgs))
    mmax = mags[-1]

    def frame_at(m):
        if m <= keys[0][0]:
            return keys[0][1].copy()
        if m >= keys[-1][0]:
            return keys[-1][1].copy()
        for (m0, i0), (m1, i1) in zip(keys, keys[1:]):
            if m0 <= m <= m1:
                return Image.blend(i0, i1, (m - m0) / (m1 - m0))
        return keys[-1][1].copy()

    fps = args.fps
    # (label, seconds, magnitude-profile(u)->fraction-of-mmax)
    plan = [
        ("hold0", 0.5, lambda u: 0.0),
        ("retract", 2.0, lambda u: smooth(u)),
        ("hold", 1.2, lambda u: 1.0),
        ("release", 1.4, lambda u: 1.0 - smooth(u)),
        ("hold0", 0.4, lambda u: 0.0),
    ]

    os.makedirs(args.frames_out, exist_ok=True)
    for f in os.listdir(args.frames_out):
        if f.endswith(".png"):
            os.remove(os.path.join(args.frames_out, f))

    idx = 0
    for _, secs, prof in plan:
        n = max(1, int(round(secs * fps)))
        for k in range(n):
            u = k / max(n - 1, 1)
            m = mmax * prof(u)
            im = frame_at(m)
            if not args.no_circle:
                dashed_circle(im, cpx, cpy, rp)
            im.save(os.path.join(args.frames_out, f"f{idx:04d}.png"))
            idx += 1

    print(f"[edit_reveal] wrote {idx} frames @ {fps}fps ({idx/fps:.1f}s) to {args.frames_out} "
          f"| circle=({cpx:.0f},{cpy:.0f}) r={rp:.0f}px")


if __name__ == "__main__":
    main()
