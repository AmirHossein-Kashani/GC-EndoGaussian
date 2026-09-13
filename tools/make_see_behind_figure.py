"""Figure: the model shows tissue that an instrument is hiding, verified against ground truth.

At some timestamp t an instrument covers a region; at another timestamp s the same region is in
plain view. Because the canonical Gaussian set is shared across time and tool pixels are excluded
from the photometric loss, rendering t still shows the tissue behind the instrument. Frame s is the
ground-truth check -- this is re-showing what was observed, not predicting.

The frame pair is chosen automatically: for every (t, s) we measure how much of t's tool footprint is
still covered at s, and take the minimum, so the outlined region is exactly the part of the claim
ground truth can verify.

  python3 tools/make_see_behind_figure.py        # CPU only, reuses existing renders
"""
import os, glob, argparse
import numpy as np
import cv2
from PIL import Image

ZOOM = (45, 5, 335, 205)     # crop of the revealed region, in image pixels


def cap(im, txt, color=(255, 255, 255), h=32):
    im = im.copy()
    cv2.rectangle(im, (0, im.shape[0] - h), (im.shape[1], im.shape[0]), (0, 0, 0), -1)
    cv2.putText(im, txt, (10, im.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", default="output/endonerf/pulling_match3k/video/ours_3000")
    ap.add_argument("--masks", default="data/endonerf/pulling/masks")
    ap.add_argument("--out", default="docs/figures/see_behind_instrument.png")
    ap.add_argument("--min_gap", type=int, default=8, help="minimum frame separation for the pair")
    args = ap.parse_args()

    M = [np.asarray(Image.open(p).convert("L")) > 127
         for p in sorted(glob.glob(os.path.join(args.masks, "*.png")))]
    N = len(M)

    # pick (t, s): most of t's tool footprint visible at s
    best = None
    for t in range(N):
        for s in range(N):
            if abs(s - t) < args.min_gap:
                continue
            ov = (M[t] & M[s]).sum() / max(M[t].sum(), 1)
            if best is None or ov < best[0]:
                best = (ov, t, s)
    _, T, S = best
    rev = M[T] & ~M[S]
    seen = np.stack([~m for m in M]).sum(0)
    print(f"[fig] pair t={T} s={S} | tool at t covers {M[T].mean()*100:.1f}% "
          f"| verifiable revealed area {rev.mean()*100:.1f}% of frame")
    print(f"[fig] revealed region directly observed in {seen[rev].mean():.0f}/{N} frames "
          f"(min {seen[rev].min()})")

    g = lambda k, i: np.asarray(Image.open(os.path.join(args.renders, k, f"{i:05d}.png")))
    gt_t, re_t, gt_s = g("gt", T), g("renders", T), g("gt", S)

    cnt, _ = cv2.findContours(
        cv2.morphologyEx(rev.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = [c for c in cnt if cv2.contourArea(c) > 2000]

    def outline(im, color):
        im = im.copy(); cv2.drawContours(im, cnt, -1, color, 2); return im

    top = np.concatenate([
        cap(outline(gt_t, (255, 70, 70)), f"1. GROUND TRUTH frame {T}  -  instrument hides the tissue", (255, 130, 130)),
        cap(outline(re_t, (70, 255, 70)), f"2. OURS frame {T}  -  the tissue behind it is reconstructed", (100, 255, 100)),
        cap(outline(gt_s, (90, 200, 255)), f"3. GROUND TRUTH frame {S}  -  same tissue, before it was hidden", (130, 215, 255)),
    ], 1)

    x0, y0, x1, y1 = ZOOM
    z = lambda im: cv2.resize(im[y0:y1, x0:x1], (960, int(960 * (y1 - y0) / (x1 - x0))),
                              interpolation=cv2.INTER_CUBIC)
    zr = np.concatenate([
        cap(z(re_t), "ZOOM  -  OURS, rendered from behind the instrument", (100, 255, 100)),
        cap(z(gt_s), f"ZOOM  -  the real tissue, observed at frame {S}", (130, 215, 255)),
    ], 1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    Image.fromarray(np.concatenate([top, zr], 0)).save(args.out)
    print(f"[fig] -> {args.out}")


if __name__ == "__main__":
    main()
