"""Cut / high-motion region evaluation.

Full-frame PSNR drowns the cut: the cut boundary is a small fraction of pixels. This localizes the
metric to the high-motion region (where the tissue is being cut/manipulated) by building a motion
mask from GT temporal differences, then reports reconstruction PSNR there. The GT (and thus the
mask) is identical across models, so it's a fair relative comparison of cut-region fidelity.

Usage:  python eval_cut.py output/endonerf/<exp> [iteration] [top_fraction]
Requires the train set to have been rendered (render.py without --skip_train).
"""
import os, sys, glob
import numpy as np
from PIL import Image

mp = sys.argv[1]
it = sys.argv[2] if len(sys.argv) > 2 else "6000"
topk = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
base = os.path.join(mp, "train", f"ours_{it}")
rf = sorted(glob.glob(os.path.join(base, "renders", "*.png")))
gf = sorted(glob.glob(os.path.join(base, "gt", "*.png")))
N = len(gf)
assert N > 1 and len(rf) == N, f"need rendered train set in {base} (renders={len(rf)}, gt={N})"

def gray(p): return np.asarray(Image.open(p).convert("L"), np.float32)
def rgb(p):  return np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0

# per-pixel motion magnitude = mean over t of |GT_t - GT_{t-1}|  ->  the cut/action region
acc = None
for i in range(1, N):
    d = np.abs(gray(gf[i]) - gray(gf[i - 1]))
    acc = d if acc is None else acc + d
acc /= max(N - 1, 1)
thr = np.quantile(acc, 1.0 - topk)
mask = acc >= thr                                   # high-motion (cut) region

se_hi, cnt_hi, se_lo, cnt_lo = 0.0, 0, 0.0, 0
for i in range(N):
    r, g = rgb(rf[i]), rgb(gf[i])
    dh = (r[mask] - g[mask]) ** 2
    dl = (r[~mask] - g[~mask]) ** 2
    se_hi += float(dh.sum()); cnt_hi += dh.size
    se_lo += float(dl.sum()); cnt_lo += dl.size
psnr_hi = -10.0 * np.log10(se_hi / cnt_hi + 1e-12)
psnr_lo = -10.0 * np.log10(se_lo / cnt_lo + 1e-12)
print(f"{mp}")
print(f"  cut-region PSNR  (top {int(topk*100)}% motion): {psnr_hi:.4f}")
print(f"  static-region PSNR (rest)                     : {psnr_lo:.4f}")
