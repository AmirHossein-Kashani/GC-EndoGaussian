"""Occlusion-holdout evaluation.

Measures how well a trained model reconstructs the tissue inside a held-out box during the
frames where that box was excluded from supervision (simulated tool occlusion). Compares the
occluded-block frames against control frames where the same box WAS supervised.

A model that propagates motion into the occluded region (the graph hypothesis) should show a
SMALLER occlusion gap (control PSNR - occluded PSNR) and a HIGHER occluded-box PSNR than one
that deforms each Gaussian independently.

Usage:  python eval_occlusion.py output/endonerf/<exp> [iteration] [x0 y0 x1 y1 blo bhi]
Defaults match the occ_* config defaults.
"""
import os, sys, glob
import numpy as np
from PIL import Image

mp = sys.argv[1]
it = sys.argv[2] if len(sys.argv) > 2 else "3000"
x0, y0, x1, y1 = 0.35, 0.30, 0.65, 0.70
blo, bhi = 0.33, 0.66
if len(sys.argv) >= 9:
    x0, y0, x1, y1, blo, bhi = map(float, sys.argv[3:9])

base = os.path.join(mp, "train", f"ours_{it}")
rf = sorted(glob.glob(os.path.join(base, "renders", "*.png")))
gf = sorted(glob.glob(os.path.join(base, "gt", "*.png")))
mf = sorted(glob.glob(os.path.join(base, "masks", "*.png")))
N = len(rf)
assert N > 0 and len(gf) == N, f"no/mismatched renders in {base} (renders={N}, gt={len(gf)})"
lo, hi = int(blo * N), int(bhi * N)

def load_rgb(p): return np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
def load_m(p):   return np.asarray(Image.open(p).convert("L")).astype(np.float32) / 255.0

def region_psnr(idxs):
    se, cnt = 0.0, 0
    for i in idxs:
        r, g = load_rgb(rf[i]), load_rgb(gf[i])
        H, W, _ = g.shape
        m = load_m(mf[i]) if i < len(mf) else np.ones((H, W), np.float32)
        box = np.zeros((H, W), bool)
        box[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = True
        sel = box & (m > 0.5)                      # box AND tissue (exclude tool pixels)
        if sel.sum() == 0:
            continue
        d = (r[sel] - g[sel]) ** 2
        se += float(d.sum()); cnt += d.size
    if cnt == 0:
        return float("nan")
    return -10.0 * np.log10(se / cnt + 1e-12)

occ_idxs = [i for i in range(N) if lo <= i <= hi]
ctl_idxs = [i for i in range(N) if not (lo <= i <= hi)]
op = region_psnr(occ_idxs)
cp = region_psnr(ctl_idxs)
print(f"{mp}")
print(f"  occluded-box tissue PSNR (frames {lo}..{hi}, n={len(occ_idxs)}): {op:.4f}")
print(f"  control-box  tissue PSNR (other frames,      n={len(ctl_idxs)}): {cp:.4f}")
print(f"  OCCLUSION GAP (control - occluded, lower=better recovery): {cp - op:.4f}")
