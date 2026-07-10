"""Figure: visual evidence for the residual-isolation ablation (Table 3).

Columns: GT | EndoGaussian | SC-GS-style (no residual) | SC-GS + residual | GC-EndoGaussian.
Row 1: rendering with a zoom-inset box; Row 2: zoom crop; Row 3: zoomed per-pixel error
heatmap (shared color scale, tool region masked out — tool pixels are excluded from the
training loss and the reported metrics, so their error is not meaningful).
The crop auto-selects the tissue region where the no-residual model's error most exceeds
the +residual model's — i.e., what the dense residual visibly fixes.

Run on the login node (CPU): python tools/make_fig_residual_ablation.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import fftconvolve

ROOT = "output/endonerf"
FRAME = "00000.png"
TEST_SRC_IDX = 1                       # test set = every 8th frame starting at 1
MODELS = [("pulling",             "EndoGaussian"),
          ("pulling_scgs",        "SC-GS-style (no residual)"),
          ("pulling_scgs_hybrid", "SC-GS-style + residual"),
          ("pulling_match3k",     "GC-EndoGaussian")]
OUT = "docs/figures/fig_residual_ablation"
CROP = 170


def load(m, kind):
    return np.asarray(Image.open(f"{ROOT}/{m}/test/ours_3000/{kind}/{FRAME}"),
                      dtype=np.float32) / 255.0


gt = load(MODELS[0][0], "gt")
renders = {m: load(m, "renders") for m, _ in MODELS}

# tissue-only weighting: tool pixels (mask=255) are excluded from loss + metrics
mask_files = sorted(os.listdir("data/endonerf/pulling/masks"))
tool = np.asarray(Image.open(
    os.path.join("data/endonerf/pulling/masks", mask_files[TEST_SRC_IDX])), dtype=np.float32)
tissue = (tool < 127).astype(np.float32)                      # 1 = tissue
errs = {m: np.abs(renders[m] - gt).mean(-1) * tissue for m, _ in MODELS}

# crop where (no-residual error - +residual error) is largest, tissue only
gain = errs["pulling_scgs"] - errs["pulling_scgs_hybrid"]
k = np.ones((CROP, CROP)) / CROP**2
score = fftconvolve(gain, k, mode="valid")
cy, cx = np.unravel_index(np.argmax(score), score.shape)
y0, x0 = int(cy), int(cx)
crop_err = [errs[m][y0:y0 + CROP, x0:x0 + CROP] for m, _ in MODELS]
vmax = max(float(np.percentile(e, 99.0)) for e in crop_err)

cols = [("GT", gt, None)] + [(lbl, renders[m], errs[m]) for m, lbl in MODELS]
fig, axes = plt.subplots(3, len(cols), figsize=(2.9 * len(cols), 7.2))
for c, (lbl, img, err) in enumerate(cols):
    ax = axes[0, c]
    ax.imshow(img); ax.set_title(lbl, fontsize=11)
    ax.add_patch(plt.Rectangle((x0, y0), CROP, CROP, ec="yellow", fc="none", lw=1.8))
    ax = axes[1, c]
    ax.imshow(img[y0:y0 + CROP, x0:x0 + CROP])
    for s in ax.spines.values(): s.set_edgecolor("yellow"); s.set_linewidth(1.8)
    ax = axes[2, c]
    if err is None:
        ax.axis("off")
    else:
        im = ax.imshow(err[y0:y0 + CROP, x0:x0 + CROP], cmap="inferno", vmin=0, vmax=vmax)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
axes[1, 0].set_ylabel("zoom", fontsize=11)
axes[2, 1].set_ylabel("|error| (zoom)", fontsize=11)
cbar = fig.colorbar(im, ax=axes[2, :].tolist(), fraction=0.015, pad=0.01)
cbar.set_label("mean abs. error (tissue only)", fontsize=9)
plt.subplots_adjust(wspace=0.02, hspace=0.05, left=0.03, right=0.91, top=0.95, bottom=0.02)
os.makedirs("docs/figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")
mean_err = {m: float(errs[m].sum() / tissue.sum()) for m, _ in MODELS}
print("wrote", OUT, "| crop", (x0, y0), "| tissue mean-abs-err:",
      {k: round(v, 5) for k, v in mean_err.items()})
