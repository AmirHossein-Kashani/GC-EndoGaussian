"""Figure 5 — Sparse-to-dense tissue localization vs. number of observed landmarks K.

Full decontamination-story plot (no "naive" label): the uncorrected/residual-active
"leak" line (grey dashed) sits near the ~2 px noise floor and demonstrates the confound;
the decontaminated (control-only) curve is the honest measure, compared against the SC-GS
learned baseline and three classical interpolants (nearest-handle, TPS, rigid).

Numbers: paper Table 3 (uncorrected/full-model leak) and Table 4 (decontaminated learned +
classical), cross-trial means over four SuPer trials; held-out reprojection error (px, lower
is better). Output: docs/figures/sparse_to_dense_localization.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K = np.array([4, 8, 16])
NOISE_FLOOR = 2.0  # ~frame-0 projection error, measurement noise floor

# means per series (px). Ordered for a legend that reads: leak, ours, learned, classical.
SERIES = [
    # label, means, color, marker, linestyle, linewidth, zorder
    ("Ours, uncorrected metric (reconstruction leak)", [2.86, 2.77, 2.92],
        "#8d8d8d", "o", (0, (6, 3)), 2.0, 3),
    ("Ours, decontaminated (control only)",            [6.82, 6.80, 8.09],
        "#1f5fd6", "o", "-", 3.2, 6),
    ("SC-GS (learned baseline)",                       [6.71, 6.74, 8.06],
        "#7b3fbf", "s", "-", 2.0, 5),
    ("Nearest-handle (classical)",                     [5.69, 4.73, 3.97],
        "#e8820c", "^", "-", 2.0, 4),
    ("Thin-plate spline (classical)",                  [11.61, 5.87, 3.45],
        "#2ca02c", "D", "-", 2.0, 4),
    ("Rigid (classical)",                              [6.89, 6.03, 6.24],
        "#c0392b", "v", "-", 2.0, 4),
]

fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=200)

for label, mean, color, marker, ls, lw, z in SERIES:
    ax.plot(K, mean, color=color, marker=marker, linestyle=ls, linewidth=lw,
            markersize=8, label=label, zorder=z,
            markeredgecolor="white", markeredgewidth=0.6)

# measurement noise floor
ax.axhline(NOISE_FLOOR, color="#555555", linestyle=":", linewidth=1.2,
           label=f"measurement noise floor (~{NOISE_FLOOR:.0f} px)", zorder=2)

ax.set_xscale("log", base=2)
ax.set_xticks(K)
ax.set_xticklabels([str(k) for k in K])
ax.set_xlabel("Number of observed landmarks  $K$", fontsize=12)
ax.set_ylabel("Held-out reprojection error (px)  $\\downarrow$", fontsize=12)
ax.set_ylim(1.6, 12.2)
ax.set_title("Sparse-to-dense tissue localization (decontaminated), mean over 4 trials",
             fontsize=12.5)
ax.grid(True, which="major", linestyle=":", linewidth=0.7, alpha=0.5)
ax.tick_params(labelsize=11)
ax.legend(fontsize=9.5, frameon=True, framealpha=0.95, loc="upper left")

fig.tight_layout()

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "sparse_to_dense_localization.png")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
