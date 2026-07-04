"""Figure 5 — Sparse-to-dense tissue localization vs. number of observed landmarks K.

Regenerates the localization curve with the finalized terminology (no "naive" label).
Numbers are the paper's Table 4 cross-trial means +/- SD over four SuPer trials
(held-out reprojection error in px; lower is better). Output: figures/sparse_to_dense_localization.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K = np.array([4, 8, 16])

# (mean, sd) per method, from paper Table 4
SERIES = {
    "Ours (match)":     dict(mean=[6.82, 6.80, 8.09], sd=[2.24, 1.98, 3.12],
                             color="#1f5fd6", marker="o", ls="-",  lw=2.4, z=5),
    "SC-GS (learned)":  dict(mean=[6.71, 6.74, 8.06], sd=[2.17, 2.00, 2.80],
                             color="#7b3fbf", marker="s", ls="-",  lw=2.0, z=4),
    "Nearest-handle":   dict(mean=[5.69, 4.73, 3.97], sd=[1.98, 1.49, 1.65],
                             color="#e8820c", marker="^", ls="--", lw=1.8, z=3),
    "Thin-plate spline":dict(mean=[11.61, 5.87, 3.45], sd=[1.03, 1.54, 0.77],
                             color="#2ca02c", marker="D", ls="--", lw=1.8, z=3),
}

fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=200)

for label, s in SERIES.items():
    mean = np.array(s["mean"], float)
    sd = np.array(s["sd"], float)
    ax.plot(K, mean, color=s["color"], marker=s["marker"], linestyle=s["ls"],
            linewidth=s["lw"], markersize=7, label=label, zorder=s["z"],
            markeredgecolor="white", markeredgewidth=0.6)
    # light +/- SD band to convey the ours/SC-GS statistical tie
    ax.fill_between(K, mean - sd, mean + sd, color=s["color"], alpha=0.08, zorder=1)

ax.set_xscale("log", base=2)
ax.set_xticks(K)
ax.set_xticklabels([str(k) for k in K])
ax.set_xlabel("Number of observed landmarks  $K$", fontsize=12)
ax.set_ylabel("Held-out reprojection error (px)  $\\downarrow$", fontsize=12)
ax.set_ylim(2.5, 13.0)
ax.grid(True, which="major", linestyle=":", linewidth=0.7, alpha=0.5)
ax.tick_params(labelsize=11)

# place the legend outside the axes so it never covers the data
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=10, frameon=True, framealpha=0.95,
          loc="upper left", bbox_to_anchor=(1.02, 1.0),
          title="learned  |  classical\ninterpolants", title_fontsize=9)

fig.tight_layout()

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "sparse_to_dense_localization.png")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
