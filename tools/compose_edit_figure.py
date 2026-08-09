"""Compose the GC-EndoGaussian 'generality' edit figure from rendered sweeps.

Runs locally (no GPU) with the repo .venv (numpy/PIL/matplotlib). Reads the
edit_figure.py outputs (before.png, after_<dir>_<mag>.png, meta.json) for two
scenes and builds a publication figure:

  rows  = scenes (pulling, cutting)
  cols  = [ before | drag A | drag B | ... | diff(A) heatmap ]

Each edit panel overlays the drag arrow (centroid -> tip) and faint control-node
dots. The diff column is |after-before| magnitude in a perceptually-uniform
colormap with a shared colorbar. Writes NEW files under docs/figures/ (never
overwrites existing images).

Example:
  .venv/bin/python tools/compose_edit_figure.py --mag 0.12 --dirs +y,-y,+x \
      --yflip --out docs/figures/edit_generality.png
"""
import os, json, argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

DIRTAG = {"+y": "py", "-y": "my", "+x": "px", "-x": "mx", "+z": "pz", "-z": "mz"}
DIRLABEL = {"+y": "drag ↓", "-y": "drag ↑", "+x": "drag →", "-x": "drag ←",
            "+z": "drag out", "-z": "drag in"}


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def diff_map(after, before):
    """Per-pixel L2 colour difference, 0..1 image space -> (H,W) magnitude."""
    return np.linalg.norm(after - before, axis=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pulling", default="output/endonerf/pulling_match/edit_fig")
    ap.add_argument("--cutting", default="output/endonerf/cutting_match/edit_fig")
    ap.add_argument("--mag", type=float, default=0.12)
    ap.add_argument("--dirs", type=str, default="+y,-y,+x")
    ap.add_argument("--yflip", action="store_true", help="flip projected node y (3DGS NDC convention)")
    ap.add_argument("--diff_dir", type=str, default="", help="direction to show in the diff column (default: first)")
    ap.add_argument("--dmax", type=float, default=0.0, help="diff colorbar max (0 = auto per-figure 99th pct)")
    ap.add_argument("--node_alpha", type=float, default=0.35)
    ap.add_argument("--out", default="docs/figures/edit_generality.png")
    args = ap.parse_args()

    dirs = [d.strip() for d in args.dirs.split(",") if d.strip()]
    diff_dir = args.diff_dir or dirs[0]
    scenes = [("pulling_soft_tissues", args.pulling), ("cutting_tissues_twice", args.cutting)]
    ncol = 1 + len(dirs) + 1                      # before + edits + diff
    fig = plt.figure(figsize=(2.55 * ncol, 2.15 * len(scenes) + 0.35))
    gs = fig.add_gridspec(len(scenes), ncol, wspace=0.04, hspace=0.10,
                          left=0.005, right=0.93, top=0.93, bottom=0.02)

    # first pass to fix a common diff scale
    all_diffs, cache = [], {}
    for _, d in scenes:
        meta = json.load(open(os.path.join(d, "meta.json")))
        before = load(os.path.join(d, "before.png"))
        cache[d] = {"meta": meta, "before": before, "after": {}}
        for dd in set(dirs + [diff_dir]):
            f = os.path.join(d, f"after_{DIRTAG[dd]}_{args.mag:.2f}.png")
            aft = load(f)
            cache[d]["after"][dd] = aft
            all_diffs.append(diff_map(aft, before))
    dmax = args.dmax if args.dmax > 0 else float(np.percentile(np.concatenate([x.ravel() for x in all_diffs]), 99))
    norm = Normalize(vmin=0.0, vmax=max(dmax, 1e-6))
    cmap = cm.get_cmap("inferno")

    diff_ax_for_bar = None
    for r, (sname, d) in enumerate(scenes):
        meta = cache[d]["meta"]; before = cache[d]["before"]
        H, W = before.shape[:2]
        spx = np.array(meta["sel_px"]); spy = np.array(meta["sel_py"])
        cpx, cpy = meta["centroid_px"], meta["centroid_py"]
        if args.yflip:
            spy = H - spy; cpy = H - cpy

        # column 0: before
        ax = fig.add_subplot(gs[r, 0]); ax.imshow(before); ax.set_xticks([]); ax.set_yticks([])
        ax.scatter(spx, spy, s=4, c="#39ff14", alpha=args.node_alpha, linewidths=0, zorder=3)
        if r == 0: ax.set_title("before", fontsize=11)
        ax.set_ylabel(sname.replace("_", "\n"), fontsize=9)

        # edit columns
        for c, dd in enumerate(dirs, start=1):
            aft = cache[d]["after"][dd]
            ax = fig.add_subplot(gs[r, c]); ax.imshow(aft); ax.set_xticks([]); ax.set_yticks([])
            # arrow tip for this direction/mag
            tip = next(e for e in meta["edits"] if e["dir"] == dd and abs(e["mag"] - args.mag) < 1e-6)
            tpx, tpy = tip["arrow_tip_px"], tip["arrow_tip_py"]
            if args.yflip:
                tpy = H - tpy
            ax.scatter(spx, spy, s=4, c="#39ff14", alpha=args.node_alpha * 0.8, linewidths=0, zorder=3)
            ax.annotate("", xy=(tpx, tpy), xytext=(cpx, cpy),
                        arrowprops=dict(arrowstyle="-|>", color="#00e5ff", lw=2.2, shrinkA=0, shrinkB=0), zorder=4)
            if r == 0: ax.set_title(DIRLABEL[dd], fontsize=11)

        # diff column
        ax = fig.add_subplot(gs[r, ncol - 1]); ax.set_xticks([]); ax.set_yticks([])
        gray = before.mean(axis=2)
        ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
        dm = diff_map(cache[d]["after"][diff_dir], before)
        ax.imshow(np.ma.masked_less(dm, dmax * 0.06), cmap=cmap, norm=norm, alpha=0.9)
        if r == 0: ax.set_title(f"edit magnitude ({DIRLABEL[diff_dir]})", fontsize=10)
        diff_ax_for_bar = ax

    # shared colorbar
    cax = fig.add_axes([0.94, 0.10, 0.012, 0.76])
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax).set_label("|after − before|", fontsize=9)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[compose] wrote {args.out}  (mag={args.mag}, dirs={dirs}, yflip={args.yflip}, dmax={dmax:.3f})")


if __name__ == "__main__":
    main()
