"""Emit SEPARATE publication panels for the GC-EndoGaussian generality edit figure.

Runs locally (repo .venv: numpy/PIL/matplotlib). For one scene, reads the
edit_figure.py renders (before.png, after_<dir>_<mag>.png, meta.json) and writes
individual full-bleed PNGs (overlays baked in), never overwriting existing files:

  <prefix>_before.png          before + dashed handle-region outline
  <prefix>_<name>.png          after + drag arrow + handle outline   (one per --dirs entry)
  <prefix>_diff.png            grayscale + |after-before| heatmap (--diff_dir)

A shared colorbar is written once via --colorbar_out.

Example:
  .venv/bin/python tools/compose_panels.py \
     --dir output/endonerf/pulling_match/edit_fig --mag 0.08 \
     --dirs +y:down,-y:up,+x:right --diff_dir +y \
     --prefix docs/figures/edit_gen_pulling --dmax 0.35 \
     --colorbar_out docs/figures/edit_gen_colorbar.png
"""
import os, json, argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Circle

DIRTAG = {"+y": "py", "-y": "my", "+x": "px", "-x": "mx", "+z": "pz", "-z": "mz"}
CMAP = "inferno"


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def full_bleed(img_shape):
    H, W = img_shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, W); ax.set_ylim(H, 0)      # image coords, y down
    return fig, ax, dpi


def handle_outline(ax, cpx, cpy, rp, color="#39ff14", lw=2.0):
    ax.add_patch(Circle((cpx, cpy), rp, fill=False, ec=color, lw=lw, ls=(0, (4, 3)), alpha=0.9, zorder=4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="edit_fig dir with before/after/meta")
    ap.add_argument("--mag", type=float, required=True)
    ap.add_argument("--dirs", required=True,
                    help="comma list of dir:name[:mag], e.g. +y:down,-x:left:0.16 (mag defaults to --mag)")
    ap.add_argument("--diff_dir", default="+y", help="dir[:mag] for the diff panel")
    ap.add_argument("--prefix", required=True, help="output path prefix (new files)")
    ap.add_argument("--dmax", type=float, default=0.35, help="fixed diff colorbar max (shared across scenes)")
    ap.add_argument("--yflip", action="store_true")
    ap.add_argument("--no_arrow", action="store_true", help="draw only the handle circle, no drag arrow")
    ap.add_argument("--circle_on_diff", action="store_true", help="also draw the handle circle on the diff panel")
    ap.add_argument("--colorbar_out", default="")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.dir, "meta.json")))
    before = load(os.path.join(args.dir, "before.png"))
    H, W = before.shape[:2]
    spx = np.array(meta["sel_px"]); spy = np.array(meta["sel_py"])
    cpx, cpy = meta["centroid_px"], meta["centroid_py"]
    if args.yflip:
        spy = H - spy; cpy = H - cpy
    # region radius in pixels: 80th pct of node distance to centroid (visible handle footprint)
    rp = float(np.percentile(np.hypot(spx - cpx, spy - cpy), 80))
    os.makedirs(os.path.dirname(args.prefix), exist_ok=True)

    # --- before panel ---
    fig, ax, dpi = full_bleed(before.shape)
    ax.imshow(before, extent=(0, W, H, 0))
    handle_outline(ax, cpx, cpy, rp)
    ax.plot([cpx], [cpy], marker="o", ms=5, color="#39ff14", zorder=5)
    fig.savefig(f"{args.prefix}_before.png", dpi=dpi); plt.close(fig)

    # --- edit panels ---
    for spec in [s for s in args.dirs.split(",") if s.strip()]:
        parts = spec.split(":")
        dd, name = parts[0], parts[1]
        mag = float(parts[2]) if len(parts) > 2 else args.mag
        aft = load(os.path.join(args.dir, f"after_{DIRTAG[dd]}_{mag:.2f}.png"))
        tip = next(e for e in meta["edits"] if e["dir"] == dd and abs(e["mag"] - mag) < 1e-6)
        tpx, tpy = tip["arrow_tip_px"], tip["arrow_tip_py"]
        if args.yflip:
            tpy = H - tpy
        # clip arrow to a readable length (point in the drag direction from centroid)
        vx, vy = tpx - cpx, tpy - cpy
        L = np.hypot(vx, vy) + 1e-6
        arrow_len = min(L, rp * 1.7 + 40)
        ex, ey = cpx + vx / L * arrow_len, cpy + vy / L * arrow_len
        fig, ax, dpi = full_bleed(aft.shape)
        ax.imshow(aft, extent=(0, W, H, 0))
        handle_outline(ax, cpx, cpy, rp, lw=1.8)
        if not args.no_arrow:
            ax.annotate("", xy=(ex, ey), xytext=(cpx, cpy),
                        arrowprops=dict(arrowstyle="-|>", color="#00e5ff", lw=3.0,
                                        mutation_scale=26, shrinkA=0, shrinkB=0), zorder=6)
        fig.savefig(f"{args.prefix}_{name}.png", dpi=dpi); plt.close(fig)

    # --- diff panel ---
    ddp = args.diff_dir.split(":")
    diff_dd = ddp[0]; diff_mag = float(ddp[1]) if len(ddp) > 1 else args.mag
    aft = load(os.path.join(args.dir, f"after_{DIRTAG[diff_dd]}_{diff_mag:.2f}.png"))
    dm = np.linalg.norm(aft - before, axis=2)
    norm = Normalize(vmin=0.0, vmax=args.dmax)
    fig, ax, dpi = full_bleed(before.shape)
    ax.imshow(before.mean(2), cmap="gray", vmin=0, vmax=1, extent=(0, W, H, 0))
    ax.imshow(np.ma.masked_less(dm, args.dmax * 0.06), cmap=CMAP, norm=norm, alpha=0.92, extent=(0, W, H, 0))
    if args.circle_on_diff:
        handle_outline(ax, cpx, cpy, rp, lw=1.8)
    fig.savefig(f"{args.prefix}_diff.png", dpi=dpi); plt.close(fig)

    if args.colorbar_out:
        fig = plt.figure(figsize=(0.5, 2.6), dpi=150)
        cax = fig.add_axes([0.05, 0.06, 0.35, 0.88])
        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
        cb.set_label("|after − before|", fontsize=9)
        cb.set_ticks([0, args.dmax / 2, args.dmax])
        fig.savefig(args.colorbar_out, dpi=150, bbox_inches="tight"); plt.close(fig)

    print(f"[compose_panels] {args.prefix}_* written (mag={args.mag}, rp={rp:.0f}px, dmax={args.dmax})")


if __name__ == "__main__":
    main()
