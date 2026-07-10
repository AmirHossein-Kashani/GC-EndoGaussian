"""Aggregate the seed study + edit metrics + training wall-clock into paper-ready numbers.

1. Seed study: mean +/- std over seeds {6666 (paper default, existing runs), 1234, 2025, 3407}
   for the four Table-3 configs on pulling @ 3000 fine iterations.
2. Edit metrics: one-row-per-model summary table from edit_metrics.json.
3. Locality-curve figure across models (docs/figures/fig_edit_locality.pdf).

Run on the login node after the jobs finish: python tools/aggregate_paper_updates.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS_NEW = [1234, 2025, 3407, 4242]   # 4242 = substitute after scgs@3407's reproducible crash
COLLAPSE_PSNR = 20.0                   # runs below this diverged; excluded from mean, counted
CONFIGS = {           # tag -> (existing seed-6666 model dir, seed-study prefix)
    "vanilla":     ("pulling",             "vanilla"),
    "match":       ("pulling_match3k",     "match"),
    "scgs":        ("pulling_scgs",        "scgs"),
    "scgs_hybrid": ("pulling_scgs_hybrid", "scgs_hybrid"),
}
LABELS = {"vanilla": "EndoGaussian", "match": "GC-EndoGaussian",
          "scgs": "SC-GS-style, no residual", "scgs_hybrid": "SC-GS-style + residual"}
METRICS = ["PSNR", "SSIM", "LPIPS", "RMSE"]


def read_results(path):
    p = os.path.join(path, "results.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["ours_3000"]


print("=" * 72)
print("SEED STUDY (pulling @ 1000+3000, seeds: 6666 + %s)" % SEEDS_NEW)
print("=" * 72)
stats = {}
for tag, (base_dir, prefix) in CONFIGS.items():
    rows = []
    r = read_results(f"output/endonerf/{base_dir}")
    if r: rows.append((6666, r))
    for s in SEEDS_NEW:
        r = read_results(f"output/endonerf/seed_study/{prefix}_s{s}")
        if r: rows.append((s, r))
    if not rows:
        print(f"{tag}: NO RESULTS"); continue
    collapsed = [(s, r) for s, r in rows if r["PSNR"] < COLLAPSE_PSNR]
    ok = [(s, r) for s, r in rows if r["PSNR"] >= COLLAPSE_PSNR]
    stats[tag] = {}
    line = f"{tag:12s} (n={len(ok)} converged, {len(collapsed)} collapsed): "
    for m in METRICS:
        vals = np.array([r[m] for _, r in ok])
        stats[tag][m] = (float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
        line += f"{m}={vals.mean():.3f}+/-{vals.std(ddof=1) if len(vals)>1 else 0:.3f}  "
    print(line)
    print("   seeds:", [s for s, _ in rows],
          " PSNRs:", [round(r["PSNR"], 3) for _, r in rows])
    if collapsed:
        print("   COLLAPSED seeds:", [s for s, _ in collapsed])

if {"vanilla", "match"} <= stats.keys():
    d = stats["vanilla"]["PSNR"][0] - stats["match"]["PSNR"][0]
    print(f"\nDelta PSNR (vanilla - match): {d:.3f} dB")
if {"scgs", "scgs_hybrid"} <= stats.keys():
    d = stats["scgs_hybrid"]["PSNR"][0] - stats["scgs"]["PSNR"][0]
    print(f"Residual gain (scgs_hybrid - scgs): {d:.3f} dB")

print()
print("=" * 72)
print("EDIT METRICS")
print("=" * 72)
EDIT_MODELS = [("pulling_match3k", "GC-EndoGaussian (3k)"),
               ("pulling_scgs_hybrid", "SC-GS-style + residual"),
               ("pulling_scgs", "SC-GS-style, no residual"),
               ("pulling_match", "GC-EndoGaussian (6k)")]
curves = {}
hdr = f"{'model':26s} {'fidelity':>10s} {'leak3d%':>8s} {'leakpx%':>8s} {'fold%':>7s} {'strain95':>9s} {'lat ms':>7s} {'res.frac':>9s}"
print(hdr)
for mdir, lbl in EDIT_MODELS:
    p = f"output/endonerf/{mdir}/edit_metrics.json"
    if not os.path.exists(p):
        print(f"{lbl:26s}  MISSING ({p})"); continue
    d = json.load(open(p))
    ef = d.get("energy_split") or {}
    print(f"{lbl:26s} {d['handle_fidelity']['median']:>10.3f} "
          f"{100*d['leak3d']['median']:>8.2f} "
          f"{100*d['leak_px']['median'] if d.get('leak_px') else float('nan'):>8.2f} "
          f"{100*d['foldover']['median']:>7.3f} {d['strain_p95']['median']:>9.3f} "
          f"{d['latency_ms']:>7.1f} {ef.get('residual_frac', float('nan')):>9.3f}")
    curves[lbl] = d["locality_curve"]

# ---- training wall-clock per config, parsed from the seed-study job logs ----
# Section start = first timestamped train.py line after the "######## <tag> seed=" header;
# section end = its "Training complete. [dd/mm HH:MM:SS]" line (render/metrics excluded).
import re, glob
print()
print("=" * 72)
print("TRAINING WALL-CLOCK (pulling @ 1000+3000, from seed-study logs)")
print("=" * 72)
TS = re.compile(r"\[(\d\d)/(\d\d) (\d\d):(\d\d):(\d\d)\]")
def ts_seconds(m):
    return int(m.group(2)) * 86400 + int(m.group(3)) * 3600 + int(m.group(4)) * 60 + int(m.group(5))
durations = {}
for logf in sorted(glob.glob("output_seed_study_*.out")) + sorted(glob.glob("output_seed_retry_*.out")):
    tag, start = None, None
    for line in open(logf, errors="ignore"):
        h = re.match(r"######## (\w+) seed=", line)
        if h:
            tag, start = h.group(1), None
            continue
        if tag is None:
            continue
        m = TS.search(line)
        if m and start is None:
            start = ts_seconds(m)
        if "Training complete." in line and m and start is not None:
            durations.setdefault(tag, []).append(ts_seconds(m) - start)
            tag, start = None, None
for tag, ds in sorted(durations.items()):
    print(f"{tag:12s}: {np.mean(ds)/60:.1f} +/- {np.std(ds)/60:.1f} min  (n={len(ds)}, {sorted(round(d/60,1) for d in ds)})")

# locality-curve figure: sharp fall-off + compact support is the message
if curves:
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    styles = {"GC-EndoGaussian (3k)": ("-", "C0"), "SC-GS-style + residual": ("--", "C1"),
              "SC-GS-style, no residual": (":", "C2"), "GC-EndoGaussian (6k)": ("-.", "C3")}
    for lbl, curve in curves.items():
        xs = [(c["bin_lo"] + c["bin_hi"]) / 2 for c in curve if c["median"] is not None]
        ys = [c["median"] for c in curve if c["median"] is not None]
        ls, col = styles.get(lbl, ("-", None))
        ax.plot(xs, ys, ls, color=col, label=lbl, lw=1.8, marker="o", ms=3)
    ax.axvspan(0, 0.10, color="0.92", zorder=0)
    ax.text(0.05, 0.55, "handle radius", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color="0.4", rotation=90)
    ax.annotate("identically zero beyond the\nbinding support ($2\\times$ radius)",
                xy=(0.25, 0.0), xytext=(0.26, 0.14), fontsize=8, color="0.3",
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
    ax.set_xlim(0, 0.35); ax.set_ylim(bottom=-0.01)
    ax.set_xlabel("distance to nearest handle node (fraction of scene extent)")
    ax.set_ylabel("median displacement / commanded")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    os.makedirs("docs/figures", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"docs/figures/fig_edit_locality.{ext}", dpi=200)
    print("\nwrote docs/figures/fig_edit_locality.png/.pdf")
