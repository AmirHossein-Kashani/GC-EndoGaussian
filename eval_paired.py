"""Paired comparison of two models' per-point tracking error (Leg 1 statistical rigor).
Reads tracking_results.json from two model dirs and reports the paired vanilla-vs-graph difference
with a bootstrap CI + sign test (+ Wilcoxon if scipy is available).

Usage: python eval_paired.py output/endonerf/super_vanilla output/endonerf/super_match
"""
import sys, json
import numpy as np

def load(p):
    return np.asarray(json.load(open(p + "/tracking_results.json"))["per_point_mean"], dtype=np.float64)

a, b = load(sys.argv[1]), load(sys.argv[2])      # a = vanilla, b = graph (paired by point index)
n = min(len(a), len(b)); a, b = a[:n], b[:n]
d = a - b                                         # >0 => graph lower error (better)

print(f"paired points: {n}")
print(f"  {sys.argv[1]} median: {np.median(a):.3f} px")
print(f"  {sys.argv[2]} median: {np.median(b):.3f} px")
print(f"  median paired diff (first-second): {np.median(d):+.3f} px  (>0 => second is better)")
rng = np.random.default_rng(0)
bs = np.array([np.median(rng.choice(d, n, replace=True)) for _ in range(10000)])
print(f"  95% CI of median diff: [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
print(f"  second better on {(d > 0).sum()}/{n} points (sign test)")
try:
    from scipy.stats import wilcoxon
    _, pv = wilcoxon(a, b)
    print(f"  Wilcoxon signed-rank p = {pv:.4g}")
except Exception as e:
    print(f"  (scipy Wilcoxon unavailable: {type(e).__name__})")
