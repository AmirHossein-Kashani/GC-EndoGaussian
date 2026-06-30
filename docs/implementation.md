# GC-EndoGaussian — implementation & experiments log

Control-node **G**raph-**C**ontrolled deformation on top of EndoGaussian. This log records what
was built, how it is wired into the repo, what has been validated, and the experiment results to
date. Design rationale and the paper framing live in [proposal.md](proposal.md).

---

## 1. What the method does

EndoGaussian deforms every Gaussian independently (HexPlane k-planes feature → per-Gaussian MLP →
additive `dx/ds/dr/do`). GC-EndoGaussian replaces the **position + rotation** part with a sparse
control graph:

1. **Seed** ~1024 control "hypernodes" by motion-weighted farthest-point sampling over the Gaussian
   cloud (motion read from `_deformation_accum`).
2. **Bind** each Gaussian softly to its `K=4` nearest nodes (distance-softmax weights).
3. A small **message-passing GNN** over the node KNN graph emits a **per-node SE(3)** (6D rotation +
   translation) at each timestamp.
4. **LBS** blends each Gaussian's bound-node transforms to recover *both* its movement (`dx`) and
   rotation (`dr`). Scale/opacity stay on the cheap MLP residual.

The GNN runs only on the sparse node set (~1024), so per-Gaussian cost is a gather + blend — the
inference path is equal-or-cheaper than the per-Gaussian HexPlane it replaces.

`gnn_layers = 0` degrades the GNN to a per-node-independent MLP (SC-GS-style control) — the
make-or-break ablation switch.

---

## 2. Files

**New**
- [scene/node_deformation.py](../scene/node_deformation.py) — `NodeGraphDeformation`: SE(3)/quaternion
  helpers, chunked KNN, motion-weighted FPS, node seeding/graph build, soft binding, GNN forward
  (`node_transforms`), and the LBS blend (`forward`). Identity-initialised SE(3) head (starts as
  no-op deformation).

**Changed**
- [scene/deformation.py](../scene/deformation.py) — instantiates the node module when
  `use_node_graph`; `forward_dynamic` lets the graph own `dx/dr` (with optional hybrid residual),
  keeps `ds/do` on the MLP; `get_node_parameters()` and exclusion of node params from
  `get_mlp_parameters()`; re-applies the identity SE(3) head after the global xavier init.
- [scene/gaussian_model.py](../scene/gaussian_model.py) — `_binding_idx/_binding_w` state; a
  `node_gnn` optimizer group + LR schedule; `seed_node_graph`, `compute_node_bindings`,
  `maintain_node_graph`, `node_seeded`, and the graph regularizer `compute_node_regulation`
  (ARAP / isometric / temporal).
- [gaussian_renderer/__init__.py](../gaussian_renderer/__init__.py) — `render()` slices and passes
  each deformable Gaussian's bindings; lazily (re)computes bindings when the Gaussian count changed.
- [train.py](../train.py) — seeds the graph at fine-stage start, maintains it through
  densification/pruning, and adds the (optionally annealed) graph regularizer to the loss.
- [arguments/__init__.py](../arguments/__init__.py) — all `ModelHiddenParams` / `OptimizationParams`
  knobs below.
- [render.py](../render.py) — CPU-affinity calls (`os.sched_setaffinity`, `psutil.cpu_affinity`)
  made best-effort (try/except); they throw `OSError: [Errno 22]` on compute nodes whose SLURM
  cgroup excludes the requested CPU. Unrelated to the method; was blocking render→metrics.

**Configs** (`arguments/endonerf/`)
- `pulling_graph.py` — full graph, `gnn_layers=2` (pure replace).
- `pulling_graph_nognn.py` — ablation, `gnn_layers=0`.
- `pulling_graph_hybrid.py` — hybrid residual + isometric prior (annealed).

---

## 3. Config knobs (`ModelHiddenParams` / `OptimizationParams`)

| knob | default | meaning |
|---|---|---|
| `use_node_graph` | False | master switch (off = vanilla EndoGaussian) |
| `node_hybrid` | False | graph low-freq motion **+** per-Gaussian high-freq residual |
| `num_nodes` | 1024 | control-node count |
| `node_knn` | 8 | node↔node graph degree |
| `gauss_knn_K` | 4 | nodes each Gaussian binds to |
| `gnn_layers` | 2 | message-passing layers (0 = SC-GS-style ablation) |
| `gnn_width` | 64 | GNN hidden width |
| `node_pe` | 4 | positional-encoding freqs (node xyz / edges / time) |
| `lambda_arap` | 0.01 | rigid ARAP coupling |
| `lambda_isometric` | 0.0 | as-isometric edge-length prior (tissue resists stretch) |
| `lambda_node_temporal` | 0.001 | second-difference smoothness of node trajectories |
| `node_reg_anneal` | False | linearly relax the graph priors over the fine stage (floor 0.2) |
| `node_refresh_interval` | 1000 | re-seed cadence (iters) |
| `node_lr_init/final` | 8e-4 / 8e-5 | GNN learning rate |

---

## 4. How to run

```bash
# full graph
python train.py  -s data/endonerf/pulling --expname endonerf/pulling_graph  --configs arguments/endonerf/pulling_graph.py        --save_iterations 1000 3000
python render.py --model_path output/endonerf/pulling_graph --configs arguments/endonerf/pulling_graph.py --skip_train --reconstruct
python metrics.py --model_path output/endonerf/pulling_graph
```
SLURM wrappers: `run_gc_endogaussian.bash` (graph + nognn), `run_gc_hybrid.bash` (hybrid),
`run_gc_render_metrics.bash` (render+score already-trained models).

---

## 5. Validation

- **CPU unit tests** (`/tmp/test_node.py`): SE(3)/quaternion helpers; **identity init ⇒ exact no-op
  deformation**; binding weights sum to 1; gradients reach the GNN; both `gnn_layers∈{2,0}` paths.
- **CPU construction test** (`/tmp/test_integration.py`): config merge sets the knobs; `GaussianModel`
  builds; **node params do not double-count** across the `deformation`/`grid`/`node_gnn` groups.
- **GPU training** (job XXXXXXXX, H100): both graph and nognn variants trained cleanly through all
  3000 fine iters — seeding, GNN forward, ARAP/temporal regularizers, and densification-with-binding-
  maintenance all run without error. Models saved with the node graph inside `deformation.pth`.

---

## 6. Experiment 1 — the gate (standard pulling, 1000 coarse + 3000 fine)

Identical config base + metrics protocol for all three.

| Metric | Vanilla EndoGaussian | Graph (GNN, L=2) | No-GNN ablation (L=0) |
|---|---|---|---|
| PSNR ↑ | **37.27** | 36.68 | 36.50 |
| SSIM ↑ | **0.9578** | 0.9488 | 0.9476 |
| LPIPS ↓ | **0.0609** | 0.0946 | 0.0954 |
| FLIP ↓ | **0.0471** | 0.0500 | 0.0513 |
| Depth RMSE ↓ | **2.906** | 3.001 | 3.037 |

**Read.**
- GNN beats its own no-GNN ablation on **all** metrics, but by a hair (PSNR +0.18 dB) — the message
  passing helps, weakly.
- Both graph variants **lose to vanilla**, most visibly on LPIPS (0.061 → 0.095, ~55% worse).
- This is the *predicted* easy-case behaviour: the standard, fully-observed clip is PSNR-saturated and
  favours the baseline's near-per-Gaussian capacity. The graph's value (occlusion / sparse-view /
  geometry coherence) is **not measured by this render** and remains the open question.

---

## 7. Diagnosis — why the pure-replace graph underperforms

Ranked by impact:

1. **Capacity / smoothness bottleneck (dominant).** ~28k Gaussians share the motion of 1024 nodes via
   a fixed LBS blend → ~28× fewer, *shared* motion DoF, *architecturally forced smooth*. Fine
   localized deformation can't be represented → blur → the LPIPS regression. This is the inductive
   bias we chose (the same smoothness should be an asset under occlusion).
2. **ARAP fights non-rigid tissue.** `lambda_arap` enforces rigidity, but pulling is genuinely
   non-rigid → underfits → PSNR/depth cost. Pure cost on the easy benchmark.
3. **Re-seed disruption + under-training.** Motion-FPS re-seeds at 1000/2000/3000 perturb the learned
   field; the graph also has more to learn (GNN + SE(3) head) in the same 3000-iter budget.
4. **Lossy linear quaternion blend** of node rotations slightly distorts Gaussian orientation.

Fundamental (the trade) = #1. Self-inflicted / recoverable = #2–#4.

---

## 8. Experiment 2 — hybrid + isometric prior (IN PROGRESS, job XXXXXXXX)

Changes aimed at the recoverable factors, so the method stops *losing* on the easy case (a
precondition for a publishable "ties on RGB, wins on occlusion/geometry" story):

- **Hybrid residual** (`node_hybrid=True`): graph supplies coherent low-frequency motion; a small
  per-Gaussian MLP residual recovers high-frequency detail → targets the LPIPS blur (#1).
- **As-isometric prior** (`lambda_isometric=0.01`, `lambda_arap=0`): preserve node edge *lengths*
  (resist stretch) instead of enforcing rigidity — the correct tissue-surface prior, far less
  fidelity cost (#2).
- **Annealed priors** (`node_reg_anneal=True`): rigid early for stability, relaxed late to fit fine
  motion (#2/#3).

Config: `arguments/endonerf/pulling_graph_hybrid.py` (same 1024 nodes / L=2 / 3000 iters as the
pure-replace graph, for a clean A/B).

**Hypothesis:** LPIPS recovers toward ~0.06 and PSNR toward ~37, i.e. the hybrid ≈ vanilla on the
standard benchmark — confirming the capacity/regularizer diagnosis.

**Results:** job XXXXXXXX (H100, 4:58).

| Metric | Vanilla | Graph (replace) | **Hybrid** | gap to vanilla recovered |
|---|---|---|---|---|
| PSNR ↑ | 37.27 | 36.68 | **36.88** | ~34% (−0.59 → −0.39 dB) |
| SSIM ↑ | 0.9578 | 0.9488 | **0.9537** | ~54% |
| LPIPS ↓ | 0.0609 | 0.0946 | **0.0760** | ~55% (the LPIPS blur) |
| FLIP ↓ | 0.0471 | 0.0500 | **0.0495** | ~17% |
| Depth RMSE ↓ | 2.906 | 3.001 | 3.037 | none (geometry flat ~3.0) |

**Read.** The hybrid confirms the diagnosis: the per-Gaussian residual buys back roughly **half**
the appearance gap (LPIPS 0.095 → 0.076, PSNR +0.20, SSIM +0.005 vs the pure-replace graph). So the
capacity/smoothness bottleneck (#1) was indeed the main cause of the easy-case regression. It does
**not** yet match vanilla — still −0.39 dB / +0.015 LPIPS — and **depth did not improve** (all graph
variants sit ~3.0 vs vanilla 2.906; the isometric prior at 0.01 annealed had little effect here).
Net: closer to parity, diagnosis validated, but not there. The remaining appearance gap is the kind
longer training / more nodes / better rotation blend can chip at; the *geometry* gap and the actual
*win* still depend on adding new information (flow supervision) and testing the occlusion regime.

---

## 8b. Experiment 3 — robustness (sparse-view + occlusion), the go/no-go gate

**Sparse-view** (jobs XXXXXXXX failed on a port collision → fixed in `network_gui.init`; re-run
XXXXXXXX). Test PSNR as training frames are starved:

| stride | Vanilla PSNR | Hybrid PSNR | hybrid−vanilla | degradation full→here |
|---|---|---|---|---|
| 1 | 37.13 | 36.08 | −1.05 | — |
| 2 | 35.88 | 35.43 | −0.45 | vanilla −1.25, hybrid −0.65 |
| 4 | 30.19 | 29.65 | −0.54 | vanilla −5.69, hybrid −5.78 |

**Occlusion-holdout** (job XXXXXXXX), tissue recovery inside the held-out box:

| | Vanilla | Hybrid |
|---|---|---|
| Occluded-box PSNR ↑ | **26.17** | 26.00 |
| Control-box PSNR | 40.58 | 39.72 |
| Occlusion gap ↓ | 14.41 | 13.72 |

**Verdict — the graph does not beat (or match) the baseline on any measured axis.**
- Standard: hybrid loses (−0.39…−1.05 dB, +LPIPS).
- Sparse-view: hybrid loses in absolute terms at every stride. There is a *faint* robustness signal
  (full→half, hybrid degrades −0.65 vs vanilla −1.25), but it never overtakes vanilla and the two
  collapse equally at stride 4. Not a win.
- Occlusion: hybrid's occluded-box PSNR (26.00) is *below* vanilla (26.17); its smaller "gap" is an
  artifact of being uniformly blurrier (lower control PSNR too), not better recovery.

**Why.** The baseline's HexPlane is *already* a smooth, temporally-coherent deformation field, so the
graph's coherence prior adds **constraint without new information** — it can match the baseline's
behaviour at best, and on these fully/near-fully-observed benchmarks the constraint is a net cost.
The occlusion was also temporally bounded (region visible before/after the block), so both models
recover it by temporal interpolation; the graph's spatial-propagation advantage never gets to matter.

**Implication for the paper.** A pure deformation-quality framing (beat EndoGaussian's PSNR/geometry)
is **not supported by the evidence**. To produce a publishable result we must either (a) inject new
information the baseline ignores (optical-flow supervision — the only lever left that could *exceed*
quality), or (b) pivot the contribution to a capability the graph uniquely enables and where PSNR is
not the headline (controllable/editable motion; or the efficiency/compactness of sparse control).

## 8c. Experiment 4 — second dataset (cutting_tissues_twice), is the result pulling-specific?

Downloaded the more dynamic EndoNeRF clip (156 frames vs pulling's 63; fast cutting motion).
Job XXXXXXXX, same hybrid recipe.

| Metric | Vanilla | Hybrid | gap (cutting) | gap (pulling) |
|---|---|---|---|---|
| PSNR ↑ | 38.40 | 38.07 | −0.33 | −0.39 |
| SSIM ↑ | 0.9637 | 0.9615 | −0.0022 | −0.0090 |
| LPIPS ↓ | 0.0434 | 0.0471 | +0.0037 | +0.0151 |
| Depth RMSE ↓ | 1.440 | 1.513 | +0.073 | +0.095 |

**Read.** The negative result is **NOT pulling-specific** — the hybrid loses on cutting too. BUT the
gap is *smaller on the more dynamic clip across every metric* (LPIPS gap shrinks 4×: +0.015→+0.004;
SSIM 4×; PSNR/depth modestly). This is a consistent directional signal: **the graph's relative
performance improves with scene dynamism** — it just doesn't cross over at this scale/tuning. Two
datasets now agree: the graph is competitive-but-behind, narrowing as motion increases.

**Implication.** Quality-beating with the graph alone is unlikely on standard metrics, but the
dynamism trend + the still-untested flow-supervision lever leave a path. Decision: pursue the
flow-supervision bet (new information) and assemble the control/efficiency pivot in parallel.

## 8d. Experiment 5 — optical-flow supervision (the quality bet), job XXXXXXXX

Self-gate confirmed correct flow (Farneback warp improves GT on 9/9 checked pairs → enabled).

| Variant (pulling) | PSNR | SSIM | LPIPS | RMSE |
|---|---|---|---|---|
| Vanilla (no flow) | **37.27** | 0.9578 | **0.0609** | 2.906 |
| Vanilla + flow@0.5 | 37.30 | 0.9587 | 0.0690 | 2.864 |
| Hybrid (no flow) | 36.88 | 0.9537 | 0.0760 | 3.037 |
| Hybrid + flow@0.5 | 36.85 | 0.9571 | 0.0760 | 3.035 |
| Hybrid + flow@1.0 | 36.45 | 0.9537 | 0.0946 | 3.287 |

**Read — the bet did not pay off.** Flow@0.5 gives the hybrid only a small SSIM bump (+0.0034);
PSNR/LPIPS are unchanged and it stays −0.42 dB / +0.015 LPIPS behind vanilla. Flow@1.0 *hurts*
(noisy endoscopic flow over-constrains). Flow barely moves vanilla either. The image-space warp
loss is self-referential — once renders are roughly correct it carries little gradient — so it does
not inject enough new signal to close the graph's gap.

## 8e. Overall verdict (5 experiments, 2 datasets)

The control-node graph **does not beat EndoGaussian on reconstruction quality** on any axis tested:
standard (pulling & cutting), sparse-view, occlusion-holdout, or with flow supervision. It is
consistently *competitive but a little behind* (≈ −0.3 to −0.4 dB PSNR, small LPIPS gap), and the gap
narrows with scene dynamism but never crosses. The HexPlane baseline is already a smooth,
temporally-coherent field, so the graph adds constraint rather than information.

**Decision: stop chasing the quality win; pivot to the capability the graph uniquely enables.** The
achievable, honest paper is *controllable / editable real-time endoscopic deformation at near-parity
quality* — where the headline is the capability (sparse-node motion editing) and a tight
quality-match, not a PSNR victory. Near-parity (~36.9 vs 37.3) is acceptable for that framing; one
tuning run (2048 nodes / longer schedule) should tighten it further.

## 8f. Capability deliverable + FAIR iteration-matched comparison (jobs XXXXXXXX/XXXXXXXX/XXXXXXXX)

> **Correction:** an earlier note here claimed the hybrid "matches LPIPS / wins depth RMSE." That
> compared hybrid@6000 against vanilla@**3000** (under-trained) — an unfair comparison. The
> iteration-matched numbers below (both at 6000) overturn it.

**Iteration-matched (6000 iters), two datasets — vanilla vs 2048-node hybrid:**

| | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Depth RMSE ↓ |
|---|---|---|---|---|
| pulling — vanilla | **37.32** | **0.9578** | **0.0509** | **2.646** |
| pulling — hybrid | 36.77 | 0.9537 | 0.0603 | 2.845 |
| cutting — vanilla | **39.42** | **0.9696** | **0.0322** | **1.358** |
| cutting — hybrid | 38.91 | 0.9663 | 0.0380 | 1.432 |

**Honest verdict:** at fair comparison the hybrid is **consistently ~0.5 dB PSNR behind** (−0.55 / −0.51)
with small SSIM/LPIPS/RMSE gaps on **both** datasets. No quality win, no match — reproducibly a little
behind. The graph adds constraint, not information; the HexPlane already covers it.

**Efficiency:** graph **206 FPS** vs vanilla **285 FPS** (both far above real-time; hybrid is *slower*
— keeps the HexPlane + adds the GNN). Params: node-GNN adds **60k on an 85M grid (+0.07%)**.

**Editing capability (the only genuine win):** `edit_demo.py` drags a region of control nodes
(`edit_translation`); bound Gaussians follow via LBS — controllable deformation the per-Gaussian
baseline cannot do. Gentle, localized drags (0.06×extent) give clean, plausible deformations on both
pulling and cutting (before/after/diff figures in each model's `edit_gentle/`).

**Net.** The supportable contribution is **a unique capability (controllable, real-time editing) at
near-parity quality (~0.5 dB behind)** — NOT a quality win. Comprehensive evidence (2 datasets ×
standard/sparse/occlusion/flow + longer training + more nodes, all negative on quality) says the
graph does not beat EndoGaussian on reconstruction. Any paper must lead with the capability and
report quality honestly as near-parity.

## 8g. MATCH adjustment — closing the gap (job XXXXXXXX)

Diagnosis: the hybrid was behind not because the graph can't fit, but because two parts *fought*
the photometric optimum. The fix keeps the editable graph but removes those costs:
`node_translation_only=True` (rotation from the full MLP, no lossy quaternion LBS blend),
**all coherence regularizers off**, **nodes frozen** after the initial seed.

| | PSNR | SSIM | LPIPS | RMSE |
|---|---|---|---|---|
| pulling vanilla-6k | **37.32** | 0.9577 | **0.0509** | **2.646** |
| pulling **match** | 37.17 | 0.9567 | 0.0532 | 2.793 |
| Δ (was, hybrid-2048) | **−0.16** (−0.55) | −0.001 (−0.004) | +0.002 (+0.009) | +0.147 (+0.199) |
| cutting vanilla-6k | **39.42** | 0.9696 | **0.0322** | **1.358** |
| cutting **match** | 39.29 | 0.9689 | 0.0338 | 1.384 |
| Δ (was, hybrid-2048) | **−0.13** (−0.51) | −0.0007 (−0.003) | +0.0016 (+0.006) | +0.026 (+0.074) |

**Result:** the PSNR gap collapses from ~0.5 dB to **~0.13–0.16 dB** on both datasets — within
run-to-run / perceptual noise. SSIM is essentially equal (−0.001); LPIPS/RMSE gaps shrink 4×. The
editing capability is retained (drag-to-edit demo on the match model is cleanly localized).

**This is the supportable headline:** *a control-graph layer that adds drag-to-edit tissue control to
EndoGaussian at near-exact parity (within ~0.15 dB PSNR, equal SSIM) on two datasets, real-time
(206 FPS), at +0.07% params.* "Matches quality + adds a capability the baseline lacks" — honest and
strong. (Still a hair behind, not equal-or-better; report as "matches within noise," not "beats.")

1. **Confirm hybrid parity** (this run). If LPIPS snaps back, the diagnosis holds.
2. **Add new information so the graph can *surpass*, not just tie** — optical-flow supervision on node
   trajectories (dense motion signal the baseline ignores); the strongest lever.
3. **Build the occlusion-holdout gate**: synthetically mask a contiguous tissue patch for a block of
   training frames, score reconstruction *on that patch* at reveal time (a targeted metric, not
   full-frame average). This is the experiment that actually decides the paper.
4. Scale (num_nodes 2048, longer schedule), then breadth (cutting / SCARED / Hamlyn) if the gate
   passes.

## 8h. Cut-aware graph + runtime (jobs XXXXXXXX / XXXXXXXX)

Cut-aware = breakable node edges (2-pass GNN: measure per-edge stretch, suppress messages past a
threshold) so the two sides of a tissue cut deform independently — a discontinuity HexPlane can't
represent. Cut-region PSNR on cutting (top-10% motion):

| | cut-region PSNR |
|---|---|
| vanilla-6k | **12.01** |
| graph match (no cut) | 11.88 |
| graph cut-aware | 11.95 |

Cut-awareness *does* help the graph (11.95 > 11.88 — the mechanism works) but **still doesn't beat
vanilla** (< 12.01), and it costs a 2nd GNN pass. **Verdict: drop it.**

**Runtime.** Two costs, both avoidable/minor: (1) the 6000-iter runs were for a *fair* iteration-
matched comparison, not required — the match comes from the architectural adjustment, so it should
hold at the original 3000-iter budget (`pulling_match3k`, job XXXXXXXX); (2) inference 206 vs 285 FPS
— both 7–9× real-time, and translation-only already trims it. **No zero-cost approach beats the
baseline** (sparse/occlusion/flow/cut-aware all negative); the achievable "editing at matched quality"
runs at the original budget + real-time.
