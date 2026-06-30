# Graph-Controlled EndoGaussian (GC-EndoGaussian)
### Topology-aware deformation for occlusion-robust endoscopic 4D reconstruction

**Status:** research proposal · **Base:** [EndoGaussian](../README.md) · **Target venue:** MICCAI / IPCAI / MedIA

---

## 1. Summary (TL;DR)

EndoGaussian reconstructs deforming surgical tissue by warping a canonical set of 3D Gaussians
with a per-Gaussian deformation field (a HexPlane k-planes grid + small MLP). That field deforms
each Gaussian **independently** — its only coupling between neighbors is the spatial smoothness of
the grid. This is enough for smooth, fully-observed motion, but it has no notion of **tissue
structure**: it cannot propagate motion into regions a surgical tool occludes, and it overfits when
timestamps are sparse or motion is fast.

We propose to **replace the position+rotation part of that field with a sparse control graph driven
by a Graph Neural Network**. A few hundred "hypernodes" are seeded in high-motion tissue regions;
each Gaussian is softly bound to several nearby nodes; a 2-layer GNN passes messages over the node
graph and emits a **per-node SE(3) transform** (rotation + translation); each Gaussian's movement
and rotation are recovered by a linear-blend-skinning (LBS) blend over its bound nodes. Because the
GNN runs on ~10³ nodes instead of ~10⁵ Gaussians, this is **equal-or-faster at render time** than the
base model, while making local coherence and occlusion-robust motion propagation an *architectural*
property rather than a soft penalty.

**One-line contribution:** *a topology-aware GNN control graph with learned Gaussian↔node binding
that produces coherent, occlusion-robust tissue deformation where per-Gaussian-independent
deformation (EndoGaussian) and per-node-independent control (SC-GS) both fail — at no extra
inference cost.*

---

## 2. Background: where EndoGaussian's deformation falls short

In the current pipeline, the fine stage queries a HexPlane field at each Gaussian's `(xyz, t)` and a
small MLP emits additive deltas `dx, ds, dr, do` for position / scale / rotation / opacity
([scene/deformation.py:57](../scene/deformation.py#L57)), applied per-Gaussian in
[gaussian_renderer/__init__.py:78](../gaussian_renderer/__init__.py#L78). The grid is spatially
continuous (resolution `[64, 64, 64, 25]`), so nearby Gaussians at the same time receive *similar*
deltas — an implicit **smoothness** prior.

That smoothness prior is real but limited. It is **not** a structural/topological prior, so three
failure modes remain:

1. **Tool–tissue occlusion.** When a tool occludes part of the surface, the occluded Gaussians get
   no photometric gradient. The grid simply keeps whatever it last learned there — it freezes or
   drifts — because nothing ties the occluded region's motion to the *visible* tissue around it.
2. **Sparse / fast frames.** The HexPlane is high-capacity (6 planes × multi-resolution). With few
   timestamps or large inter-frame motion it overfits the observed frames and interpolates the
   unobserved ones poorly.
3. **Geometric incoherence.** Per-point deltas can be locally inconsistent (small non-physical
   shear/jitter) even when the rendered RGB looks fine — visible in depth/geometry, not PSNR.

These are exactly the regimes that matter clinically (instruments constantly occlude tissue) and
exactly where a structural prior on connected tissue should help.

---

## 3. Related work & positioning (the gap we must clear)

The "sparse control nodes warp dense Gaussians" idea already exists — we must be explicit about our
delta:

- **SC-GS (CVPR 2024)** — sparse control points, each with a *per-node MLP* predicting a 6-DoF
  transform; Gaussians warped by **fixed distance-based LBS**; ARAP rigidity as a *loss*; adaptive
  node density. This is the closest prior work and overlaps with the naive version of our idea.
- **Node-guided family** — MB-GS (learnable weight painting), Mango-GS, DG-4DGS, HEIR (graph
  motion hierarchies). Node-based control of dynamic Gaussians is a crowded space.
- **Surgical 4DGS** — EndoGaussian (our base), EndoGS, Deform3DGS, Endo-4DGS, SurgicalGaussian,
  EH-SurGS, and newer 2025/26 work. PSNR on standard EndoNeRF clips is already ~37–38 (saturated).

**Our delta vs SC-GS (the defensible novelty):**

| | SC-GS | GC-EndoGaussian (ours) |
|---|---|---|
| Node motion | per-node MLP, **independent** | **GNN message passing** — each node conditioned on its neighborhood |
| Neighbor coupling | only via ARAP **loss** | built into the **forward map** (generalizes to unobserved `t` & occluded regions) |
| Gaussian↔node weights | fixed, distance-based | **learned** (weight painting) |
| Motivation | general dynamic scenes | **surgical**: single deforming surface, tool occlusion, depth supervision |
| Headline claim | editable dynamic NVS | **occlusion-robust, geometrically-coherent** tissue deformation |

The GNN turning neighbor-coupling from a *loss* into an *architectural inductive bias* is the
technical core; **occlusion-robust motion propagation** is the framing that gives it a reason to
exist in surgery.

---

## 4. Our idea: Graph-Controlled Deformation

### 4.1 Overview

Keep the entire EndoGaussian base — canonical 3DGS init, depth-supervised two-stage coarse→fine
training, densification/pruning, the EndoNeRF/SCARED/Hamlyn data pipeline. **Replace** only the
position+rotation deformation with a control graph. Crucially we *replace*, not *add*: if the
full HexPlane ran in parallel it would swallow the signal and the graph would learn nothing.

### 4.2 Hypernodes & motion-aware seeding

Seed `M ≈ 512–2048` nodes at the start of the fine stage. **Seed by motion, not density** — dense
regions of the static cloud are not where deformation is hard; the high-strain pulled/cut region is
often *sparse*. We reuse the motion signal the repo already accumulates, `_deformation_accum`
([scene/gaussian_model.py:491](../scene/gaussian_model.py#L491)), as a sampling weight for
**farthest-point sampling** (coverage + concentration in high-motion tissue). Nodes are adaptively
split/pruned through training, mirroring SC-GS.

### 4.3 Soft multi-node binding (learned)

Each Gaussian is bound to its `K ≈ 4` nearest nodes with soft weights `w_ik` (a Gaussian can — and
should — belong to multiple nodes). Weights are initialized from distance, then **refined by a small
learned weight-painting head** — a second novelty axis over SC-GS's purely distance-based weights.
Stored as per-Gaussian tensors `_binding_idx (N,K)`, `_binding_w (N,K)`.

### 4.4 GNN message passing → per-node SE(3)

Per timestamp `t`:
1. Node input feature `h⁰_m = [γ(node_xyz_m), timenet(t), node_feat_m]` (optionally seeded with the
   HexPlane feature at the node, to fuse a global-motion cue).
2. **2–3 message-passing layers** over the node KNN graph (edge features `γ(n_n − n_m)`), residual
   EdgeConv-style aggregation — each node's update sees its 2–3-hop neighborhood.
3. Output head emits a **per-node SE(3)**: rotation `R_m` (6-D parameterization) + translation `t_m`.

### 4.5 LBS assembly → per-Gaussian movement *and* rotation

For Gaussian `i` with bindings `{(k, w_ik)}`:
- **Movement:** `dx_i = Σ_k w_ik · [ R_k (x_i − n_k) + t_k + n_k − x_i ]`
- **Rotation:** `dr_i = blend_k( w_ik · R_k )` applied to the canonical quaternion
- **Scale / opacity:** kept on a *tiny* residual MLP head (or dropped).

This yields exactly the additive `dx`/`dr` that `render()` already consumes — both the *movement and
rotation* come from the graph, which is the behavior we want.

### 4.6 Regularization

Reuse the existing stage-gated regularizer hook ([train.py:161](../train.py#L161)):
- **ARAP node rigidity:** `Σ ‖ (p_m − p_n) − R_m (n_m − n_n) ‖²` over node edges (≤16k edges, cheap).
- **Node-trajectory temporal smoothness:** second-difference of node translation/rotation across
  `t ± Δ` — the node-space analogue of the current hexplane time-smoothness, much better targeted.
- (optional) binding-consistency keeping LBS valid as the cloud densifies.

---

## 5. Why it improves (and where it won't)

| Regime | EndoGaussian smoothness covers it? | GC-EndoGaussian improves? |
|---|---|---|
| Standard pulling/cutting, full views | Yes (PSNR saturated) | ❌ ~0 PSNR/SSIM gain — *do not sell on this* |
| **Tool occlusion** | No — grid freezes/drifts in unobserved region | ✅ graph propagates motion from visible neighbor nodes |
| **Sparse / fast frames** | No — high-capacity grid overfits observed `t` | ✅ low-dim node graph + ARAP generalizes |
| **Geometry / depth quality** | Partially | ✅ ARAP + node SE(3) → physically plausible surfaces |

The improvements live in **robustness and geometry**, not the saturated PSNR table — so the paper is
framed and evaluated around occlusion/sparse-view/geometry, not standard NVS averages.

---

## 6. Computational cost: equal-or-faster

| | Per-frame deformation cost |
|---|---|
| EndoGaussian | `O(N)` HexPlane + MLP at **every** Gaussian (N ≈ 1–3×10⁵) |
| GC-EndoGaussian | `O(M)` GNN on M ≈ 1024 nodes  +  `O(N·K)` gather + weighted-add LBS (**no per-Gaussian nonlinearity**) |

We trade `O(N)` MLP evaluations for `O(M)` GNN evaluations plus cheap arithmetic; since `M ≪ N`,
**render FPS is equal-or-higher** than the base model (the same reason SC-GS is faster than
per-Gaussian MLP deformation). Honest caveat: we add modest **memory** (the `(N,K)` bindings + node
params) and **train-time** work (KNN graph build every ~500 iters, ARAP loss) — small, not zero.

---

## 7. Potential contributions / improvements

1. **Topology-aware deformation for surgical 4DGS** — GNN message passing makes local tissue
   coherence an architectural property, not just a soft loss.
2. **Occlusion-robust motion propagation** — motion flows from visible to tool-occluded tissue
   through the graph; the headline, clinically-grounded result.
3. **Learned Gaussian↔node binding** (weight painting) — beyond SC-GS's distance-only weights.
4. **Motion-aware node seeding** — nodes allocated by deformation, not static density.
5. **Real-time, equal-or-faster than the EndoGaussian base** — a clean secondary selling point.
6. **A geometric-coherence evaluation protocol** (depth RMSE + deformation-Jacobian smoothness +
   occlusion/sparse-view stress tests) suited to the saturated-PSNR regime.

---

## 8. Experimental plan

**Baselines.** (1) vanilla EndoGaussian (must-beat); (2) **SC-GS reimplementation = our method with
the GNN removed** (per-node MLP + distance-LBS + ARAP) — *the decisive ablation*; (3) Deform3DGS;
(4) optional Endo-4DGS / SurgicalGaussian.

**Datasets.** EndoNeRF pulling + cutting, SCARED, Hamlyn (`test_every=8`, 640×512).

**Metrics.** PSNR / SSIM / LPIPS **plus** depth RMSE / abs-rel (geometry) **plus** a
deformation-Jacobian smoothness metric (coherence), and FPS / train time / memory / #nodes.

**Ablations.** GNN layers L∈{0,1,2,3} (L=0 ≈ SC-GS); M∈{256…2048}; K∈{1,2,4,8}; seeding
density-only vs FPS vs **motion-FPS**; fixed vs **learned binding**; ±ARAP; ±node-temporal;
replace-dx vs hybrid vs MLP-only.

**Stress tests (the crux).**
1. **Tool-occlusion holdout** — mask a tissue region for a block of training frames; evaluate
   render + depth there when it reappears.
2. **Sparse views** — aggressively subsample timestamps.
3. **Fast non-rigid frames** — highest-motion cutting sub-sequences; report error vs motion magnitude.

---

## 9. Risks & honest assessment

| Risk | Severity | Mitigation |
|---|---|---|
| Novelty vs SC-GS | **Highest** | headline = occlusion robustness; ship SC-GS as ablation; win stress tests; lean on surgical framing |
| GNN real-time cost | Low | GNN on ≤2048 nodes is negligible; verify FPS ≥ base |
| Dynamic-graph maintenance under densification | Medium | bindings inherit from parents on clone/split; full re-seed every ~500 iters |
| PSNR saturation hides gains | High | sell on depth RMSE / Jacobian smoothness / robustness, not PSNR |
| Crowded, fast-moving field | Medium | due-diligence at submission: confirm no surgical node/graph-deformation paper already exists |

**MICCAI positioning (calibrated).** Base acceptance ~30%. The idea is an architectural variation on
an existing concept in a crowded subfield, so the outcome is dominated by **execution and the
empirical result**, not the concept: strong execution with a clear occlusion/geometry win → **~35–45%**;
marginal results / only a standard PSNR table → **~15–20%**; realistic point estimate for a solid
first attempt → **~30–35%** (legitimate but not safe). Decisive levers: the occlusion result and a
crisp one-sentence "vs SC-GS" delta backed by the no-GNN ablation.

---

## 10. Milestones / next step

1. **Gate experiment (≈1 day, do this first).** Prototype the deformation swap with a
   `--gnn_layers 0` switch; run the **occlusion-holdout** on `pulling`; compare full vs `gnn_layers=0`
   on held-out-region depth RMSE. *Clear gap → build the full system; no gap → stop.*
2. Full module + densification-safe graph maintenance + ARAP/temporal regularizers.
3. Full benchmark (baselines + ablations + 3 stress tests) across 4 datasets.
4. Writing, framed around occlusion-robust coherent deformation.

---

## Appendix: integration points in this repo

- **New** `scene/node_deformation.py` — `NodeGraphDeformation` (seeding, KNN graph, binding, GNN, LBS).
- `scene/deformation.py` — instantiate the module; in `forward_dynamic`
  ([:57](../scene/deformation.py#L57)) have the graph **own** `dx`/`dr`.
- `scene/gaussian_model.py` — `_binding_idx`/`_binding_w` state; thread through
  `densification_postfix` ([:389](../scene/gaussian_model.py#L389)) and `prune_points`
  ([:350](../scene/gaussian_model.py#L350)); add `"nodes"`/`"gnn"`/`"binding"` optimizer groups in
  `training_setup` ([:168](../scene/gaussian_model.py#L168)); new `refresh_node_graph()`.
- `gaussian_renderer/__init__.py` — slice bindings by the `_deformation_table` mask at the call site
  ([:78](../gaussian_renderer/__init__.py#L78)); no signature change.
- `train.py` — node-refresh hook around densify/prune ([:217](../train.py#L217)); graph regularizers
  in the stage-gated loss block ([:161](../train.py#L161)).
- `arguments/__init__.py` + new `arguments/endonerf/pulling_graph.py` — new hyperparameters and config.

*Constraints preserved: no change to the Python 3.12 / torch 2.5.1 / CUDA stack; new module is pure
PyTorch (no heavy new deps unless explicitly approved).*
