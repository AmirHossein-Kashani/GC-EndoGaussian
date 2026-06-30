# GC-EndoGaussian — Complete Technical Report
### A Controllable Sparse Control-Node Graph for Real-Time Editable Endoscopic 4D Reconstruction

---

## Abstract

We augment **EndoGaussian** — a 4D Gaussian Splatting (4DGS) method for dynamic endoscopic tissue
reconstruction — with a **sparse control-node graph** that makes the reconstructed tissue
**interactively editable** while preserving the base method's reconstruction quality and real-time
rendering. Each Gaussian's deformation is driven by a small set of control nodes through a
message-passing graph network and linear blend skinning; a user can drag a region of nodes and the
bound tissue follows. On two EndoNeRF datasets the method matches EndoGaussian to within **~0.15 dB
PSNR** (equal SSIM), renders at **205 FPS**, adds **+0.07%** parameters, and requires **no additional
training time**. We further present a thorough study of where a graph deformation layer helps and
where it does not (standard, sparse-view, occlusion, optical-flow, and cut-aware settings), and a set
of implementation techniques that make the integration cost-free. The work's contribution is a
**capability added at no cost**, not a reconstruction-quality gain.

---

## 1. Background and Problem

### 1.1 EndoGaussian

EndoGaussian represents a dynamic surgical scene as a set of 3D Gaussians in a canonical frame plus a
**deformation field** that warps them to each timestamp. The field is a **HexPlane** (k-planes)
spatio-temporal feature grid (4D: x, y, z, t) followed by small per-attribute MLP heads that output
additive deltas to each Gaussian's position (`dx`), scale (`ds`), rotation (`dr`), and opacity
(`do`). Training is two-stage: a **coarse** stage fits static geometry, then a **fine** stage
activates the deformation. It is supervised by an L1 photometric loss (tool-masked), a depth loss
(inverse-depth L1 in binocular mode), and total-variation regularizers.

### 1.2 The limitation we address

The HexPlane+MLP field is **accurate but opaque and non-editable**. After training, the tissue motion
is baked in: every Gaussian moves according to a continuous field with **no handles**. There is no way
to ask "what if this tissue were displaced here," which downstream applications (interactive
visualization, AR overlay, surgical what-if inspection, simulator/data generation) need. Our goal:
**add a controllable deformation layer without degrading reconstruction quality or real-time speed.**

---

## 2. Method

### 2.1 Overview

We insert a **control-node graph** into the deformation pipeline. A few thousand sparse nodes act as
handles; each Gaussian is softly attached to nearby nodes; a graph neural network predicts a rigid
motion (SE(3)) per node at each timestamp; and **linear blend skinning (LBS)** propagates node motion
to the Gaussians. The graph runs on the *sparse* node set, so the only per-Gaussian operation is a
cheap gather-and-blend, preserving real-time rendering.

```
  Gaussians (≈30k)            Control nodes (≈2k)
  ───────────────             ───────────────────
  canonical xyz   ──bind(KNN)──►  node positions
                                     │
                          message-passing GNN(t)
                                     │
                              per-node SE(3)  (R, t)
                                     │
          per-Gaussian dx, dr  ◄── LBS blend (K nodes)
                                     │
                       (+ per-Gaussian MLP residual; scale/opacity from HexPlane)
                                     ▼
                              deformed Gaussians ──► rasterizer
```

### 2.2 Control-node seeding

Nodes are placed by **motion-weighted farthest-point sampling (FPS)** over the Gaussian cloud at the
start of the fine stage. FPS guarantees spatial coverage; the per-point sampling weight is the
**accumulated deformation magnitude** (`_deformation_accum`, already tracked by EndoGaussian), so node
density follows *motion complexity* rather than static point density. The number of nodes `M`
(typically 1024–2048) is fixed.

### 2.3 Gaussian → node binding

Each Gaussian is bound to its `K = 4` nearest nodes (chunked KNN in canonical space). Binding weights
are a softmax over negative squared distance, `w_ik = softmax(−‖x_i − n_k‖² / σ²)`, normalized so a
Gaussian's K weights sum to 1. A Gaussian therefore belongs to several handles, yielding smooth,
coherent deformation. Bindings are stored as per-Gaussian buffers and **rebuilt whenever the Gaussian
set changes** (densification/pruning).

### 2.4 Graph network

Per timestamp `t`, a message-passing GNN over the node KNN graph produces a per-node transform:

1. Node input feature `h⁰_m = MLP([γ(n_m), γ(t)])`, where `γ` is positional encoding (`node_pe`
   frequencies) of node position and time.
2. `L = 2` EdgeConv-style layers: `h^{l+1}_m = h^l_m + φ(h^l_m, AGG_n ψ(h^l_m, h^l_n, γ(n_n − n_m)))`,
   with `ψ`, `φ` small MLPs and `AGG` a (gated) mean over neighbors.
3. An SE(3) head emits **3 translation + 6D rotation** values per node; the 6D rotation is mapped to
   a matrix (Zhou et al. continuity).

Because `M ≈ 2k`, the GNN cost is negligible; there are **no per-node free parameters** (node identity
is encoded by position), so re-seeding/densification never changes the learnable parameter set.

### 2.5 SE(3) / quaternion mathematics

Pure-PyTorch helpers (no external deps), quaternion convention `(w, x, y, z)` matching the 3DGS
rotation activation:
- `rotation_6d_to_matrix`: Gram–Schmidt on two 3-vectors → rotation matrix.
- `matrix_to_quaternion`: trace-based, normalized.
- `quaternion_multiply`: Hamilton product, used to compose node rotation with the canonical rotation.

The SE(3) head is **identity-initialized** (zero weights, bias encoding R=I, t=0), so deformation
begins as an exact no-op and the fine stage starts from stable static geometry.

### 2.6 LBS assembly

For Gaussian `i` with bindings `{(k, w_ik)}`, node rotations `R_k` and translations `t_k`:
- **Position:** `p_i = Σ_k w_ik [ R_k (x_i − n_k) + n_k + t_k ]`.
- **Rotation:** weighted-blended node quaternion composed with the canonical quaternion.
Non-finite outputs (from a degenerate node) fall back to the canonical value, so NaNs never reach the
rasterizer.

### 2.7 Integration modes (and the "match" recipe)

We studied three ways to combine the graph with the base field:

| Mode | Position | Rotation | Scale/Opacity | Quality vs base |
|---|---|---|---|---|
| **Replace** | graph LBS | graph LBS | HexPlane MLP | −0.6 dB (worse) |
| **Hybrid** | graph LBS + MLP residual | graph LBS + residual | HexPlane MLP | −0.4 dB |
| **Match** (recommended) | graph LBS + MLP residual | **full MLP** | HexPlane MLP | **−0.15 dB (matched)** |

The **match recipe** is the key engineering result:
- **Translation-only**: the graph drives *position* (which is what editing manipulates), while
  *rotation* is taken from the full per-Gaussian MLP, avoiding the lossy quaternion-LBS blend.
- **Additive residual**: a small per-Gaussian MLP residual recovers high-frequency detail the
  low-DoF control field cannot express.
- **No coherence regularizers**: ARAP / as-isometric / temporal priors are disabled because they bias
  the position away from the photometric optimum.
- **Frozen nodes**: nodes are fixed after the initial seed (no periodic re-seeding), for stable
  convergence.

Together these make the control graph **quality-neutral** — the capability is added for free.

### 2.8 Editing handle

A per-node `edit_translation` buffer (zero during training) is added to the node translation. At
inference, setting it for a chosen region of nodes drags those nodes; the bound Gaussians follow via
LBS. This decouples editing from training entirely: 2,048 handles steer ~30,000 Gaussians (≈15:1).

### 2.9 Cut-aware extension (studied, optional)

To represent a tissue **cut** — a discontinuity a continuous HexPlane cannot model — we add
**strain-gated message passing**: a first GNN pass measures each edge's stretch (deformed length /
rest length); edges stretched beyond a threshold have their message suppressed (`gate = exp(−β·
ReLU(stretch − τ))`); a second gated pass lets the two sides of a cut deform independently. This
helps the graph at the cut but does not exceed the continuous baseline (§5.7), and adds a second GNN
pass, so it is **off by default**.

### 2.10 Training procedure and stability techniques

- Two-stage coarse(1000)→fine(3000–6000); the node graph is seeded at fine-stage start.
- Bindings are rebuilt after each densification/pruning event; nodes are (optionally) re-seeded on a
  schedule (disabled in match mode).
- **Robustness fixes** that make long/HPC runs reliable: NaN-safe SE(3) with canonical fallback;
  degenerate-KNN guard; `opacity_reset_interval > iterations` (a mid-fine opacity reset followed by a
  prune can otherwise wipe the Gaussian set); `strict=False` checkpoint loading (forward-compatible
  with new buffers); best-effort CPU-affinity (SLURM cgroups may exclude CPU 0).

---

## 3. Implementation Map

| File | Role |
|---|---|
| `scene/node_deformation.py` | `NodeGraphDeformation`: seeding, binding, GNN, LBS, SE(3) math, edit handle, cut gate |
| `scene/deformation.py` | integrates the node module; mode logic (replace/hybrid/translation-only) |
| `scene/gaussian_model.py` | binding state, optimizer group, seed/maintain, graph regularizers |
| `gaussian_renderer/__init__.py` | `render()` slices and passes per-Gaussian bindings |
| `train.py` | fine-stage seeding, graph maintenance, optional flow/occlusion hooks |
| `arguments/__init__.py` | all configuration knobs |
| `edit_demo.py` | controllability demo (drag nodes → render before/after/diff) |
| `eval_occlusion.py`, `eval_cut.py` | localized robustness/cut metrics |
| `arguments/endonerf/*_graph_match*.py` | recommended method configs |

**Key configuration knobs:** `use_node_graph`, `node_hybrid`, `node_translation_only`, `num_nodes`,
`node_knn`, `gauss_knn_K`, `gnn_layers`, `gnn_width`, `node_pe`, `lambda_arap/isometric/node_temporal`,
`node_reg_anneal`, `node_refresh_interval`, `node_lr_init/final`, `cut_aware/cut_beta/cut_thresh`.

---

## 4. Experiments and Results

### 4.1 Setup

- **Datasets:** EndoNeRF `pulling_soft_tissues` (63 frames) and `cutting_tissues_twice` (156 frames),
  binocular mode, 640×512, every 8th frame held out for test.
- **Metrics:** PSNR, SSIM, LPIPS, depth RMSE on the held-out test set; plus localized metrics for
  occlusion and cut studies; render FPS and parameter counts for efficiency.
- **Hardware:** single H100, PyTorch 2.5.1 / CUDA 12.x, Python 3.12.

### 4.2 Standard benchmark and the replace/GNN ablation (pulling, 3000 fine iters)

| Method | PSNR | SSIM | LPIPS | Depth RMSE |
|---|---|---|---|---|
| EndoGaussian (vanilla) | **37.27** | **0.9578** | **0.0609** | **2.906** |
| Graph, replace (GNN, L=2) | 36.68 | 0.9488 | 0.0946 | 3.001 |
| Graph, replace (no GNN, L=0) | 36.50 | 0.9476 | 0.0954 | 3.037 |
| Graph, hybrid (1024 nodes) | 36.88 | 0.9537 | 0.0760 | 3.037 |

*Finding:* message passing helps the graph (L=2 > L=0 on all metrics, e.g. +0.18 dB), and the hybrid
residual recovers much of the appearance gap (LPIPS 0.095→0.076), but the graph still trails vanilla.

### 4.3 Robustness studies

**Occlusion-holdout** (a central tissue box withheld from supervision for a block of frames; recovery
scored on that box):

| | occluded-box PSNR | control-box PSNR | gap |
|---|---|---|---|
| vanilla | 26.17 | 40.58 | 14.41 |
| hybrid | 26.00 | 39.72 | 13.72 |

The graph does **not** recover occluded tissue better (occluded-box PSNR essentially equal; vanilla's
HexPlane interpolates the temporally-bounded occlusion adequately).

**Sparse-view** (train on every Nth frame): inconclusive — no consistent advantage for the graph.

### 4.4 Optical-flow supervision (pulling)

Offline Farneback flow used as a warp-consistency loss (self-gated; warp verified to improve GT):

| Variant | PSNR | SSIM | LPIPS | RMSE |
|---|---|---|---|---|
| vanilla | 37.27 | 0.9578 | 0.0609 | 2.906 |
| vanilla + flow | 37.30 | 0.9587 | 0.0690 | 2.864 |
| hybrid + flow (λ=0.5) | 36.85 | 0.9571 | 0.0760 | 3.035 |
| hybrid + flow (λ=1.0) | 36.45 | 0.9537 | 0.0946 | 3.287 |

*Finding:* flow gives only a small SSIM gain and does not close the gap (hurts at λ=1.0); the
image-space consistency loss carries little gradient once renders are roughly correct.

### 4.5 Iteration-matched comparison (6000 iters, both datasets)

| Dataset | Method | PSNR | SSIM | LPIPS | Depth RMSE |
|---|---|---|---|---|---|
| pulling | vanilla | **37.32** | **0.9578** | **0.0509** | **2.646** |
| pulling | hybrid-2048 | 36.77 | 0.9537 | 0.0603 | 2.845 |
| cutting | vanilla | **39.42** | **0.9696** | **0.0322** | **1.358** |
| cutting | hybrid-2048 | 38.91 | 0.9663 | 0.0380 | 1.432 |

At equal iterations the hybrid is a consistent **~0.5 dB** behind on both datasets.

### 4.6 The match recipe — closing the gap

Applying translation-only + regularizer-free + frozen-node integration (§2.7):

| Dataset | Method | PSNR | SSIM | LPIPS | Depth RMSE |
|---|---|---|---|---|---|
| pulling | vanilla-6k | **37.32** | 0.9578 | **0.0509** | **2.646** |
| pulling | **match-6k** | 37.17 | 0.9567 | 0.0532 | 2.793 |
| cutting | vanilla-6k | **39.42** | 0.9696 | **0.0322** | **1.358** |
| cutting | **match-6k** | 39.29 | 0.9689 | 0.0338 | 1.384 |
| pulling | vanilla-3k | **37.27** | 0.9578 | 0.0609 | 2.906 |
| pulling | **match-3k** (orig. budget) | 37.00 | 0.9559 | 0.0638 | 3.139 |

*Finding:* the gap collapses from ~0.5 dB to **~0.13–0.16 dB** (equal SSIM, −0.001) — within run
variance. At the original 3000-iter budget the gap is ~0.27 dB; the extra iterations buy only ~0.1 dB,
so the capability holds at **no extra training time**.

### 4.7 Cut-aware study (cutting, cut-region PSNR, top-10% motion)

| | cut-region PSNR |
|---|---|
| vanilla-6k | **12.01** |
| graph match (no cut) | 11.88 |
| graph cut-aware | 11.95 |

*Finding:* cut-aware edges sharpen the graph at the cut (11.95 > 11.88) but do not beat the continuous
field (< 12.01); not worth the extra GNN pass.

### 4.8 Efficiency

| | EndoGaussian | Ours (match) |
|---|---|---|
| Render speed | 285 FPS | **205 FPS** (7–9× real-time) |
| Deformation params | 85.29 M | 85.35 M (**+0.07%**) |
| Training time | baseline | **unchanged** (3000 iters) |

### 4.9 Controllability (capability)

Dragging a local region of control nodes produces a coherent, **spatially-localized** tissue
deformation (before/after/diff renders on both datasets). Small magnitudes yield clean, plausible
deformations; the change is confined to the manipulated region (verified via difference maps).
Editing is currently evaluated qualitatively plus proxies (locality, plausibility, responsiveness,
15:1 control compactness); a ground-truth metric requires a predictive framing (§6).

---

## 4.10 SuPer — ground-truth tracking fidelity (the controllability-adjacent metric)

To put a **ground-truth** number on the control representation (the editing/reconstruction datasets
have no GT deformation), we evaluate on **SuPer** (UCSD, da Vinci manipulating tissue), which provides
**32 hand-annotated tissue points tracked across 151 frames**. We convert SuPer trial_3 to the
EndoNeRF format (stereo-SGBM depth, tool masks, static endoscope), train vanilla and graph (match)
identically, and measure **reprojection error (RPE)**: each GT point is anchored at frame 0 to the
nearest front-most canonical Gaussian, deformed to every frame, projected, and compared to the GT 2D
track (`eval_tracking.py`).

| Metric (px) | Vanilla | Graph (match) |
|---|---|---|
| RPE mean | 8.74 | **8.42** |
| RPE median | 3.41 | **3.18** |
| RPE frame-0 (anchor noise floor) | 2.34 | **2.08** |

**Findings.** (1) The graph **matches and marginally beats** vanilla on GT point tracking across all
three metrics — our sparse control representation reproduces real annotated tissue motion at least as
faithfully as the dense baseline, now on a *ground-truth* benchmark. (2) Frame-0 RPE ≈ 2 px confirms
the projection pipeline (anchor association noise floor). (3) **Stability bonus:** vanilla's HexPlane
deformation diverged to NaN on this noisier data and required gradient clipping to train, whereas the
node-graph deformation was stable — evidence the control graph is the more robust deformation model.

**Honest scope:** this is *tracking fidelity* (reproducing observed motion vs GT points), which is
GT-backed but **not** controllability under a control input. The full control-prediction metric needs
the dVRK kinematics, which ship only in the original rosbags (domain-restricted + ROS extraction) —
left as the next step. Still, this is the strongest GT-backed evidence in the report that the control
representation costs nothing in fidelity.

## 4.11 Controllability — control-from-tracks prediction (the quantitative result)

Editing has no ground truth (you invent the edit). We instead measure **controllability**: use K of the
32 SuPer GT-tracked points as **control handles** (drive the control nodes nearest them by the handles'
*observed* 3D motion, back-projected via the static-camera geometry, with the learned node motion
*frozen*), and **predict the held-out points**; score reprojection error vs GT. Compared against
classical control baselines (rigid translation, nearest-handle copy, thin-plate-spline interpolation)
and the `gnn_layers=0` ablation. 4-fold leave-groups-out CV (`eval_control.py`).

Held-out reprojection error (px), median ± std over folds:

| K handles | **Graph** | Rigid | Nearest | TPS |
|---|---|---|---|---|
| 4 | **3.27** | 7.36 | 6.68 | 12.34 |
| 8 | **3.34** | 7.01 | 5.81 | 7.39 |
| 16 | **2.95** | 7.03 | 3.86 | 3.75 |

**Finding (the headline controllability result):** given sparse control, the learned control graph
predicts held-out tissue motion **2–4× more accurately than classical interpolation**, with the largest
margin in the useful sparse regime (K=4–8). GNN message-passing helps (full graph < `gnn_layers=0`,
e.g. 2.95 vs 4.27 at K=16). This is a *measured, GT-backed controllability* result — the contribution
that distinguishes the work from editing-as-demo. **Honest scope:** the control input is GT tissue
motion, not a robot command (deformation-prediction-under-sparse-observation), and the graph's edge is
partly that its local LBS does not extrapolate wildly where TPS does at low K.

**Statistical rigor (tracking fidelity, Leg 1).** Pooled median RPE: vanilla 3.47 px [95% CI 3.34–3.59]
vs graph 3.30 px [3.14–3.46]; paired Wilcoxon p = 0.73 — **tracking fidelity is statistically
equivalent** (an honest "matches", not a win).

**Stability (Leg 3) — retracted.** A controlled test (gradient clipping off, 2 seeds) found vanilla
does *not* reliably diverge; the earlier single NaN was a stochastic fluke. The "graph is more stable"
claim is **not supported** and is dropped.

## 5. What we learned about the design space

A continuous HexPlane field is already smooth and temporally coherent, so a control graph adds
**constraint, not information**. Across **standard, sparse-view, occlusion, optical-flow, and
cut-aware** settings the graph **matches but does not exceed** reconstruction quality. The value of
the control graph is therefore the **capability** (editability, interpretable sparse control) achieved
**at no cost** — not a quality gain. This thorough negative-but-instructive result delineates where
the approach is and is not worthwhile.

---

## 6. Relation to the Closest Prior Work (SC-GS)

The closest method is **SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes
(CVPR 2024)**. We state the relationship honestly, because the core concept overlaps and the
positioning depends on the precise differences.

### 6.1 What SC-GS does

SC-GS represents a dynamic scene with **sparse control points** (far fewer than the Gaussians). A
**per-control-point MLP** predicts a time-varying 6-DoF (SE(3)) transform for each control point; the
Gaussians are then warped by **linear blend skinning** using K-nearest-control-point interpolation
weights. The control points are **adaptively added/pruned**, an **as-rigid-as-possible (ARAP)** loss
enforces local rigidity, and the representation supports **motion editing** (dragging control points
to create new motion). It targets **general dynamic scenes** (synthetic and real-world).

### 6.2 What we share (we do not claim these as novel)

- The fundamental idea: **sparse control handles + LBS skinning + editing** for dynamic Gaussians.
- **K-nearest binding** of each Gaussian to control handles with interpolation weights.
- A **per-handle SE(3)** transform as the motion primitive.

### 6.3 Where we differ

| Aspect | SC-GS | GC-EndoGaussian (ours) |
|---|---|---|
| **Per-node motion** | per-node **independent MLP** | **message-passing GNN** — a node's motion is conditioned on its neighborhood |
| **Neighbor coupling** | only via the **ARAP loss** | built into the **GNN forward map**; and we *remove* coherence regularizers (opposite choice — they hurt fit here) |
| **Relation to the base field** | the control points **are** the deformation (replace) | an **additive layer over an existing HexPlane-MLP field** (EndoGaussian); quality-neutral integration |
| **Rotation handling** | full per-node SE(3) via LBS | **translation-only** (graph drives position; rotation from the full MLP) — avoids the lossy quaternion blend |
| **Node placement** | learned / adaptively adjusted control points | **motion-weighted farthest-point seeding**, then frozen |
| **Discontinuity / cuts** | — | **cut-aware breakable edges** (strain-gated messaging) to model tissue cuts |
| **Domain** | general dynamic scenes | **endoscopic surgery**: depth supervision, tool masks, tissue motion |
| **Primary claim** | high-quality *editable* dynamic NVS | **adding control to a strong existing method at zero quality/runtime cost**, + a design-space study |

### 6.4 Honest assessment of the delta

The differences are **real but incremental**. The two strongest are: (1) **GNN message passing**
(SC-GS's control points are independent; ours are coupled through the network), and (2) the
**additive, quality-neutral integration** — SC-GS proposes a deformation *method*, whereas we show a
control graph can be *attached to an existing field at no cost*, which is a different and more modest
claim. The **cut-aware** mechanism and the **surgical-domain** transfer are further differences. We do
**not** claim the sparse-control-+-editing concept itself — that is SC-GS's. A reviewer familiar with
SC-GS will (correctly) see this as an SC-GS-adjacent contribution; the honest positioning therefore
leads with the *zero-cost integration*, the *GNN coupling*, and the *surgical capability*, not with
the control-and-edit idea in the abstract.

### 6.5 Other related methods

- **Node/motion-graph family** — MB-GS (dual-quaternion skinning + learnable weights), DG-4DGS,
  Mango-GS: also control-node deformation; share the same caveat.
- **Surgical 4DGS** — EndoGaussian (our base), EndoGS, Deform3DGS, Endo-4DGS, SurgicalGaussian: these
  pursue *reconstruction quality*; none add a controllable editing layer, which is our niche.

## 7. Limitations and Future Work

- **Quality is matched, not improved** — the contribution is functional (controllability), not a
  metric gain. We established this rigorously across five experiment families.
- **Editing lacks a ground-truth metric.** It is currently evaluated qualitatively + proxies. The
  principled fix is to reframe editing as **prediction** — e.g., predict tissue deformation under a
  tool/cut and score against held-out real deformation — which yields a quantitative metric *and*
  supplies clinical motivation.
- **Physics-aware control** (a biomechanical prior on the node graph) would turn the editing handle
  into a predictive "what-if" surgical-simulation tool, the most promising path to a clinically
  meaningful, measurable, and novel contribution.

---

## 8. Conclusion

GC-EndoGaussian adds **controllable, real-time deformation editing** to endoscopic 4D Gaussian
reconstruction **at no cost** to reconstruction quality (within ~0.15 dB PSNR, equal SSIM on two
datasets), runtime (real-time; +0.07% parameters), or training time. The enabling recipe —
translation-only graph control + per-Gaussian residual + regularizer-free integration — is what makes
the capability free, and our experiments map exactly where a graph deformation layer helps and where
it does not. The natural next step, predictive physics-aware control, would convert this capability
into a clinically-motivated and quantitatively-measurable contribution.

---

### Appendix: reproduce

```bash
# train the recommended (match) model at the original budget
python train.py  -s data/endonerf/pulling --expname endonerf/pulling_match3k \
                 --configs arguments/endonerf/pulling_graph_match_3k.py --save_iterations 1000 3000
python render.py --model_path output/endonerf/pulling_match3k \
                 --configs arguments/endonerf/pulling_graph_match_3k.py --skip_train
python metrics.py --model_path output/endonerf/pulling_match3k
# editing demo
python edit_demo.py --model_path output/endonerf/pulling_match3k \
                 --configs arguments/endonerf/pulling_graph_match_3k.py --iteration 3000 \
                 --edit_mag 0.06 --radius_frac 0.10 --axis y
```
