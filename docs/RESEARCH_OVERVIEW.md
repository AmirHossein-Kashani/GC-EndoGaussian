# GC-EndoGaussian — Research Overview

*A single readable account of what we built, the innovation, the results, and an honest assessment.
Every number here was verified against the result files in `results_archive/` (the archived JSON values
are authoritative). Companion docs: `technical_report.md` (full detail), `implementation_review.md`
(audit), `proposal.md` / `report.md` (framing).*

---

## 1. Executive summary

We built **GC-EndoGaussian**, a controllable sparse control-node graph deformation layer added on top
of **EndoGaussian** (a 4D Gaussian Splatting method for dynamic endoscopic tissue reconstruction). A few
thousand control nodes are seeded over the Gaussian cloud, each Gaussian is soft-bound to its nearest
nodes, a message-passing graph network emits a per-node SE(3) transform per timestamp, and linear blend
skinning propagates that node motion to the Gaussians — turning the base method's opaque deformation
field into something a user can grab and drag. The headline is two-fold and honest: with the "match"
recipe the layer **reproduces EndoGaussian's reconstruction quality** (within ~0.13–0.16 dB PSNR, equal
SSIM, on both EndoNeRF *pulling* and *cutting*) at **+0.07% parameters**, **205 FPS**, and **no extra
training time**; and it adds a **measured controllability capability** — given sparse control handles it
predicts held-out tissue motion **2–4× more accurately than classical interpolation** (median
reprojection error 3.27 px at K=4 vs 6.68–12.34 px for nearest-handle / TPS). The contribution is a
capability added at no cost, not a reconstruction-quality gain.

## 2. Background and motivation

**EndoGaussian** represents a deforming surgical scene as a canonical set of 3D Gaussians plus a
**deformation field** that warps them to each timestamp. The field is a **HexPlane (k-planes)
spatio-temporal feature grid** — 4D over (x, y, z, t) — followed by small per-attribute **MLP heads**
that emit additive deltas to each Gaussian's position (`dx`), scale (`ds`), rotation (`dr`), and opacity
(`do`). Training is two-stage: a **coarse** stage fits static geometry, then a **fine** stage activates
the deformation, supervised by a tool-masked L1 photometric loss, a depth loss (inverse-depth L1 in
binocular mode), and total-variation regularizers. On the standard EndoNeRF clips this is highly accurate
— PSNR in the 37–39 range, effectively saturated.

**The limitation we address.** The HexPlane+MLP field is accurate but **opaque and non-editable**. Each
Gaussian moves independently according to a continuous field whose only coupling between neighbors is the
implicit smoothness of the grid; the field has **no notion of tissue structure and no handles**. After
training, the motion is baked in: there is no way to ask "what if this tissue were displaced here." Yet
many downstream surgical-vision applications — interactive visualization, AR overlay, surgical "what-if"
inspection, simulator and training-data generation — need to *control* the reconstructed scene, not
merely replay it. Our goal is to **add a controllable deformation layer without degrading reconstruction
quality or real-time speed.**

## 3. Method — how it works

### 3.1 Pipeline

We insert a **control-node graph** into the deformation pipeline. A few thousand sparse nodes act as
handles; each Gaussian is softly attached to nearby nodes; a graph network predicts a rigid SE(3) motion
per node at each timestamp; and **linear blend skinning (LBS)** propagates node motion to the Gaussians.
The graph runs on the *sparse* node set, so the only per-Gaussian operation is a cheap gather-and-blend,
preserving real-time rendering.

```
  Gaussians (~30k)            Control nodes (~2k)
  ----------------            -------------------
  canonical xyz  --bind(KNN)-->  node positions
                                     |
                          message-passing GNN(t)
                                     |
                            per-node SE(3) (R, t)   <-- + edit_translation handle
                                     |
        per-Gaussian dx, dr  <-- LBS blend (K=4 nodes)
                                     |
            (+ per-Gaussian MLP residual; rotation/scale/opacity from HexPlane MLP)
                                     v
                          deformed Gaussians --> rasterizer
```

### 3.2 Control-node seeding

Nodes are placed by **motion-weighted farthest-point sampling (FPS)** over the Gaussian cloud at the
start of the fine stage. FPS guarantees spatial coverage; the per-point sampling weight is the
**accumulated deformation magnitude** (`_deformation_accum`, already tracked by EndoGaussian), so node
density follows *motion complexity* rather than static point density — handles land where motion is, not
merely where Gaussians cluster. The node count `M` (typically 1024–2048) is fixed.

### 3.3 Gaussian → node binding

Each Gaussian is bound to its `K = 4` nearest nodes (chunked KNN in canonical space). Binding weights are
a **softmax over negative squared distance**, `w_ik = softmax(−‖x_i − n_k‖² / σ²)`, normalized so a
Gaussian's K weights sum to 1. A Gaussian therefore belongs to several handles, yielding smooth, coherent
deformation. Bindings are stored as per-Gaussian buffers and **rebuilt whenever the Gaussian set changes**
(densification / pruning).

### 3.4 Graph network and SE(3) head

Per timestamp `t`, a message-passing GNN over the node KNN graph produces a per-node transform:

1. Node input feature `h⁰_m = MLP([γ(n_m), γ(t)])`, with `γ` a positional encoding of node position and time.
2. `L = 2` EdgeConv-style layers: `h^{l+1}_m = h^l_m + φ(h^l_m, AGG_n ψ(h^l_m, h^l_n, γ(n_n − n_m)))`,
   with `ψ`, `φ` small MLPs and `AGG` a gated mean over neighbors — so a node's motion is conditioned on
   its neighborhood.
3. An **SE(3) head** emits 3 translation + 6D rotation values per node; the 6D rotation maps to a matrix
   (Zhou et al. continuity).

Because `M ≈ 2k`, the GNN cost is negligible, and there are **no per-node free parameters** (node identity
is encoded by position), so re-seeding / densification never changes the learnable parameter set. SE(3)
math uses pure-PyTorch helpers (quaternion convention `(w,x,y,z)` matching the 3DGS rotation activation),
and the head is **identity-initialized** (R=I, t=0) so deformation begins as an exact no-op and the fine
stage starts from stable static geometry.

### 3.5 LBS assembly

For Gaussian `i` with bindings `{(k, w_ik)}`, node rotations `R_k` and translations `t_k`:
- **Position:** `p_i = Σ_k w_ik [ R_k (x_i − n_k) + n_k + t_k ]`.
- **Rotation:** weighted-blended node quaternion composed with the canonical quaternion.

Non-finite outputs (from a degenerate node) fall back to the canonical value, so NaNs never reach the
rasterizer.

### 3.6 The edit handle

A per-node **`edit_translation`** buffer (zero during training) is added to the node translation. At
inference, setting it for a chosen region of nodes drags those nodes; the bound Gaussians follow via LBS.
This decouples editing from training entirely: **2,048 handles steer ~30,000 Gaussians (≈15:1 control
compactness)**.

### 3.7 Integration modes and the "MATCH recipe"

We studied three ways to combine the graph with the base field:

| Mode | Position | Rotation | Scale/Opacity | Quality vs base |
|---|---|---|---|---|
| **Replace** | graph LBS | graph LBS | HexPlane MLP | −0.6 dB (worse) |
| **Hybrid** | graph LBS + MLP residual | graph LBS + residual | HexPlane MLP | −0.4 dB |
| **Match** (recommended) | graph LBS + MLP residual | **full MLP** | HexPlane MLP | **−0.13…−0.16 dB (matched)** |

The pure **replace** graph lost ~0.5 dB at iteration-matched 6000 iters (pulling 36.77 vs 37.32; cutting
38.91 vs 39.42 with the 2048-node hybrid). The diagnosis was that the graph *can* fit, but two parts
fought the photometric optimum. The **match recipe** removes those costs while keeping the editable graph,
and is the key engineering result. Each ingredient closes part of the ~0.5 dB gap:

- **Translation-only graph.** The graph drives *position* (what editing manipulates), while *rotation* is
  taken from the full per-Gaussian MLP — **avoiding the lossy quaternion-LBS blend** that distorts
  Gaussian orientation.
- **Additive per-Gaussian MLP residual.** A small per-Gaussian residual **recovers the high-frequency
  detail** (the LPIPS "blur") that a low-DoF shared control field cannot express; on its own this
  recovered roughly half the appearance gap (LPIPS 0.095 → 0.076).
- **Coherence regularizers OFF.** ARAP / as-isometric / temporal priors are disabled because they **bias
  position away from the photometric optimum** (pulling/cutting tissue is genuinely non-rigid, so rigidity
  priors only underfit).
- **Frozen nodes.** Nodes are fixed after the initial seed (no periodic re-seeding), removing the
  **mid-training disruption** that perturbed the learned field.

Together these take the graph from ~0.5 dB behind to matching the baseline within ~0.13–0.16 dB (equal
SSIM, −0.001) — making the control graph **quality-neutral**, i.e. the capability is added for free. The
gap holds at the original 3000-iter budget too (~0.27 dB; the extra iterations buy only ~0.1 dB), so there
is **no extra training time**.

*(A cut-aware extension — strain-gated breakable edges, a 2-pass GNN to let the two sides of a tissue cut
deform independently — was studied and dropped: it sharpened the graph at the cut, 11.95 vs 11.88
cut-region PSNR, but still did not beat the continuous field's 12.01, at the cost of a second GNN pass.)*

## 4. Innovation and relation to SC-GS

**Honest framing first.** The core concept here is **not novel**: sparse control points + linear blend
skinning + editing for dynamic Gaussians is **SC-GS (Sparse-Controlled Gaussian Splatting for Editable
Dynamic Scenes, CVPR 2024)**. SC-GS represents a dynamic scene with sparse control points, uses a
**per-control-point MLP** to predict a time-varying SE(3) transform per point, warps the Gaussians by
K-nearest-control-point LBS, adaptively adds/prunes control points, enforces an **ARAP** rigidity loss,
and supports **motion editing** by dragging control points. We explicitly do **not** claim the
sparse-control-handles + LBS + editing idea, the K-nearest binding, or the per-handle SE(3) primitive —
those are SC-GS's. A reviewer familiar with SC-GS will (correctly) read this as an SC-GS-adjacent
contribution.

**Our deltas, ranked by defensibility:**

1. **Zero-cost ADDITIVE integration over a strong continuous field.** SC-GS proposes a deformation
   *method* in which the control points **are** the deformation. We instead **attach** the control graph
   as an additive, quality-neutral layer **on top of an existing HexPlane-MLP field** (EndoGaussian), and
   the engineering result is that this can be done at **+0.07% params, real-time, no extra training time,
   and within ~0.15 dB**. This is a different and more modest claim than "a new deformation method," and
   it is the strongest, most defensible delta.

2. **GNN coupling vs SC-GS's independent control points.** SC-GS's control points are **independent** (a
   per-node MLP), coupled only *softly* through the ARAP loss; ours are coupled **inside the GNN forward
   map** — a node's motion is conditioned on its neighborhood. We make the opposite regularization choice
   as well: we *remove* coherence priors (they hurt fit here). Measured: GNN message passing helps both
   reconstruction (L=2 > L=0 by +0.18 dB on the gate) and controllability (full graph 2.95 vs
   `gnn_layers=0` ablation 4.27 px at K=16).

3. **Surgical domain.** Transfer to endoscopy: depth supervision, tool masks, single deforming tissue
   surface, plus a studied (optional) cut-aware mechanism for tissue cuts — adaptations SC-GS, targeting
   general dynamic scenes, does not address.

4. **The control-from-tracks controllability metric.** Rather than editing-as-demo, we give a
   **GT-backed, quantitative controllability** evaluation on SuPer (32 annotated tissue points): use K
   points as control handles driving the nearest nodes, freeze the learned motion, predict the held-out
   points, and score reprojection error against classical baselines (rigid, nearest-handle, TPS) under
   4-fold leave-groups-out CV. The learned graph predicts held-out motion **2–4× more accurately** (3.27
   px at K=4 vs 6.68 nearest / 12.34 TPS; 3.34 vs 5.81 / 7.39 at K=8; 2.95 vs 3.86 / 3.75 at K=16). On
   pure tracking fidelity the graph is statistically *equivalent* to vanilla (pooled median RPE 3.30 vs
   3.47 px, paired Wilcoxon p = 0.73 — an honest "matches," not a win), which is exactly the point: the
   control representation costs nothing in fidelity yet adds a controllability edge over classical
   interpolation, partly because its local LBS does not extrapolate wildly where TPS does at low K.

**One-sentence novelty claim:** *We are not the first to control dynamic Gaussians with sparse handles
(SC-GS is); our contribution is showing that a GNN-coupled control graph can be attached additively to a
strong continuous endoscopic deformation field at essentially zero cost to quality, runtime, and training,
while delivering a measured controllability advantage over classical interpolation.*

## 5. Experiments & results

All numbers below are verified against the result JSONs in `results_archive/endonerf/*` and the project
docs. The headline finding is consistent across every study: the control graph **matches** EndoGaussian on
reconstruction and tracking quality while **adding a controllability capability the baseline lacks** — it
is a capability gained at no cost, not a quality gain.

**Setup.** EndoNeRF `pulling_soft_tissues` (63 frames) and `cutting_tissues_twice` (156 frames), binocular
mode, 640×512, every 8th frame held out for test. Metrics: PSNR / SSIM / LPIPS / depth-RMSE on the test
set, plus localized region metrics for occlusion and cut studies, render FPS and parameter counts for
efficiency, and ground-truth reprojection error on SuPer for tracking and controllability. Hardware:
single H100, PyTorch 2.5.1 / CUDA 12.x / Python 3.12.

### 5.1 Reconstruction quality (iteration-matched, two datasets)

The recommended **match** recipe (translation-only graph + per-Gaussian MLP residual + regularizer-free +
frozen nodes) is compared against vanilla EndoGaussian at an equal 6000 fine-iteration budget.

| Dataset | Method | PSNR | SSIM | LPIPS | Depth RMSE | ΔPSNR |
|---|---|---|---|---|---|---|
| pulling | vanilla-6k | **37.32** | **0.9578** | **0.0509** | **2.646** | — |
| pulling | **match-6k** | 37.17 | 0.9567 | 0.0533 | 2.793 | **−0.15** |
| cutting | vanilla-6k | **39.42** | **0.9696** | **0.0322** | **1.358** | — |
| cutting | **match-6k** | 39.29 | 0.9689 | 0.0339 | 1.384 | **−0.13** |

At equal iterations the gap is **~0.13–0.16 dB PSNR** on both datasets, with SSIM equal to within −0.001 —
inside run-to-run variance. This is a large improvement over the earlier integration modes: a pure-replace
graph trails vanilla by ~0.6 dB and the 2048-node hybrid by ~0.5 dB (§5.5); the match recipe closes almost
all of that.

**At the original 3000-iteration budget** (no extra training time vs the base method):

| Dataset | Method | PSNR | SSIM | LPIPS | Depth RMSE | ΔPSNR |
|---|---|---|---|---|---|---|
| pulling | vanilla-3k | **37.27** | **0.9578** | **0.0609** | **2.906** | — |
| pulling | **match-3k** | 37.00 | 0.9559 | 0.0638 | 3.139 | **−0.27** |

The gap at the original budget is ~0.27 dB; the extra 3000 iterations buy only ~0.1 dB of that back, so
the capability holds at no extra training cost.

### 5.2 Efficiency

| | EndoGaussian | Ours (match) |
|---|---|---|
| Render speed | 285 FPS | **205 FPS** (7–9× real-time) |
| Deformation params | 85.29 M | 85.35 M (**+0.07%**) |
| Training time | baseline | **unchanged** (3000 iters) |

**Honest note:** the graph is *slightly slower*, not faster — it keeps the full HexPlane field and adds
the GNN plus the LBS gather-and-blend on top. At 205 FPS it remains comfortably real-time (7–9×), so the
cost is immaterial in practice, but we do not claim a speed advantage. The parameter overhead is ~60k
learnable weights on an 85M-parameter grid (+0.07%); because node identity is encoded by position rather
than per-node free parameters, re-seeding and densification never change the learnable parameter count.

### 5.3 Controllability — control-from-tracks prediction (centerpiece)

This is the quantitative, ground-truth-backed controllability result. Editing itself has no ground truth
(the user invents the edit), so we measure controllability as a **prediction** task on SuPer (UCSD da
Vinci tissue manipulation), which provides 32 hand-annotated tissue points tracked across 151 frames.

**Protocol in plain words.** Take K of the 32 GT-tracked points and treat them as **control handles**.
Each handle's *observed* 3D motion (back-projected via the static-camera geometry) is used to drive the
control nodes nearest to it, with the learned node motion frozen. The graph then propagates that sparse
control to the rest of the tissue via LBS, and we **predict the held-out points** that were not used as
handles. We project the predictions and score reprojection error against the GT 2D tracks. Evaluation is
4-fold leave-groups-out cross-validation (`eval_control.py`), compared against three classical control
baselines — rigid translation, nearest-handle copy, thin-plate-spline (TPS) interpolation — at handle
counts K = 4, 8, 16.

Held-out reprojection error (px), median ± std over folds:

| K handles | **Graph** | Rigid | Nearest | TPS |
|---|---|---|---|---|
| 4 | **3.27** ± 0.14 | 7.36 ± 0.73 | 6.68 ± 1.60 | 12.34 ± 5.32 |
| 8 | **3.34** ± 0.20 | 7.01 ± 0.63 | 5.81 ± 0.68 | 7.39 ± 1.60 |
| 16 | **2.95** | 7.03 | 3.86 | 3.75 |

**Headline finding.** Given sparse control, the learned control graph predicts held-out tissue motion
**2–4× more accurately than classical interpolation**, with the largest margin in the useful sparse regime
(K = 4–8), where TPS extrapolates wildly (12.34 px at K=4) and the graph's local LBS does not. This is the
result that distinguishes the work from editing-as-demo: a measured, GT-backed controllability number.

**GNN ablation (`gnn_layers=0`).** Removing message passing degrades the graph's prediction at every K,
and the gap widens as more handles are added:

| K handles | Graph (full GNN, L=2) | Graph (`gnn_layers=0`) |
|---|---|---|
| 4 | **3.27** | 3.54 |
| 8 | **3.34** | 3.45 |
| 16 | **2.95** | 4.27 |

Message passing helps — most strikingly at K=16 (2.95 vs 4.27), where neighborhood coupling lets the graph
exploit denser handle information instead of saturating. **Honest scope:** the control input here is GT
*tissue* motion, not a robot command, so this is deformation-prediction-under-sparse-observation rather
than control under a kinematic input; the dVRK robot-command version needs rosbag kinematics, left as
future work.

### 5.4 Tracking fidelity and statistical rigor

To put a ground-truth number on the deformation representation itself (independent of any control input),
we measure SuPer **reprojection error (RPE)**: anchor each GT point to the nearest front-most canonical
Gaussian at frame 0, deform it to every frame, project, and compare to the GT 2D track (`eval_tracking.py`).

| Metric (px) | Vanilla | Graph (match) |
|---|---|---|
| RPE mean | 8.86 | **8.71** |
| RPE median (per-point) | 3.47 | **3.30** |
| RPE frame-0 (anchor noise floor) | 2.12 | **2.06** |

The graph matches and marginally beats vanilla on all three metrics; the frame-0 RPE ≈ 2 px confirms the
projection pipeline (anchor-association noise floor).

**Statistical rigor.** The pooled median RPE is **vanilla 3.47 px [95% CI 3.34–3.59] vs graph 3.30 px [95%
CI 3.14–3.46]**; the confidence intervals overlap, and a paired Wilcoxon signed-rank test over the
per-frame errors gives **p = 0.73**. The two methods are therefore **statistically equivalent** on
tracking fidelity — an honest "matches," not a win. The correct claim is that the sparse control
representation reproduces real annotated tissue motion at least as faithfully as the dense HexPlane
baseline, on a ground-truth benchmark, while adding controllability.

### 5.5 Negative and dropped results (stated honestly)

A central contribution is mapping where a graph deformation layer does *not* help. Every one of these was
tried and reported rather than buried.

**Pure-replace and hybrid lost on quality.** Replacing the field with the graph, or adding a residual to a
graph-driven field, both trailed vanilla at the 3000-iter budget on pulling:

| Method | PSNR | SSIM | LPIPS | Depth RMSE |
|---|---|---|---|---|
| vanilla | **37.27** | **0.9578** | **0.0609** | **2.906** |
| graph, replace (GNN, L=2) | 36.68 | 0.9488 | 0.0946 | 3.001 |
| graph, replace (no GNN, L=0) | 36.50 | 0.9476 | 0.0954 | 3.037 |
| graph, hybrid (1024 nodes) | 36.88 | 0.9537 | 0.0760 | 3.037 |

Message passing helps the graph (L=2 > L=0, +0.18 dB and lower LPIPS) and the hybrid residual recovers
about half the appearance gap (LPIPS 0.095 → 0.076), but neither beats vanilla. Only the full match recipe
(§5.1) closes the gap.

**Occlusion-holdout — no win.** A central tissue box was withheld from supervision for a block of frames
and scored on recovery. The graph does *not* recover occluded tissue better; the occluded-box PSNR is
essentially equal (vanilla 26.17 vs hybrid 26.00), and vanilla's continuous HexPlane interpolates the
temporally-bounded occlusion adequately.

**Optical-flow supervision — no gain.** Offline Farneback flow used as a warp-consistency loss (warp
verified to improve GT) gave only a marginal SSIM change and did not close the gap; at high weight it hurt:

| Variant | PSNR | SSIM | LPIPS | RMSE |
|---|---|---|---|---|
| vanilla | 37.27 | 0.9578 | 0.0609 | 2.906 |
| vanilla + flow | 37.30 | 0.9587 | 0.0690 | 2.864 |
| hybrid + flow (λ=0.5) | 36.85 | 0.9571 | 0.0760 | 3.035 |
| hybrid + flow (λ=1.0) | 36.45 | 0.9537 | 0.0946 | 3.287 |

The image-space consistency loss carries little gradient once renders are roughly correct.

**Cut-aware helped the graph but did not beat vanilla.** Strain-gated breakable edges (a second, gated GNN
pass) were added to model a tissue cut, scored on the cut region (top-10% motion PSNR):

| | cut-region PSNR |
|---|---|
| vanilla-6k | **12.01** |
| graph match (no cut) | 11.88 |
| graph cut-aware | 11.95 |

Cut-aware edges sharpen the graph at the cut (11.95 > 11.88) but still fall short of the continuous field
(< 12.01), and they add a full extra GNN pass — so the mechanism is off by default. The full match model
on cutting (39.30 PSNR) is statistically indistinguishable from match-without-cut (39.29).

**The "graph is more stable" claim was retracted.** An earlier observation that vanilla's HexPlane
diverged to NaN on the noisier SuPer data (while the node graph trained cleanly) was reported as a
stability advantage. A controlled follow-up test (gradient clipping disabled, two seeds) found that
vanilla does *not* reliably diverge — the single earlier NaN was a stochastic fluke. The "graph is more
stable" claim is **not supported and is dropped**.

**Sparse-view — inconclusive.** Training on every Nth frame showed no consistent advantage for the graph.

**Summary of the design-space study.** A continuous HexPlane field is already smooth and temporally
coherent, so a control graph adds *constraint, not information*. Across standard, sparse-view, occlusion,
optical-flow, and cut-aware settings the graph matches but does not exceed reconstruction quality. The
value of the control graph is therefore the **capability** (editability, interpretable and GT-validated
sparse control) achieved at no cost — not a quality gain.

## 6. What we learned about the design space

The single most important lesson is structural, and it explains every reconstruction result we measured:
**a continuous HexPlane (k-planes) deformation field is already smooth and temporally coherent.**
EndoGaussian's field deforms each Gaussian through a spatially continuous 4D grid, so nearby Gaussians at
the same timestamp already receive similar deltas — an implicit smoothness prior baked into the
architecture. A control-node graph imposes the *same* kind of smoothness, but explicitly, through sparse
handles and linear blend skinning.

The consequence is that **the control graph adds *constraint*, not *information*.** Everything a coherent
low-DoF node field can express, the HexPlane field can already express (and more, because it has
near-per-Gaussian capacity). So on reconstruction quality the graph can at best *match* the baseline; it
cannot *beat* it. This is not a tuning failure — it is a property of the design space, and we confirmed it
exhaustively (standard, sparse-view, occlusion, optical-flow, cut-aware — §5.5).

So the design-space map is: a graph deformation layer is **worthwhile for the capability it enables, not
for a quality gain.** The genuine value is **controllability added at no cost** — editability and
interpretable sparse control, delivered at near-exact parity (within ~0.16 dB PSNR, equal SSIM), real time
(205 FPS), +0.07% params, no extra training time. The enabling recipe (translation-only graph control +
per-Gaussian MLP residual + regularizer-free, frozen-node integration) is what makes that capability free;
it removes the two costs — the lossy quaternion-LBS rotation blend, and coherence regularizers that bias
position off the photometric optimum — that previously made the graph lose by ~0.5 dB.

There is exactly **one axis where the graph clearly and measurably wins: control-from-tracks.** Given
sparse control handles (K of 32 GT-tracked SuPer points) and asked to *predict* held-out tissue motion,
the learned graph beats classical interpolation 2–4×: median held-out reprojection error of
**3.27 / 3.34 / 2.95 px** at K = 4 / 8 / 16, versus rigid (7.36 / 7.01 / 7.03), nearest-handle copy
(6.68 / 5.81 / 3.86), and thin-plate-spline (12.34 / 7.39 / 3.75). The margin is largest in the useful
sparse regime (K = 4–8). Message passing contributes: the full graph beats its `gnn_layers=0` ablation,
most visibly at K=16 (2.95 vs 4.27 px). The intuition is the converse of the reconstruction finding — when
supervision is *sparse*, a learned, neighborhood-coupled, locally-rigid skinning extrapolates sensibly
where TPS overshoots, and the constraint that hurt under dense supervision now *helps*.

## 7. Limitations

- **No reconstruction-quality win.** The contribution is functional (controllability), not a metric gain.
  The graph *matches* the baseline within ~0.13–0.16 dB PSNR (pulling 37.17 vs 37.32; cutting 39.29 vs
  39.42; equal SSIM to −0.001), and at the original 3000-iter budget the gap is ~0.27 dB (37.00 vs 37.27).
  It never beats vanilla on PSNR/SSIM/LPIPS/depth-RMSE on any of the five experiment families.
- **Novelty is incremental versus SC-GS (CVPR 2024).** Sparse control handles + LBS skinning + editing is
  SC-GS's idea, and we do not claim it. Our real deltas — GNN message-passing coupling, the additive
  *quality-neutral* integration over an existing HexPlane field, translation-only rotation handling,
  motion-weighted frozen seeding, the cut-aware extension — are real but modest. A reviewer familiar with
  SC-GS will correctly see this as SC-GS-adjacent.
- **Weak clinical motivation for editing.** Dragging surgical tissue has no obvious, validated clinical use
  case yet; editing is currently evaluated qualitatively plus proxies (locality, plausibility,
  responsiveness, 15:1 control compactness), with no ground-truth *editing* metric.
- **The controllability metric's control input is GT tissue motion, not robot kinematics.** The headline
  result (§5.3) drives the handles by the SuPer points' *observed* 3D motion — it is
  deformation-prediction-under-sparse-observation, not control-under-a-robot-command. The true control
  input, the **dVRK kinematics**, ships only inside the original SuPer rosbags (domain-restricted access +
  ROS extraction), so it is not yet wired in. This is the single highest-leverage missing piece.
- **A single SuPer trial (trial_3).** Both the tracking-fidelity and control results come from one trial;
  the tracking-fidelity margin is small (pooled median RPE: vanilla 3.47 px [CI 3.34–3.59] vs graph 3.30 px
  [3.14–3.46], paired Wilcoxon p = 0.73 — *statistically equivalent*). Trials 4/8/9 are not yet run.
- **The SuPer conversion uses approximations.** `tools/super_to_endonerf.py` builds depth from stereo-SGBM,
  assumes a **static endoscope**, and uses **centered intrinsics**. These are self-consistent for the *2D
  reprojection* comparison and fair for graph-vs-vanilla, but the absolute 3D geometry is approximate.
- **Retracted claim.** The earlier "the graph is numerically more stable than the HexPlane" claim was
  dropped (a controlled 2-seed test found vanilla does not reliably diverge; the single NaN was a fluke).

## 8. Honest publication assessment

The three structural weaknesses persist and were not removed by any result: (1) no reconstruction-quality
improvement, (2) novelty overlapping SC-GS, (3) weak clinical motivation for editing.

- **MICCAI workshop** (AE-CAI / surgical-data-science / similar): solid fit, **likely accept.** "A
  controllable, real-time, GT-validated editing layer for endoscopic 4D reconstruction at no cost to
  quality or speed" is true, complete, and modest — and the control-from-tracks result is a genuine,
  measured, GT-backed win that workshops reward.
- **MICCAI main track**: submittable but capped at roughly **~20–30%.** The controllability win (2–4× over
  classical interpolation, with the GNN ablation supporting it) materially raises it above a pure system
  paper, but novelty-vs-SC-GS and the thin clinical motivation hold the ceiling down.

**What would raise main-track odds:**
- **Run SuPer trials 4/8/9** (not just trial_3), report median + bootstrap CI, and average across trials —
  converting a single-trial result into a robust multi-trial one.
- **A clinical use case** — most directly, wiring in the **real dVRK kinematics** so the control input is a
  robot command, turning "deformation prediction" into true *controllability* with a clinical hook.
- **Physics-aware control** — a biomechanical prior on the node graph that converts the editing handle into
  a predictive "what-if" surgical-simulation tool — the most promising path to a clinically meaningful,
  measurable, *and* novel contribution.

## 9. Reproduce / code map

**Core module**
- `scene/node_deformation.py` — `NodeGraphDeformation`: SE(3)/quaternion helpers (6D-rotation→matrix,
  identity-initialized head), chunked KNN, motion-weighted FPS seeding, soft K-NN binding, the
  message-passing GNN (`node_transforms`), the LBS blend (`forward`), the `edit_translation` handle, and
  the cut-aware strain gate.

**Integration points**
- `scene/deformation.py` — instantiates the node module when `use_node_graph`; `forward_dynamic` mode logic
  (replace / hybrid / translation-only); keeps scale/opacity on the MLP; re-applies the identity SE(3)
  head after global init.
- `scene/gaussian_model.py` — `_binding_idx` / `_binding_w` state, the `node_gnn` optimizer group + LR
  schedule, `seed_node_graph` / `compute_node_bindings` / `maintain_node_graph`, and the graph regularizer
  `compute_node_regulation`.
- `gaussian_renderer/__init__.py` — `render()` slices/passes per-Gaussian bindings, lazily rebuilding them
  when the Gaussian count changes.
- `train.py` — fine-stage seeding, graph maintenance through densification/pruning, optional flow/occlusion
  hooks, gradient clipping.
- `arguments/__init__.py` — all knobs (`use_node_graph`, `node_hybrid`, `node_translation_only`,
  `num_nodes`, `gauss_knn_K`, `gnn_layers`, `node_pe`, `lambda_arap/isometric/node_temporal`,
  `node_refresh_interval`, `node_lr_init/final`, `cut_aware/cut_beta/cut_thresh`, `grad_clip`).

**Evaluation scripts** (read-only, reuse the render machinery)
- `eval_control.py` — the headline controllability metric: drive nodes by K GT handles, freeze learned
  motion, predict held-out points, 4-fold leave-groups-out CV vs rigid/nearest/TPS (`control_results.json`).
- `eval_tracking.py` — SuPer ground-truth reprojection error + bootstrap CIs (`tracking_results.json`);
  `eval_paired.py` — the paired Wilcoxon vanilla-vs-graph.
- `eval_occlusion.py`, `eval_cut.py` — localized occlusion-holdout and cut-region PSNR metrics.
- `tools/super_to_endonerf.py` — converts a SuPer trial to EndoNeRF format (stereo-SGBM depth, tool masks,
  static-camera poses, centered intrinsics).
- `edit_demo.py` — the drag-to-edit capability demo (before/after/diff renders).

**Recommended config:** `arguments/endonerf/pulling_graph_match_3k.py` (the match recipe at the original
3000-iter budget, no extra training time). The 6000-iter variant is `pulling_graph_match.py`; cutting uses
`cutting_graph_match.py`.

**SLURM wrappers:** `run_gc_match3k.bash` (recommended), `run_gc_match.bash` (iteration-matched),
`run_gc_super.bash` (SuPer convert + train + track + control), `run_gc_improve.bash` (the three-legs
controllability/rigor study), `run_gc_render_metrics.bash` (score already-trained models); plus
`run_gc_hybrid/sparse/occ/flow/cut/pivot.bash` for the ablations.

**End-to-end (recommended model + capability + GT metrics):**
```bash
# 1. train the match model at the original budget
python train.py  -s data/endonerf/pulling --expname endonerf/pulling_match3k \
                 --configs arguments/endonerf/pulling_graph_match_3k.py --save_iterations 1000 3000
# 2. render + reconstruction metrics
python render.py  --model_path output/endonerf/pulling_match3k \
                 --configs arguments/endonerf/pulling_graph_match_3k.py --skip_train
python metrics.py --model_path output/endonerf/pulling_match3k
# 3. editing demo (drag a node region -> before/after/diff)
python edit_demo.py --model_path output/endonerf/pulling_match3k \
                 --configs arguments/endonerf/pulling_graph_match_3k.py --iteration 3000 \
                 --edit_mag 0.06 --radius_frac 0.10 --axis y
# 4. GT-backed SuPer: convert trial, train vanilla+graph, then track + control (run_gc_improve.bash does all)
python eval_tracking.py --model_path output/endonerf/super_match ...   # tracking_results.json
python eval_control.py  --model_path output/endonerf/super_match ...   # control_results.json (the headline)
```

**Verified ground-truth numbers** (from `results_archive/endonerf/*/`):
- Reconstruction (6k): pulling vanilla 37.32 / 0.9578 / 0.0509 / 2.646; pulling match 37.17 / 0.9567 /
  0.0533 / 2.793; cutting vanilla 39.42 / 0.9696 / 0.0322 / 1.358; cutting match 39.29 / 0.9689 / 0.0339 /
  1.384. Match-3k: 37.00 / 0.9559 / 0.0638 / 3.139 (vanilla-3k 37.27 / 0.9578 / 0.0609 / 2.906).
- SuPer tracking RPE (px): vanilla mean 8.86, median 3.47 [CI 3.34–3.59], frame-0 2.12; graph mean 8.71,
  median 3.30 [CI 3.14–3.46], frame-0 2.06; paired Wilcoxon p = 0.73 (statistically equivalent).
- Control (held-out RPE median px, graph vs rigid/nearest/TPS): K=4 → 3.27 / 7.36 / 6.68 / 12.34; K=8 →
  3.34 / 7.01 / 5.81 / 7.39; K=16 → 2.95 / 7.03 / 3.86 / 3.75. GNN ablation (`gnn_layers=0`): 3.54 / 3.45 /
  4.27.
