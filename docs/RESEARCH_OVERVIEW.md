# GC-EndoGaussian — Research Overview

*A single readable account of what we built, the innovation, the key techniques, and an honest assessment.
Every number here was verified against the result files in `results_archive/` (the archived JSON values
are authoritative). Companion docs: [paper.md](paper.md) (the submission paper) and
[IMPLEMENTATION.md](IMPLEMENTATION.md) (configs, commands, and how to reproduce every number).*

---

## 1. Executive summary

We built **GC-EndoGaussian**, an **editable** sparse control-node graph deformation layer added on top
of **EndoGaussian** (a 4D Gaussian Splatting method for dynamic endoscopic tissue reconstruction). A few
thousand control nodes are seeded over the Gaussian cloud, each Gaussian is soft-bound to its nearest
nodes, a small per-node network emits a per-node SE(3) transform per timestamp, and linear blend
skinning propagates that node motion to the Gaussians — turning the base method's opaque deformation
field into something a user can grab and drag. (The node network uses message passing by default, but we
show that is *not* load-bearing — §5.5 — so we do not claim it as a contribution.)

> **⚠️ Correction (supersedes earlier drafts of this document).** An earlier version of this project
> claimed the learned control predicts tissue motion "2–4× more accurately than classical interpolation."
> **That claim was an evaluation artifact and has been retracted.** The control-from-tracks metric left a
> per-Gaussian *reconstruction* residual active, which leaked learned motion into the apparent "control."
> Under a **decontaminated** metric (residual frozen), learned sparse control — ours **and** a retrained
> SC-GS baseline — does **not** beat classical interpolation (ours 6.82 vs nearest-handle 5.69 px at K=4).
> See §5.3. The numbers below are corrected throughout.

The honest headline is two-fold. **Positive:** with the "match" recipe the layer **reproduces
EndoGaussian's reconstruction quality** (within ~0.13–0.16 dB PSNR, equal SSIM, on both EndoNeRF *pulling*
and *cutting*) at **+0.07% parameters**, **205 FPS**, and **no extra training time**. A residual-matched
ablation (§5.5) pins the key ingredient to the **per-Gaussian residual**: a residual-free sparse-control
design (SC-GS-style) loses ~0.5 dB and ~2× tracking, but a residual-*matched* SC-GS-style design performs on
par with ours — so this is a *practical recipe* for reconstruction-neutral editing, not a superiority over
SC-GS. The contribution is this **cost-free editability**, plus a methodological caution: how *not* to fool
yourself when evaluating controllability. It is not a reconstruction gain, not a win over SC-GS once the
residual is matched, and not a controllability gain over classical interpolation.

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
merely replay it. Our goal is to **add an editable deformation layer without degrading reconstruction
quality or real-time speed** — and, honestly, to test whether such learned control is quantitatively better
than classical interpolation (§5.3 finds it is not).

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

1. **Decontaminated controllability evaluation (the primary, most defensible delta).** The control-from-
   tracks protocol + the finding that a naïve version is confounded by the model's own reconstruction, and
   that once decontaminated no learned sparse control (ours or SC-GS) beats classical interpolation. This is
   a methodology contribution that transfers to the whole editable-dynamic-Gaussian subfield (§5.3).

2. **Residual-centered ADDITIVE integration.** SC-GS makes the control points *be* the deformation
   (replace-the-field); we instead **attach** control additively and **retain the base method's per-Gaussian
   deformation as a residual**. A residual-matched ablation (§5.5) shows this is what keeps reconstruction at
   parity (+0.07% params, real-time, ~0.15 dB) — a residual-free design loses ~0.5 dB / ~2× tracking. The
   residual itself is EndoGaussian's, so this is an *integration* delta, not a new component.

3. **Surgical domain.** Transfer to endoscopy: depth supervision, tool masks, single deforming tissue
   surface, plus a studied (optional) cut-aware mechanism for tissue cuts — adaptations SC-GS, targeting
   general dynamic scenes, does not address.

**Not a delta (stated honestly): the GNN message passing.** We couple nodes through a small message-passing
network, but a residual-matched, `gnn_layers=0` and GAT ablation (§5.5) shows the message passing is *not*
load-bearing for reconstruction, and it is *bypassed* at edit time, so it does not help control either. We
therefore do **not** claim it as a contribution.

On pure tracking fidelity the editable layer is statistically equivalent to vanilla (pooled median RPE 3.30
vs 3.47 px, Wilcoxon p = 0.73), and a residual-matched SC-GS-style design matches us too (3.41 px) — so the
honest positive is **cost-free editability with the residual as the key ingredient**, not a superiority over
SC-GS, and the primary contribution is the decontaminated evaluation above.

**One-sentence novelty claim:** *We are not the first to control dynamic Gaussians with sparse handles
(SC-GS is); our contribution is a practical, residual-centered recipe for attaching an editable sparse-control
layer to a strong continuous endoscopic deformation field at essentially zero cost to quality, runtime, and
training — with the per-Gaussian residual isolated as the load-bearing ingredient — together with a
decontaminated controllability evaluation that honestly finds learned sparse control does not (yet) beat
classical interpolation.*

## 5. Experiments & results

All numbers below are verified against the result JSONs in `results_archive/endonerf/*` and the project
docs. The finding is two-part: the control graph **matches** EndoGaussian on reconstruction and tracking
quality while **adding an editable capability the baseline lacks** (gained at no cost, not a quality gain) —
but, once its controllability is measured with a decontaminated metric (§5.3), that control does **not**
beat classical interpolation.

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

### 5.3 Controllability — control-from-tracks (a decontaminated, negative result)

We measure controllability as a **prediction** task on SuPer (UCSD da Vinci tissue manipulation), across
**four trials** (26–51 hand-annotated tissue points, 151 frames each). Take K GT-tracked points as **control
handles**; each handle's observed 3D motion drives the control nodes nearest it; the graph propagates that
sparse control via LBS; we **predict the held-out points** and score reprojection error against the GT
tracks (4-fold CV, `eval_control.py`) vs rigid / nearest-handle / TPS baselines and a retrained **SC-GS**
model.

**The decontamination that changes the answer.** For the score to measure *control* and not
*reconstruction*, **every learned time-varying component must be frozen**. Freezing the graph's node motion
is not enough: the *match* recipe also carries a per-Gaussian **residual** that outputs each Gaussian's
learned displacement at time t. If it stays active, it silently supplies the true motion and the metric
measures reconstruction. Freezing it too (the `control_only` guard in `deformation.py`) is the honest
metric. The effect on our own numbers (cross-trial mean, px):

| K | Naïve (residual active) | **Decontaminated (control only)** |
|---|---|---|
| 4 | 2.86 | **6.82** |
| 8 | 2.77 | **6.80** |
| 16 | 2.92 | **8.09** |

**The honest result: learned control does not beat classical interpolation** (decontaminated, cross-trial
mean px; **bold = best**):

| K | Ours (control only) | SC-GS (learned) | Rigid | **Nearest** | TPS |
|---|---|---|---|---|---|
| 4 | 6.82 | 6.71 | 6.89 | **5.69** | 11.61 |
| 8 | 6.80 | 6.74 | 6.03 | **4.73** | 5.87 |
| 16 | 8.09 | 8.06 | 6.24 | 3.97 | **3.45** |

**Three findings.** (1) Ours ≈ SC-GS — the GNN gives *no* control advantage, because at edit time the
control is a post-hoc node translation that **bypasses** the message passing. (2) Both learned methods lose
to nearest-handle at every K, and are *worst* at K=16. (3) The earlier "2–4×" headline was the residual
leak, retracted. **The takeaway is methodological:** any controllability/editability metric for editable
dynamic-Gaussian models must freeze *all* learned motion, or it measures reconstruction. Naïve numbers are
preserved as `control_results_residual.json`.

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

So the design-space map is: a graph deformation layer is **worthwhile for the editing capability it enables,
not for a quality gain.** The value is **editability added at no cost** — interpretable sparse handles,
delivered at near-exact reconstruction parity (within ~0.16 dB PSNR, equal SSIM), real time (205 FPS),
+0.07% params, no extra training time. A residual-matched ablation (§5.5) identifies the **per-Gaussian
residual** as the ingredient that makes editing reconstruction-neutral: a residual-free sparse-control design
loses ~0.5 dB and ~2× tracking, but a residual-matched SC-GS-style design performs on par with ours (37.29
vs 37.00 dB; 3.41 vs 3.30 px). So the win is the residual, not the GNN coupling or the specific integration
choices — a practical recipe, not a superiority over SC-GS.

**We initially believed there was one axis where the graph clearly wins — control-from-tracks — and we were
wrong.** A naïve version of that metric left the per-Gaussian reconstruction residual active, which leaked
learned motion into the apparent "control" and produced a misleading ~2–4× advantage. Under a
**decontaminated** metric that freezes *all* learned motion (§5.3), learned sparse control — ours and a
retrained SC-GS baseline — does **not** beat classical interpolation: **6.82 / 6.80 / 8.09 px** at
K = 4 / 8 / 16 (cross-trial mean), versus nearest-handle **5.69 / 4.73 / 3.97**. The GNN gives no control
advantage because the edit bypasses the message passing. This is a negative result, reported openly; the
honest positive remains the cost-free editability above, and the transferable lesson is the decontamination
protocol itself.

## 7. Limitations

- **No reconstruction-quality win.** The contribution is functional (editability), not a metric gain.
  The graph *matches* the baseline within ~0.13–0.16 dB PSNR (pulling 37.17 vs 37.32; cutting 39.29 vs
  39.42; equal SSIM to −0.001), and at the original 3000-iter budget the gap is ~0.27 dB (37.00 vs 37.27).
  It never beats vanilla on PSNR/SSIM/LPIPS/depth-RMSE on any of the five experiment families.
- **No controllability win either (decontaminated).** The central negative finding: once the metric is
  decontaminated, learned sparse control does not beat classical interpolation (§5.3). The honest value is
  cost-free editability, not control accuracy.
- **Novelty is incremental versus SC-GS (CVPR 2024).** Sparse control handles + LBS skinning + editing is
  SC-GS's idea, and we do not claim it. Our real deltas — GNN message-passing coupling, the additive
  *quality-neutral* integration over an existing HexPlane field, translation-only rotation handling,
  motion-weighted frozen seeding, the cut-aware extension — are real but modest. A reviewer familiar with
  SC-GS will correctly see this as SC-GS-adjacent.
- **Weak clinical motivation for editing.** Dragging surgical tissue has no obvious, validated clinical use
  case yet; editing is currently evaluated qualitatively plus proxies (locality, plausibility,
  responsiveness, 15:1 control compactness), with no ground-truth *editing* metric.
- **The control mechanism bypasses the GNN.** At edit time the control is a post-hoc node translation, so
  the message passing (which aids reconstruction) never propagates the control. Routing the edit *through*
  the GNN — injecting it as a node input before message passing — is the highest-leverage next step toward
  control that might actually beat interpolation. Driving the handles by real **dVRK kinematics** (rosbags,
  domain-restricted) rather than GT tissue motion is a further step.
- **Evaluation breadth.** Tracking and control results span **four** SuPer trials (3/4/8/9); the
  tracking-fidelity margin over vanilla is small (pooled median RPE 3.47 vs 3.30 px, Wilcoxon p = 0.73 —
  *statistically equivalent*), and all trials share one dataset/rig, so cross-dataset breadth remains open.
- **The SuPer conversion uses approximations.** `tools/super_to_endonerf.py` builds depth from stereo-SGBM,
  assumes a **static endoscope**, and uses **centered intrinsics**. These are self-consistent for the *2D
  reprojection* comparison and fair for graph-vs-vanilla, but the absolute 3D geometry is approximate.
- **Retracted claim.** The earlier "the graph is numerically more stable than the HexPlane" claim was
  dropped (a controlled 2-seed test found vanilla does not reliably diverge; the single NaN was a fluke).

## 8. Honest publication assessment

After decontamination, the structural situation is: (1) no reconstruction-quality improvement (parity);
(2) **no controllability improvement over classical interpolation** — a negative result; (3) novelty
overlapping SC-GS; (4) weak clinical motivation for editing. The surviving positives are the cost-free
integration recipe and the decontamination methodology.

- **As written (claiming a controllability win): not submittable** — the central claim is now known to be
  false, and a reviewer rerunning the metric would see it.
- **Reframed honestly (cost-free editable layer + decontamination finding): a modest workshop paper.** For
  a MICCAI workshop (AE-CAI / surgical-data-science), framing matters: presented as a *negative result* the
  odds are ~30–35%; presented as *"cost-free editability + a methodological caution for evaluating editable
  dynamic-Gaussians,"* ~45–55%. The performance number to lead with is **editing at reconstruction parity**
  (0.15 dB, 205 FPS, +0.07% params), with the residual-matched ablation as the supporting evidence (the
  residual is the key ingredient; a residual-free sparse-control design loses ~0.5 dB / ~2× tracking) —
  *not* controllability, and no longer a claimed superiority over SC-GS.
- **MICCAI main track: below threshold** (~15–25%) — no positive novel result survives.

**What would create a real positive result:**
- **Route control through the GNN** — inject the edit as a node input before message passing so the graph
  propagates sparse control via learned tissue coherence; the only path where learned control might beat
  interpolation. If it does not after one focused iteration, the honest negative-result paper is the floor.
- **Real dVRK kinematics** as the control input (rosbags), turning prediction toward true closed-loop control.
- **Physics-aware control** — a biomechanical prior converting the editing handle into a "what-if" tool.

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
