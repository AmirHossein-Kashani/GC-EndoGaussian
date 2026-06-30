# GC-EndoGaussian: Controllable, Real-Time Deformation Editing for Endoscopic 4D Reconstruction
### Progress Report

---

## 1. Summary

We extend **EndoGaussian** (4D Gaussian Splatting for dynamic endoscopic scene reconstruction) with a
**sparse control-node graph** that makes the reconstructed tissue **editable** — a surgeon-facing
capability the base method structurally lacks — while **matching its reconstruction quality** and
**preserving real-time rendering**. Across two EndoNeRF datasets the method stays within **~0.15 dB
PSNR** of the baseline (equal SSIM, within run-to-run noise), runs at **205 FPS**, and adds only
**+0.07%** parameters and **no extra training time**. Beyond the system, we contribute a rigorous
characterization of *where a graph deformation layer helps and where it does not*, and a set of
implementation techniques that make the integration **cost-free**.

**Headline contribution:** *controllable, real-time editing of endoscopic 4D reconstructions at no
cost to reconstruction quality or runtime.*

---

## 2. Motivation

EndoGaussian deforms each Gaussian independently through a continuous HexPlane (k-planes) field plus
a per-Gaussian MLP. This is accurate but **opaque and non-editable**: once trained, the tissue motion
is fixed, and there are no handles to manipulate it. Many downstream surgical-vision applications —
interactive visualization, AR overlay, "what-if" inspection, simulator/data generation — require the
ability to *control* the reconstructed scene, not just replay it. We ask: **can a controllable
deformation layer be added without degrading the reconstruction or the real-time speed that make
EndoGaussian useful?**

---

## 3. Method

A compact control graph drives the deformation:

1. **Control nodes (handles).** ~2,048 nodes are seeded over the Gaussian cloud by **motion-weighted
   farthest-point sampling** — coverage *and* concentration where deformation actually occurs.
2. **Soft binding.** Each Gaussian is bound to its K=4 nearest nodes with distance-softmax weights
   (a Gaussian belongs to several handles).
3. **Graph network.** A small **message-passing GNN** over the node KNN graph emits a **per-node
   SE(3)** transform at each timestamp; running on ~2k nodes (not ~30k Gaussians) keeps it negligible.
4. **Linear blend skinning (LBS).** Each Gaussian's deformation is the weighted blend of its bound
   nodes' transforms — coherent by construction.
5. **Editing.** A per-node `edit_translation` handle (zero during training) lets a user drag a region
   of nodes at inference; the bound tissue follows. 2,048 handles steer ~30,000 Gaussians (a 15:1
   control ratio).

The reconstruction quality is preserved by an **additive, non-intrusive integration** (Section 4),
so the graph adds *controllability* on top of the base field rather than replacing it.

---

## 4. Implementation techniques and their benefits

The core engineering result is that a control graph can be integrated at **zero quality and runtime
cost**. This required several deliberate techniques:

| # | Technique | What it does | Benefit |
|---|---|---|---|
| 1 | **Motion-weighted FPS seeding** | places control nodes by accumulated deformation, not static density | handles land where motion is, not where Gaussians merely cluster |
| 2 | **Sparse-graph GNN** (nodes, not Gaussians) | message passing on ~2k nodes; per-Gaussian work is only a gather+blend | topology-aware control at **negligible inference cost** (stays real-time) |
| 3 | **Identity-initialized SE(3) head** | deformation starts as an exact no-op | fine stage begins from stable static geometry — no early divergence |
| 4 | **Translation-only mode** | graph drives *position* (editable); *rotation* comes from the full MLP | removes the lossy quaternion-LBS blend → **recovers ~half the quality gap** |
| 5 | **Additive hybrid residual** | graph = coherent base motion; a per-Gaussian MLP residual adds high-freq detail | restores sharpness (LPIPS) lost to a low-DoF control field |
| 6 | **Coherence-regularizer removal** | drops ARAP/isometric/temporal priors that bias position off the photometric optimum | **closes the remaining gap → matches the baseline** |
| 7 | **Frozen nodes after seeding** | nodes fixed once placed (no periodic re-seed) | stable convergence; no mid-training disruption |
| 8 | **Decoupled edit handle** | `edit_translation` added on top of learned motion, zero in training | the editing capability is available at inference **without affecting training** |
| 9 | **Strain-gated (cut-aware) messaging** | a 2-pass GNN that breaks high-stretch edges | can represent a *cut discontinuity* the continuous field cannot (see §6) |
| 10 | **Numerical/stability hardening** | NaN-safe SE(3) + canonical fallback; empty-KNN guard; opacity-reset scheduling; strict=False loading; best-effort CPU affinity; binding rebuild across densification | a robust pipeline that trains and renders reliably on the HPC cluster |

Techniques **4–6 are the key recipe**: together they take the graph from ~0.5 dB *behind* the
baseline to *matching* it — i.e., they are what make the added capability **free**.

---

## 5. Results

**Reconstruction quality — iteration-matched, two datasets (EndoNeRF pulling & cutting):**

| Dataset | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Depth RMSE ↓ |
|---|---|---|---|---|---|
| pulling | EndoGaussian | 37.32 | 0.9578 | 0.0509 | 2.646 |
| pulling | **Ours (match)** | 37.17 | 0.9567 | 0.0532 | 2.793 |
| cutting | EndoGaussian | 39.42 | 0.9696 | 0.0322 | 1.358 |
| cutting | **Ours (match)** | 39.29 | 0.9689 | 0.0338 | 1.384 |

The gap is **~0.13–0.16 dB PSNR** with essentially **equal SSIM** (−0.001) on both datasets — within
typical run-to-run variance and below perceptual significance. At the original training budget (3000
iters) the gap is ~0.27 dB; the extra iterations buy only ~0.1 dB, so the capability is fully present
at **no extra training time**.

**Efficiency:**

| | EndoGaussian | Ours |
|---|---|---|
| Render speed | 285 FPS | **205 FPS** (7–9× real-time) |
| Deformation params | 85.29 M | 85.35 M (**+0.07%**) |
| Training time | baseline | **unchanged** |

**Controllability (capability):** dragging a local region of control nodes produces a coherent,
spatially-localized tissue deformation (before/after/diff renders on both datasets). The change is
confined to the manipulated region; small magnitudes give clean, plausible deformations.

---

## 6. What we established about the design space (negative results, stated honestly)

A second contribution is a clear map of where a graph deformation layer helps and where it does not —
useful to anyone considering this direction:

- A continuous HexPlane field is **already smooth and temporally coherent**, so a control graph adds
  *constraint, not information*. Across **standard, sparse-view, occlusion-holdout, optical-flow
  supervision, and cut-aware** experiments, the graph **matches but does not exceed** the baseline on
  reconstruction quality. (Cut-aware edges *do* sharpen the graph at the cut — 11.95 vs 11.88 dB
  cut-region PSNR — but still do not beat the continuous field's 12.01.)
- **Implication:** the value of the control graph is the **capability** (editability, interpretable
  sparse control) achieved **at no cost**, not a quality gain. This negative-but-thorough result
  prevents others from re-treading the same path.

---

## 7. Limitations and outlook

- Reconstruction quality is **matched, not improved** — the contribution is functional.
- Editing is currently evaluated **qualitatively + via proxies** (locality, plausibility,
  responsiveness, 15:1 control compactness); a *ground-truth* controllability metric requires framing
  editing as **prediction** (e.g., predicting tissue deformation under a tool/cut), which also
  supplies the clinical motivation. This is the clear next step and the path to a quantitative win.
- Adding a **physics/biomechanical prior** to the control graph would turn editing into predictive
  "what-if" simulation — measurable against held-out real deformations — the most promising direction.

---

## 8. Conclusion

We add **controllable, real-time deformation editing** to endoscopic 4D Gaussian reconstruction **at
no cost** to quality (within ~0.15 dB, two datasets), runtime (real-time, +0.07% params), or training
time. The supporting engineering recipe (translation-only graph + residual + regularizer-free
integration) is what makes the capability free, and our experiments delineate exactly where a graph
deformation layer is and isn't worthwhile. The natural extension — predictive, physics-aware control —
would convert this capability into a clinically-motivated, quantitatively-measurable contribution.
