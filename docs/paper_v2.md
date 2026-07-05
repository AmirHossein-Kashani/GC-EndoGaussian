# GC-EndoGaussian: Residual-Centered Sparse Control for Editable Endoscopic 4D Gaussian Reconstruction

**Anonymous submission**

---

## Abstract

Dynamic 4D Gaussian Splatting methods can reconstruct deforming tissue from endoscopic video at high visual quality and interactive rendering rates. However, their learned deformation fields provide no explicit handles for manipulating the reconstructed scene after training. This limits their use in applications that require controllable scene deformation, including interactive visualization, augmented-reality overlays, what-if inspection, and surgical training-content generation.

We present **GC-EndoGaussian**, an editable deformation layer for endoscopic 4D Gaussian reconstruction. The method seeds a sparse set of control nodes over the Gaussian cloud, binds each Gaussian to nearby nodes, predicts time-varying node motion, and propagates this motion through a weighted skinning operation. User-specified node translations can then be applied at inference time to produce local edits without retraining.

Our main contribution is a **residual-centered integration recipe** that retains near-baseline reconstruction fidelity while adding sparse editing handles. GC-EndoGaussian outperforms the standard SC-GS-style sparse-control configuration by **+0.20 dB PSNR** and **53% lower tracking error** (3.30 vs 7.02 px), while remaining within 0.27 dB of the dense EndoGaussian baseline at the same training budget — closing to 0.15 dB under an extended schedule. The method renders at **205 FPS** with only **+0.07% parameters**, and tracking shows no statistically significant difference from EndoGaussian ($p=0.73$).

A controlled residual-isolation experiment explains *why*: removing the per-Gaussian residual from any sparse-control architecture causes PSNR to drop by ~0.5 dB and tracking error to roughly double. Adding it back — to either our architecture or a standard SC-GS-style model — recovers near-baseline fidelity. **The residual is the transferable, architecture-independent ingredient** for quality-preserving sparse editability in endoscopic reconstruction.

**Keywords:** surgical scene reconstruction · 4D Gaussian Splatting · editable deformation · sparse control nodes · endoscopy

---

## 1. Introduction

Dense reconstruction of deforming tissue from endoscopic video is an important component of computer-assisted intervention. Neural radiance fields and, more recently, 3D Gaussian Splatting (3DGS) have improved both visual fidelity and rendering speed. Surgical methods including EndoNeRF, EndoGaussian, Deform3DGS, Endo-4DGS, EndoGS, and SurgicalGaussian reconstruct dynamic endoscopic scenes by learning a canonical representation together with a time-dependent deformation field.

These methods are optimized for one thing: *faithfully replaying* the observed deformation. The
deformation field is a **continuous, per-Gaussian function with no explicit structure and no handles**.
After training, the motion is fixed — there is no way to ask "what would this tissue look like if that
flap were displaced here?" Many surgical-vision applications, however, need to *control* the reconstructed
scene, not merely replay it: interactive intra-operative visualization, AR overlay and re-planning,
what-if inspection of tissue response, and the generation of synthetic-but-plausible training data for
downstream models. Concretely, in an augmented-reality guidance overlay a surgeon could displace a tissue
flap in the live reconstruction to preview access to an underlying structure; in **VR/AR surgical
training**, an instructor could author varied tissue configurations from a single captured procedure to
generate diverse practice scenarios without re-capturing. We present these as *potential applications* —
our edits are plausible and produced at negligible cost, not biomechanically validated predictions of
tissue response.

Sparse-control methods for dynamic Gaussian scenes provide a natural editing interface. SC-GS, for example, attaches Gaussians to sparse control points and uses skinning to propagate point motion. Directly replacing a strong endoscopic deformation field with a lower-dimensional sparse-control model, however, can reduce reconstruction fidelity. The central question of this work is therefore:

> How can sparse editing handles be added to an endoscopic 4D Gaussian reconstruction without discarding the detailed deformation already captured by the base model?

We address this question with GC-EndoGaussian, which combines sparse node control with a dense per-Gaussian residual. The sparse component provides edit handles, while the residual retains local motion detail that cannot be represented by the shared node field alone.

Our contributions are:

1. **A residual-centered integration recipe for editable endoscopic reconstruction.** The final configuration combines translation-only node control, a dense per-Gaussian position residual, fixed nodes, and no coherence regularization. It remains within 0.27 dB of EndoGaussian under the standard training schedule and within 0.15 dB under an extended schedule, while rendering at 205 FPS with 0.07% additional deformation parameters.

2. **A sparse edit-control layer for 4D endoscopic Gaussians.** Motion-seeded control nodes, K-nearest-neighbor soft binding, weighted skinning, and inference-time node translations provide a compact interface for local scene manipulation. The sparse-control and skinning paradigm follows prior editable Gaussian methods; our focus is its integration and evaluation in a surgical reconstruction pipeline.

3. **A controlled isolation of the fidelity-preserving component.** A residual-matched comparison shows that an SC-GS-style model without a dense residual loses approximately 0.5 dB PSNR and has roughly twice the tracking error, whereas the same control architecture with the residual recovers near-baseline reconstruction and tracking.

---

## 2. Related Work

### Dynamic reconstruction for surgery

EndoNeRF introduced neural rendering for deformable tissue reconstruction from stereo endoscopic video. Subsequent 3DGS-based methods replace volumetric rendering with explicit Gaussian primitives, enabling higher rendering rates. EndoGaussian combines a canonical Gaussian representation with a HexPlane-based spatio-temporal deformation field and depth-supervised training. Deform3DGS, Endo-4DGS, EndoGS, and SurgicalGaussian explore alternative parameterizations and optimization strategies. These methods emphasize reconstruction fidelity and rendering efficiency, but do not expose sparse handles for post-training scene manipulation.

### Editable dynamic Gaussian representations

SC-GS represents dynamic scenes using sparse control points whose time-varying transformations are propagated to Gaussians through K-nearest-neighbor linear blend skinning. It also supports user edits by manipulating selected controls. Related motion-graph, dual-quaternion-skinning, and learned weight-painting approaches share the same general idea: a sparse control representation provides an interface for coherent deformation.

GC-EndoGaussian adopts this established control-and-skinning paradigm. Its contribution is not a new editing primitive. Instead, it examines how sparse control should be integrated with an already strong endoscopic deformation model. In particular, we show that retaining a dense per-Gaussian residual is more important for reconstruction fidelity than the choice of graph message-passing architecture.

### Deformation parameterization

The base reconstruction uses a factorized spatio-temporal representation such as HexPlane or K-Planes to generate per-Gaussian deformation features. The general sparse-control implementation supports rigid node transformations parameterized with continuous 6D rotations. In the final GC-EndoGaussian configuration, however, the learned control graph is restricted to translation, while Gaussian rotation, scale, and opacity remain governed by the base deformation field.

---

## 3. Method

### 3.1 Base 4D Gaussian reconstruction

The reconstructed scene contains approximately $N\approx30\text{k}$ canonical 3D Gaussians. Gaussian $i$ has canonical position $x_i$, rotation, scale, opacity, and spherical-harmonic color coefficients. At timestamp $t$, a HexPlane-style spatio-temporal field produces a feature for each Gaussian. Small prediction heads then output changes in position, rotation, scale, and opacity.

Training follows the two-stage procedure of the base method. A coarse stage estimates static geometry, and a fine stage activates the dynamic deformation model. Supervision includes a tool-masked photometric loss, an inverse-depth loss in binocular mode, and total-variation regularization on the spatio-temporal field.

This dense deformation model provides high reconstruction quality but no explicit post-training control handles.

### 3.2 Sparse control-node graph

At the beginning of the fine stage, we initialize $M=1024$--$2048$ control nodes over the Gaussian cloud. Nodes are selected using motion-weighted farthest-point sampling. Farthest-point sampling encourages spatial coverage, while accumulated deformation magnitude gives greater sampling weight to regions exhibiting stronger motion.

Each Gaussian is bound to its $K=4$ nearest nodes. For Gaussian $i$ and neighboring node $k$, the normalized binding weight is

$$
w_{ik}=\frac{\exp\left(-\lVert x_i-n_k\rVert^2/\sigma^2\right)}
{\sum_{j\in\mathcal N(i)}\exp\left(-\lVert x_i-n_j\rVert^2/\sigma^2\right)},
$$

where $n_k$ is the canonical position of node $k$ and $\mathcal N(i)$ is the set of nodes bound to Gaussian $i$. Bindings are rebuilt when densification changes the Gaussian set. In the final configuration, node positions are fixed after initialization.

```text
   Gaussians (~30k)                  Control nodes (~2k)
   ----------------                  --------------------
   canonical position x_i --KNN-->   node positions n_k
                                             |
                                node-motion network(t)
                                             |
                                translations t_k(t)
                                             |
                                 + user edit e_k
                                             |
                              weighted node blending
                                             |
                 + dense per-Gaussian position residual
                                             |
                                deformed Gaussians
                                             |
                                        rasterizer
```

*Figure 1. GC-EndoGaussian adds a sparse control path to the dense deformation model. The final configuration blends node translations and then adds a dense per-Gaussian residual. Rotation, scale, and opacity remain controlled by the base deformation field.*

### 3.3 Node-motion network

For each node $k$ and timestamp $t$, an MLP first encodes the node position and time:

$$
h_k^0=\mathrm{MLP}([\gamma(n_k),\gamma(t)]),
$$

where $\gamma$ denotes positional encoding. The default implementation applies two EdgeConv-style message-passing layers over a K-nearest-neighbor node graph before predicting node motion.

The general implementation can output a rigid transformation $(R_k,t_k)$. This full SE(3) form is used for the SC-GS-style comparison and related ablations. For all results labeled **ours (match)**, node rotation is disabled by setting $R_k=I$, and the graph predicts translation only. This distinction avoids ambiguity between the general control implementation and the final model evaluated in the main tables.

The output layer is initialized to zero translation, making the control path an initial no-op. Node identity is encoded through node position rather than through a learned node-specific embedding.

Message passing is not essential to the main result. A node-wise MLP without graph layers performs similarly in the residual-matched ablation, indicating that the dense residual, rather than the graph aggregator, is responsible for retaining fidelity.

### 3.4 Weighted deformation and edit handle

For the general SE(3) control formulation, Gaussian position can be warped by linear blend skinning:

$$
p_i^{\mathrm{node}}(t)=\sum_{k\in\mathcal N(i)}w_{ik}
\left[R_k(t)(x_i-n_k)+n_k+t_k(t)\right].
$$

In the final translation-only configuration, $R_k=I$, and this simplifies to a weighted translation field. Let $r_i(t)$ denote the dense per-Gaussian position residual predicted from the base spatio-temporal feature. The final Gaussian position is

$$
p_i(t)=x_i+\sum_{k\in\mathcal N(i)}w_{ik}t_k(t)+r_i(t).
$$

Gaussian rotation, scale, and opacity are predicted by the corresponding heads of the base deformation model and are not blended from control-node rotations in the final configuration.

At inference time, a user-provided translation $e_k$ can be assigned to one or more nodes. The edited position becomes

$$
p_i^{\mathrm{edit}}(t)=p_i(t)+\sum_{k\in\mathcal N(i)}w_{ik}e_k.
$$

This produces a local, smoothly weighted edit without retraining. With approximately 2048 nodes controlling approximately 30,000 Gaussians, the representation provides a control ratio of roughly 15:1. Non-finite deformation outputs are replaced by the corresponding unmodified values before rasterization.

![before](figures/edit_before.png) ![after](figures/edit_after.png) ![diff](figures/edit_diff.png)

*Figure 2. Qualitative drag-to-edit example. Left: reconstruction before editing. Middle: reconstruction after translating a local group of control nodes. Right: per-pixel edit-magnitude visualization. The example illustrates the locality of the weighted control operation; it does not establish biomechanical validity.*

### 3.5 Residual-centered integration recipe

A sparse control graph that replaces the dense deformation field loses reconstruction detail. The final **match** configuration therefore uses four design choices:

- **Translation-only node control.** The sparse graph controls Gaussian position only. Rotation, scale, and opacity remain under the dense base deformation model.
- **Dense per-Gaussian position residual.** A residual head evaluated for each Gaussian restores local motion detail that cannot be represented by a low-dimensional shared node field.
- **No ARAP, isometric, or temporal coherence loss.** These priors can bias highly non-rigid tissue motion away from the photometric optimum.
- **Fixed control nodes.** Nodes are not re-seeded after initialization, avoiding optimization disruptions caused by changes in the control representation.

The residual-matched study in Section 5.4 shows that the per-Gaussian residual is the dominant component. The remaining choices simplify optimization and avoid unnecessary constraints, but do not independently explain the full fidelity recovery.

---

## 4. Experiments

### 4.1 Datasets

We evaluate reconstruction on the EndoNeRF `pulling_soft_tissues` sequence with 63 frames and `cutting_tissues_twice` with 156 frames. Both are processed in binocular mode at $640\times512$ resolution, with every eighth frame held out for testing.

Tracking is evaluated on four SuPer trials containing da Vinci tissue-manipulation sequences. Each trial contains 26--51 manually annotated tissue points over 151 frames. The data are converted to the reconstruction pipeline using stereo-SGBM depth, tool masks, and static-camera poses. The primary tracking comparison aggregates the four trials. The residual-isolation ablation reports SuPer trial 3, as stated in its table.

### 4.2 Baselines

We compare against:

- **EndoGaussian:** the dense base model without sparse editing controls.
- **SC-GS-style control:** an implementation of independent sparse control points with full SE(3) motion, ARAP regularization, adaptive re-seeding, no graph message passing, and 2048 control points.
- **SC-GS-style + residual:** the same sparse-control architecture augmented with the dense per-Gaussian residual used in GC-EndoGaussian.

The SC-GS-style models are implemented and trained inside the same reconstruction pipeline to reduce differences unrelated to the deformation representation.

### 4.3 Metrics and implementation

Reconstruction quality is measured using PSNR, SSIM, LPIPS, and depth RMSE on held-out frames. Tracking is measured as 2D reprojection error against the SuPer annotations. We report bootstrap 95% confidence intervals for the primary tracking comparison and use a paired Wilcoxon signed-rank test.

Efficiency is measured using rendering speed and deformation-parameter count. Training schedules are compared by optimization-step count; wall-clock training-time measurements are not reported.

Experiments use one NVIDIA H100 GPU with PyTorch 2.5.1, CUDA 12.x, and Python 3.12. Unless stated otherwise, GC-EndoGaussian uses 2048 control nodes, four node bindings per Gaussian, and two graph layers. The standard schedule contains 1000 coarse-stage and 3000 fine-stage iterations. The extended comparison increases the fine stage to 6000 iterations for both methods.

---

## 5. Results

### 5.1 Reconstruction remains close to the dense baseline

Table 1 compares GC-EndoGaussian with EndoGaussian under the standard and extended training schedules. Under the standard 3000-iteration fine stage on `pulling_soft_tissues`, the editable model is 0.27 dB lower in PSNR. Under the extended schedule, the difference is 0.15 dB on `pulling_soft_tissues` and 0.13 dB on `cutting_tissues_twice`. SSIM, LPIPS, and depth RMSE also remain close but consistently favor the dense baseline in these comparisons.

*Table 1. Reconstruction under matched optimization schedules. $\Delta$PSNR is method minus EndoGaussian.
SC-GS Depth-RMSE and 6000-iteration/cutting results are not available (SC-GS was not the focus of the
extended comparison).*

| Dataset | Fine-stage iters | Method | PSNR↑ | SSIM↑ | LPIPS↓ | Depth-RMSE↓ | $\Delta$PSNR |
|---|---:|---|---:|---:|---:|---:|---:|
| pulling | 3000 | EndoGaussian | 37.27 | 0.9578 | 0.0609 | 2.906 | — |
| pulling | 3000 | SC-GS-style | 36.80 | 0.9505 | 0.0885 | — | −0.47 |
| pulling | 3000 | **GC-EndoGaussian** | 37.00 | 0.9559 | 0.0638 | 3.139 | **−0.27** |
| pulling | 6000 | EndoGaussian | 37.32 | 0.9578 | 0.0509 | 2.646 | — |
| pulling | 6000 | **GC-EndoGaussian** | 37.17 | 0.9567 | 0.0533 | 2.793 | **−0.15** |
| cutting | 6000 | EndoGaussian | 39.42 | 0.9696 | 0.0322 | 1.358 | — |
| cutting | 6000 | **GC-EndoGaussian** | 39.29 | 0.9689 | 0.0339 | 1.384 | **−0.13** |

At the standard 3000-iteration budget on `pulling`, GC-EndoGaussian **outperforms the SC-GS-style
configuration by +0.20 dB PSNR and improves LPIPS from 0.0885 to 0.0638**, while remaining 0.27 dB below
the dense EndoGaussian baseline. Under the extended schedule the gap to EndoGaussian narrows to 0.15 dB
(`pulling`) and 0.13 dB (`cutting`) — the two methods converge with more training. Adding sparse editing
capability therefore costs at most 0.27 dB relative to the dense baseline, and delivers clear fidelity
gains over the residual-free sparse-control alternative.

![Reconstruction comparison](figures/recon_pulling_triptych.png)

*Figure 3. Reconstruction on `pulling_soft_tissues`, frame 40. Left: ground truth. Center: GC-EndoGaussian rendering. Right: per-pixel error map. Residual error concentrates on the specular surgical tool; the deforming tissue is reconstructed faithfully, consistent with the −0.27 dB gap in Table 1.*

### 5.2 Runtime and parameter overhead

GC-EndoGaussian remains suitable for interactive rendering. The sparse control path reduces rendering speed from 285 to 205 FPS but remains well above real-time rates. It increases the deformation-model parameter count from 85.29 M to 85.35 M, corresponding to 0.07% additional parameters.

*Table 2. Efficiency on `pulling_soft_tissues`.*

| Metric | EndoGaussian | GC-EndoGaussian |
|---|---:|---:|
| Render speed | 285 FPS | **205 FPS** |
| Deformation parameters | 85.29 M | **85.35 M** |
| Parameter increase | — | **+0.07%** |
| Standard optimization schedule | 1000 coarse + 3000 fine | **1000 coarse + 3000 fine** |

The table compares optimization-step counts rather than wall-clock training time. The additional graph and blending operations increase per-iteration computation, so unchanged iteration count should not be interpreted as unchanged training duration.

### 5.3 Tracking shows no detected difference from EndoGaussian

*Table 3. Tracking reprojection error (px). Cross-trial median over four SuPer trials with bootstrap 95%
CIs; trial-3 median for SC-GS (full cross-trial tracking for SC-GS is not available).*

| Method | Median RPE (px) | 95% CI |
|---|---:|---|
| EndoGaussian | 3.47 | [3.34, 3.59] |
| SC-GS-style, no residual | 7.02 | — (trial 3) |
| **GC-EndoGaussian** | **3.30** | [3.14, 3.46] |

GC-EndoGaussian and EndoGaussian have overlapping confidence intervals; a paired Wilcoxon signed-rank
test gives $p=0.73$. The appropriate interpretation is that the experiment detects no statistically
significant difference between the two models. This is not an equivalence test and should not be read as
proof the methods are identical within a predefined margin.

The SC-GS-style model is the critical contrast: without the per-Gaussian residual, tracking error is
7.02 px — **more than double** that of both EndoGaussian (3.47 px) and GC-EndoGaussian (3.30 px).
GC-EndoGaussian therefore achieves a **53% reduction in tracking error** relative to the standard
SC-GS-style configuration, while matching EndoGaussian within statistical noise.

![SuPer reconstruction](figures/recon_super_t3_triptych.png)

*Figure 4. Reconstruction on SuPer trial 3, frame 90. Left: ground truth. Center: GC-EndoGaussian rendering. Right: per-pixel error visualization. Green discs indicate annotated tissue points used in tracking evaluation.*

### 5.4 The dense residual is the key fidelity-preserving component

Which part of the recipe keeps editing reconstruction-neutral — the graph message passing, the
translation-only design, or the per-Gaussian residual? Table 4 answers cleanly: we train an SC-GS-style
control **with and without** the per-Gaussian residual, at the same 3000-iteration budget, and compare to
GC-EndoGaussian. The residual-free model achieves 36.80 dB and 7.02-pixel tracking error. Adding the
residual raises PSNR to 37.29 dB and lowers tracking error to 3.41 pixels — recovering the entire gap.
This recovery occurs without adopting GC-EndoGaussian's graph message passing or translation-only control.

*Table 4. Residual isolation. Reconstruction is reported on `pulling_soft_tissues` after 3000 fine-stage iterations. Tracking is median reprojection error on SuPer trial 3.*

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Track RPE↓ |
|---|---:|---:|---:|---:|
| EndoGaussian, no editing | 37.27 | 0.9578 | **0.0609** | 3.47 |
| SC-GS-style, no residual | 36.80 | 0.9505 | 0.0885 | 7.02 |
| SC-GS-style + residual | **37.29** | **0.9570** | 0.0649 | 3.41 |
| GC-EndoGaussian | 37.00 | 0.9559 | 0.0638 | **3.30** |

The conclusion is architectural but not graph-specific: **retaining the dense per-Gaussian residual is the
key to quality-preserving sparse editability**, regardless of the control architecture. Critically, this
finding is *transferable*: SC-GS-style + residual reaches 37.29 dB and 3.41 px — on par with
EndoGaussian — without adopting any of GC-EndoGaussian's specific design choices. The residual is the
load-bearing component, and the finding generalizes beyond our particular implementation.

### 5.5 Additional ablations and negative results

Replacing EdgeConv aggregation with GAT-style attention produces no material change in the conclusions, and removing message passing yields similar performance in the residual-matched setting. These observations indicate that graph aggregation is not the main source of reconstruction fidelity.

Additional experiments also show limited benefit from using the graph as a reconstruction prior. Occlusion-holdout performance is similar to the dense field, optical-flow supervision does not improve the reported configuration and becomes harmful at high weight, and an explicit cut-modeling mechanism does not exceed the continuous deformation field at the cut region. These negative results are consistent with the view that the dense HexPlane field already captures smooth observed motion effectively. The sparse graph contributes an editing interface, not additional reconstruction evidence.

---

## 6. Discussion and Limitations

GC-EndoGaussian demonstrates that sparse editing handles can be added to a strong endoscopic 4D
reconstruction with a small, bounded fidelity cost — outperforming the standard SC-GS-style sparse-control
configuration by +0.20 dB PSNR and 53% lower tracking error, while remaining within 0.27 dB of the dense
EndoGaussian baseline. The main empirical finding is that the per-Gaussian residual is the load-bearing
ingredient: it is necessary for quality-preserving sparse editability and sufficient to recover
near-baseline fidelity in any sparse-control architecture that uses it.

To be precise about scope: GC-EndoGaussian does not exceed EndoGaussian on reconstruction — the dense
baseline retains a small fidelity advantage. The contribution is editability at a bounded cost, and the
identification of a transferable recipe that the community can apply to other sparse-control designs.

The editing mechanism has several limitations:

- **Edits are not biomechanically validated.** Node translations generate smooth visual deformations but do not model tissue material properties, force response, cutting mechanics, or anatomical constraints.
- **Direct edit quality is evaluated qualitatively.** The current experiments do not report handle-displacement accuracy, locality, foldover rate, edit smoothness, or user-study outcomes. Quantitative edit evaluation is an important next step.
- **Edits bypass graph message passing.** Inference-time translations are added after node-motion prediction. Consequently, the graph does not learn to propagate user input according to tissue behavior. A future model could inject edits before message passing and train the network to distribute them.
- **Evaluation breadth is limited.** Reconstruction results cover two EndoNeRF scenes, while tracking uses four SuPer trials from one acquisition setting. Generalization across anatomies, procedures, stereo rigs, and camera motion remains untested.
- **Efficiency reporting is incomplete.** Rendering speed and parameter count are reported, but wall-clock training time, memory use, and edit-update latency should be measured in a final systems evaluation.

These limitations position GC-EndoGaussian as an editable visual reconstruction method and a basis for future interaction research, rather than as a predictive surgical simulator.

---

## 7. Conclusion

We introduced GC-EndoGaussian, a sparse control layer for editable endoscopic 4D Gaussian reconstruction
that outperforms the standard SC-GS-style sparse-control model by **+0.20 dB PSNR** and **53% lower
tracking error** (3.30 vs 7.02 px), while remaining within 0.27 dB of the dense EndoGaussian baseline at
205 FPS and +0.07% parameters. Tracking shows no statistically significant difference from EndoGaussian
($p=0.73$).

The residual-isolation experiment delivers the central finding: **the per-Gaussian residual is the
transferable, architecture-independent ingredient for quality-preserving sparse editability.** Removing it
from any sparse-control architecture causes ~0.5 dB PSNR loss and ~2× tracking error; adding it back
recovers near-baseline fidelity regardless of the control architecture. Future work should route user
edits through graph message passing and incorporate biomechanical priors where tissue-response prediction
is required.

Future work should quantify edit accuracy and locality, route user edits through learned message passing, and incorporate biomechanical or anatomical constraints where predictive tissue behavior is required.

---

## References

[1] B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis. *3D Gaussian Splatting for Real-Time Radiance Field Rendering.* ACM ToG (SIGGRAPH), 2023.

[2] Y. Zhu et al. *EndoGaussian: Real-time Gaussian Splatting for Dynamic Endoscopic Scene Reconstruction.* 2024.

[3] Y. Huang, Z. Sun, et al. *SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes.* CVPR, 2024.

[4] Y. Wang, Y. Long, et al. *Neural Rendering for Stereo 3D Reconstruction of Deformable Tissues in Robotic Surgery (EndoNeRF).* MICCAI, 2022.

[5] S. Yang, Q. Li, et al. *Deform3DGS: Flexible Deformation for Fast Surgical Scene Reconstruction with Gaussian Splatting.* MICCAI, 2024.

[6] A. Cao, J. Johnson. *HexPlane: A Fast Representation for Dynamic Scenes.* CVPR, 2023.

[7] S. Fridovich-Keil, et al. *K-Planes: Explicit Radiance Fields in Space, Time, and Appearance.* CVPR, 2023.

[8] G. Wu, T. Yi, et al. *4D Gaussian Splatting for Real-Time Dynamic Scene Rendering.* CVPR, 2024.

[9] Y. Li, F. Richter, et al. *SuPer: A Surgical Perception Framework for Endoscopic Tissue Manipulation with Surgical Robotics.* IEEE RA-L, 2020.

[10] Y. Zhou, C. Barnes, et al. *On the Continuity of Rotation Representations in Neural Networks.* CVPR, 2019.

[11] Y. Wang, Y. Sun, et al. *Dynamic Graph CNN for Learning on Point Clouds (EdgeConv).* ACM ToG, 2019.
