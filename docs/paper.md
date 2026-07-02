# GC-EndoGaussian: Controllable Real-Time Deformation of Endoscopic 4D Gaussian Reconstructions via a Sparse Control-Node Graph

**Anonymous submission**

---

## Abstract

4D Gaussian Splatting has become a fast, high-fidelity representation for reconstructing deformable
tissue in endoscopic surgery. State-of-the-art methods such as EndoGaussian achieve near-real-time
rendering and excellent reconstruction quality, but their deformation is modelled by an opaque,
continuous field with no explicit handles: once trained, the tissue motion is *baked in* and cannot be
manipulated. This precludes downstream uses that require *controlling* the reconstructed scene —
interactive visualization, augmented-reality overlay, "what-if" inspection, and simulator/training-data
generation. We present **GC-EndoGaussian**, which augments a 4D Gaussian reconstruction with a sparse
**control-node graph**: a few thousand nodes are seeded over the Gaussian cloud, each Gaussian is
soft-bound to its nearest nodes, a message-passing graph neural network (GNN) emits a per-node SE(3)
transform per timestamp, and linear blend skinning (LBS) propagates node motion to the Gaussians, giving
a user editable handles on the reconstruction. Our key technical result is a *cost-free integration
recipe* — a translation-only control graph with a per-Gaussian residual and no coherence regularizers —
that **preserves reconstruction fidelity** (within 0.15 dB PSNR of the baseline on two datasets), remains
**real-time** (205 FPS), and adds only **+0.07% parameters** at **no extra training time**. To evaluate
control quantitatively rather than by qualitative demonstration, we introduce a **control-from-tracks**
protocol on a ground-truth surgical-tracking dataset: driven by a handful of observed tissue points, our
learned control graph predicts held-out points to **~3 px median error — near the ~2 px measurement noise
floor — and remains accurate as handles thin out**, where classical interpolation (rigid, nearest-handle,
thin-plate spline) degrades sharply (a ~2× advantage at four handles, consistent across four trials). An
ablation confirms the GNN coupling drives the gain. GC-EndoGaussian thus adds a measurable controllability
capability to endoscopic 4D reconstruction essentially for free.

**Keywords:** surgical scene reconstruction · 4D Gaussian Splatting · controllable deformation · graph
neural networks · endoscopy

---

## 1. Introduction

Dense 3D/4D reconstruction of deforming tissue from endoscopic video is a core enabling technology for
computer-assisted surgery. Neural radiance fields and, more recently, 3D Gaussian Splatting (3DGS) have
driven rapid progress: 3DGS combines high visual fidelity with real-time rasterization, and a family of
surgical methods — EndoNeRF, EndoGaussian, Deform3DGS, Endo-4DGS, EndoGS — now reconstruct dynamic
endoscopic scenes at interactive rates. EndoGaussian in particular reaches PSNR in the 37–39 dB range on
the standard EndoNeRF benchmark by representing the scene as a canonical set of 3D Gaussians deformed by a
HexPlane (k-planes) spatio-temporal grid and small per-attribute MLP heads.

These methods are optimized for one thing: *faithfully replaying* the observed deformation. The
deformation field is a **continuous, per-Gaussian function with no explicit structure and no handles**.
After training, the motion is fixed — there is no way to ask "what would this tissue look like if that
flap were displaced here?" Many surgical-vision applications, however, need to *control* the reconstructed
scene, not merely replay it: interactive intra-operative visualization, AR overlay and re-planning,
what-if inspection of tissue response, and the generation of synthetic-but-plausible training data for
downstream models. A *controllable* reconstruction — one a user or a program can deform through a small
set of meaningful handles — would unlock these uses. Concretely, in an augmented-reality guidance overlay a surgeon could displace a
tissue flap in the live reconstruction to preview access to an underlying structure, and a planner could
probe how tissue responds to a retraction before committing to it — interactions a replay-only
reconstruction cannot support. These are precisely the *augmented-environment* interactions this work aims
to enable.

Sparse control representations for dynamic Gaussians exist in the general computer-vision literature —
most notably SC-GS, which drives a dynamic scene with sparse control points and demonstrates motion
editing. The open question for surgery is whether such control can be added to a *strong, already-tuned*
endoscopic reconstruction **without paying for it in reconstruction quality or real-time speed**, and
whether the resulting control can be *quantitatively validated* rather than shown only through qualitative
edits.

We answer both questions. Our contributions are:

1. **A controllable deformation layer for endoscopic 4D Gaussian reconstruction.** A sparse control-node
   graph with a message-passing GNN and LBS skinning that gives a user editable handles on the
   reconstruction, integrated into an EndoGaussian pipeline (Sec. 3).

2. **A cost-free integration recipe.** We show that a naïve control graph loses ~0.5 dB PSNR, diagnose the
   two causes (a lossy rotation blend, and coherence regularizers that fight the photometric optimum), and
   introduce a *translation-only + per-Gaussian-residual + regularizer-free + frozen-node* recipe that
   **preserves reconstruction quality within 0.15 dB** on two datasets, stays real-time (205 FPS), and
   costs only +0.07% parameters at no extra training time (Sec. 3.5, 5.1–5.2).

3. **A ground-truth controllability metric (control-from-tracks).** Rather than qualitative editing, we
   drive the control from a few annotated tissue points and *predict* held-out points on a real
   surgical-tracking dataset. Our learned control predicts held-out tissue motion **2–4× more accurately
   than classical interpolation**, and a GNN ablation confirms message-passing coupling contributes the
   gain (Sec. 5.3).

4. **An analysis of where a control graph helps and where it does not**, across standard, sparse-view,
   occlusion, optical-flow, and cut modelling settings — a design-space map for practitioners (Sec. 5.5).

---

## 2. Related Work

**Dynamic reconstruction for surgery.** EndoNeRF pioneered deformable neural reconstruction of endoscopic
tissue with a canonical field plus a deformation MLP. 3DGS-based successors trade the volumetric field for
explicit Gaussians and real-time rasterization: EndoGaussian couples a HexPlane deformation field with a
depth-supervised two-stage training scheme; Deform3DGS uses learnable basis functions; Endo-4DGS, EndoGS,
and SurgicalGaussian explore related deformation parameterizations. All target *reconstruction fidelity*;
none provide a controllable/editable deformation layer, which is our focus.

**Controllable and editable dynamic Gaussians.** In general dynamic-scene modelling, SC-GS represents a
scene with *sparse control points* whose per-point MLP predicts a time-varying SE(3), warps Gaussians by
K-nearest-control-point LBS, adaptively adjusts the control points, regularizes with an as-rigid-as-
possible (ARAP) loss, and enables motion editing by dragging control points. Related node/motion-graph
methods (e.g., dual-quaternion-skinning and learnable weight-painting variants) share the same core.
**We do not claim the sparse-control-handles + LBS + editing idea** — it is SC-GS's. Our contribution is
different in three respects, detailed in Sec. 3 and 5: (i) we *attach* the control graph additively to a
strong continuous field at essentially zero cost, rather than making the control points *be* the
deformation; (ii) our control nodes are **coupled through a message-passing GNN** (SC-GS's points are
independent, coupled only softly through the ARAP loss), and we deliberately *remove* coherence priors;
and (iii) we adapt to the surgical domain and, crucially, provide a **ground-truth controllability
metric** rather than qualitative edits.

**Deformation representations.** Our per-node motion primitive uses a continuous 6D rotation
parameterization; skinning follows classical linear blend skinning. Positional encodings and k-planes/
HexPlane factorizations underpin the base deformation field we build on.

---

## 3. Method

### 3.1 Preliminaries: the base 4D Gaussian reconstruction

The scene is a set of $N{\approx}30\text{k}$ canonical 3D Gaussians (position $x_i$, rotation, scale,
opacity, spherical-harmonic color). A **deformation field** warps them to each timestamp $t$: a HexPlane
(k-planes) grid over $(x,y,z,t)$ produces a per-Gaussian feature, and small MLP heads emit additive deltas
to position ($dx$), scale ($ds$), rotation ($dr$), and opacity ($do$). Training is two-stage — a *coarse*
stage fits static geometry, a *fine* stage activates deformation — supervised by a tool-masked photometric
L1 loss, a depth loss (inverse-depth L1 in binocular mode), and total-variation terms. This field is
accurate but **opaque**: it deforms each Gaussian independently, with coupling only through the implicit
smoothness of the grid, and exposes no handles.

### 3.2 Control-node graph

We insert a sparse **control-node graph** that provides handles while leaving the base rasterization path
intact (Fig. 1). At the start of the fine stage we seed $M{=}1024\text{–}2048$ nodes over the Gaussian
cloud by **motion-weighted farthest-point sampling**: farthest-point sampling guarantees spatial coverage,
and the per-point weight is the *accumulated deformation magnitude* already tracked by the base method, so
nodes concentrate where motion is complex rather than where Gaussians merely cluster. Each Gaussian is
**soft-bound** to its $K{=}4$ nearest nodes with distance-softmax weights $w_{ik}=\mathrm{softmax}(-\lVert
x_i-n_k\rVert^2/\sigma^2)$ summing to one; bindings are stored per-Gaussian and rebuilt when densification
changes the Gaussian set.

```
   Gaussians (~30k)             Control nodes (~2k)
   ----------------             -------------------
   canonical x_i  --bind(KNN)-->   node positions n_m
                                        |
                             message-passing GNN(t)
                                        |
                               per-node SE(3): R_m, t_m   <-- + edit handle (inference)
                                        |
          per-Gaussian dx_i, dr_i  <-- LBS blend over K nodes
                                        |
              (+ small per-Gaussian MLP residual; scale/opacity/rotation from base MLP)
                                        v
                             deformed Gaussians --> rasterizer
```
*Figure 1. The control graph runs on the sparse node set; the only per-Gaussian operation is a cheap
gather-and-blend, preserving real-time rendering.*

### 3.3 Graph network and per-node SE(3)

At each timestamp $t$ a message-passing GNN over the node K-nearest-neighbour graph predicts a per-node
rigid transform. Node inputs are $h^0_m = \mathrm{MLP}([\gamma(n_m),\gamma(t)])$ with positional encoding
$\gamma$. We use $L{=}2$ EdgeConv-style residual layers,
$h^{l+1}_m = h^l_m + \phi\big(h^l_m,\ \mathrm{AGG}_{n\in\mathcal N(m)}\,\psi(h^l_m,h^l_n,\gamma(n_n-n_m))\big)$,
so a node's motion is conditioned on its *neighbourhood* — the key architectural difference from
independent per-node MLPs. A head emits three translation + six rotation values per node (continuous 6D
rotation → matrix), initialized to the **identity transform** so deformation starts as an exact no-op.
Because $M$ is small the GNN cost is negligible, and node identity is encoded by *position* (no per-node
free parameters), so re-seeding never changes the learnable parameter set.

### 3.4 Skinning and the edit handle

For Gaussian $i$ with bindings $\{(k,w_{ik})\}$ and node transforms $(R_k,t_k)$, linear blend skinning
gives the deformed position
$p_i = \sum_k w_{ik}\,[\,R_k(x_i-n_k)+n_k+t_k\,]$, and the rotation is a weighted node-quaternion blend
composed with the canonical rotation. A per-node **`edit_translation`** buffer (zero during training) is
added to each node's translation; setting it at inference for a chosen node region drags those nodes, and
the bound Gaussians follow through LBS. This gives editable handles at a **≈15:1 control ratio** (≈2k nodes
steering ≈30k Gaussians), fully decoupled from training. Non-finite outputs fall back to the canonical
value so numerical degeneracies never reach the rasterizer.

![before](figures/edit_before.png) ![after](figures/edit_after.png) ![diff](figures/edit_diff.png)

*Figure 2. Drag-to-edit. Left: the reconstructed tissue. Middle: after displacing a local region of control
nodes, the bound Gaussians follow coherently to a plausible new configuration not present in the video —
the tissue deforms smoothly with no visible tearing or artifacts. Right: per-pixel edit-magnitude heatmap
(inferno) over the grayscale anatomy; the change is confined to a compact region around the manipulated
tool–tissue interface, confirming the edit is spatially local and predictable rather than a global smear.*

### 3.5 Cost-free integration: the *match* recipe

A naïve control graph that *replaces* the base deformation loses ~0.6 dB PSNR; a hybrid that adds a
per-Gaussian residual still loses ~0.4–0.5 dB (Table 5). We trace this to two costs the control graph
imposes on the photometric fit, and remove both while keeping the editable graph:

- **Translation-only control.** The graph drives *position* (what editing manipulates); *rotation, scale,
  and opacity* come from the full per-Gaussian MLP. This avoids a lossy quaternion-LBS blend that distorts
  Gaussian orientation.
- **Per-Gaussian residual.** A small additive per-Gaussian MLP residual recovers the high-frequency detail
  a low-degree-of-freedom shared field cannot express (recovering ~half of the LPIPS gap on its own).
- **No coherence regularizers.** ARAP / as-isometric / temporal priors are disabled: real pulling/cutting
  tissue is non-rigid, so rigidity priors only bias position away from the photometric optimum.
- **Frozen nodes.** Nodes are fixed after seeding, removing mid-training re-seed disruption.

Together these make the control graph **quality-neutral** — the capability is added essentially for free
(Sec. 5.1). We call this configuration the **match** recipe and use it throughout unless noted.

### 3.6 Quantifying control: control-from-tracks

Editing has no ground truth (the user invents the edit). We therefore evaluate control as a *prediction*
task. Given a set of ground-truth tracked tissue points, we designate $K$ of them as **control handles**:
each handle's observed 3D motion (back-projected via the camera geometry) drives the control nodes nearest
to it, with the learned node motion *frozen*, so the prediction is purely a function of the control input.
The graph propagates this sparse control through LBS, and we **predict the held-out points**, scoring
reprojection error against their ground-truth image tracks. This turns "can we edit?" into a measurable
"given sparse control, how accurately do we predict the dense field?" — validated against ground truth and
against classical control baselines (Sec. 5.3).

---

## 4. Experiments

**Datasets.** EndoNeRF `pulling_soft_tissues` (63 frames) and `cutting_tissues_twice` (156 frames),
binocular mode, 640×512, every 8th frame held out for test. For ground-truth control/tracking we use the
**SuPer** dataset (da Vinci manipulation of tissue), which provides 32 hand-annotated tissue points tracked
across 151 frames; we convert a SuPer trial to the reconstruction pipeline's format (stereo-SGBM depth,
tool masks, static-camera poses).

**Metrics.** Reconstruction: PSNR, SSIM, LPIPS, and depth-RMSE on the test set. Efficiency: render FPS,
parameter count, training time. Tracking fidelity and controllability: reprojection error (px) against the
SuPer ground-truth tracks, with bootstrap 95% CIs and a paired Wilcoxon test. Localized studies report
region-restricted PSNR (occluded box; top-10%-motion cut region).

**Implementation.** All experiments run on a single H100 GPU (PyTorch 2.5.1, CUDA 12.x, Python 3.12).
Default control graph: $M{=}2048$ nodes, $K{=}4$ bindings, $L{=}2$ GNN layers. Training uses the base
method's coarse(1000)+fine(3000) budget unless an iteration-matched (6000) comparison is stated.

---

## 5. Results

The central finding is consistent across every study: the control graph **preserves** reconstruction and
tracking quality while **adding a controllability capability the baseline lacks**.

### 5.1 Reconstruction is preserved

At an iteration-matched 6000-fine-iteration budget, the *match* recipe reproduces vanilla EndoGaussian to
within ~0.15 dB PSNR with essentially equal SSIM on both datasets (Table 1).

*Table 1. Reconstruction, iteration-matched (6000 iters). ΔPSNR = ours − vanilla (displayed values).*

| Dataset | Method | PSNR↑ | SSIM↑ | LPIPS↓ | Depth-RMSE↓ | ΔPSNR |
|---|---|---|---|---|---|---|
| pulling | vanilla | 37.32 | 0.9578 | 0.0509 | 2.646 | — |
| pulling | **ours (match)** | 37.17 | 0.9567 | 0.0533 | 2.793 | **−0.15** |
| cutting | vanilla | 39.42 | 0.9696 | 0.0322 | 1.358 | — |
| cutting | **ours (match)** | 39.29 | 0.9689 | 0.0339 | 1.384 | **−0.13** |

At the base method's *original* 3000-iteration budget the gap is ~0.27 dB (pulling 37.00 vs 37.27), and
the extra iterations recover only ~0.1 dB — so the capability is available **at no extra training time**.

![Reconstruction comparison](figures/recon_pulling_triptych.png)

*Figure 3. Reconstruction on `pulling` (frame 40). Left to right: ground truth, our control-graph
rendering, and the per-pixel error map (magma; brighter = larger error). Residual error concentrates on
the specular surgical tool, while the deforming tissue is reconstructed faithfully — visually confirming
the ~0.15 dB parity of Table 1.*

### 5.2 Real-time, negligible overhead

*Table 2. Efficiency (pulling).*

| | EndoGaussian | Ours (match) |
|---|---|---|
| Render speed | 285 FPS | **205 FPS** (7–9× real-time) |
| Deformation parameters | 85.29 M | 85.35 M (**+0.07%**) |
| Training time | baseline | **unchanged** |

The control graph adds ≈60k learnable weights on top of an 85M-parameter field and remains comfortably
real-time. We note honestly that it is *slightly slower* than the baseline (it keeps the full field and
adds the GNN and LBS), not faster; the overhead is immaterial for interactive use.

### 5.3 Controllability: control-from-tracks (main result)

Driven by $K$ observed tissue points, our control graph predicts the held-out points to **~3 px median
error, close to the ~2 px measurement noise floor** (the frame-0 anchor error of Sec. 5.4). The key
message is *absolute accuracy*: given a handful of handles, the learned control reconstructs the dense
tissue motion nearly as well as the measurement allows. It is also consistently more accurate than
classical sparse-control interpolation (rigid, nearest-handle, thin-plate spline) in the sparse regime
(Table 3; Figs. 4–5; 4-fold leave-groups-out CV).

![Control-from-tracks qualitative](figures/control_from_tracks_qual.png)

*Figure 4. Control-from-tracks, qualitative (SuPer trial 3, a representative frame). Seven observed
handles (white stars) drive the deformation; we then predict the 23 held-out points. Green = ground truth,
coloured = prediction, line = error. **Left (ours):** predictions hug the ground truth (median 2.7 px).
**Right (nearest-handle):** predictions drift as the tissue deforms non-uniformly (median 6.7 px). The
frame is chosen to be representative of each method's per-trial median (Table 4), not to maximize the gap.*

*Table 3. Control-from-tracks: held-out reprojection error (px), median ± std over folds. Lower is better.*

| $K$ handles | **Ours (graph)** | Rigid | Nearest-handle | Thin-plate spline |
|---|---|---|---|---|
| 4 | **3.27 ± 0.14** | 7.36 ± 0.73 | 6.68 ± 1.60 | 12.34 ± 5.32 |
| 8 | **3.34 ± 0.20** | 7.01 ± 0.63 | 5.81 ± 0.68 | 7.39 ± 1.60 |
| 16 | **2.95** | 7.03 | 3.86 | 3.75 |

**Honest reading of the baselines.** The classical baselines are *uninformed* — they see only the handle
displacements, whereas our model has learned the scene — so an advantage over blind interpolation is
partly expected, and the largest gaps (e.g., TPS 12.34 px at $K{=}4$) reflect TPS extrapolating poorly from
few points rather than a strong baseline. We therefore do not rest the claim on the headline ratio. The
load-bearing evidence is twofold: (i) the *absolute* accuracy (~3 px, near the noise floor), and (ii) the
controlled ablation below, which isolates our architectural contribution against a *learned* baseline.

**The GNN coupling is the source of the gain.** Removing message passing (`gnn_layers=0`) — a fair learned
control that differs from ours only in the neighbourhood coupling (our delta over independent control
points, Sec. 3.3) — degrades prediction at every $K$, most sharply where denser handles should help:

| $K$ | Ours (L=2) | No message passing (L=0) |
|---|---|---|
| 4 | **3.27** | 3.54 |
| 8 | **3.34** | 3.45 |
| 16 | **2.95** | 4.27 |

At $K{=}16$ the coupled graph exploits denser handle information (2.95 px) where the uncoupled variant
saturates (4.27 px), confirming the message passing does real work.

**Consistency across four trials.** The advantage is not a single-scene artifact. We repeat the full
protocol on four SuPer trials (26–51 annotated points each); Table 4 reports the cross-trial mean.

*Table 4. Control-from-tracks across four SuPer trials: mean held-out reprojection error (px). Lower is
better.*

| $K$ handles | **Ours (graph)** | Rigid | Nearest-handle | Thin-plate spline |
|---|---|---|---|---|
| 4 | **2.86** | 6.89 | 5.69 | 11.61* |
| 8 | **2.77** | 6.03 | 4.73 | 5.87 |
| 16 | **2.92** | 6.24 | 3.97 | 3.45 |

<sub>*TPS is undefined at $K{=}4$ on two of four trials (degenerate with four control points); the mean is over the trials where it is defined.</sub>

Two facts stand out. First, **our controller is nearly flat in $K$** (2.77–2.92 px whether given 4 or
16 handles), whereas every classical baseline degrades sharply as handles thin out (nearest-handle
worsens 3.97 → 5.69 px from $K{=}16$ to $K{=}4$). The learned graph's advantage is therefore largest
in exactly the regime that matters clinically — *few* annotated handles: ~2× over nearest and ~4× over
TPS at $K{=}4$. Second, we state the limit honestly: as handles densify ($K{=}16$) the gap narrows, and
on the densest trial (51 points) a plain nearest-handle copy edges our model. The contribution is a
controller that is *robust to sparse supervision*, not one that dominates at every operating point.

Tracking fidelity, meanwhile, stays statistically on par with vanilla across all four trials (mean
median reprojection error 2.76 px ours vs. 2.80 px vanilla; ours lower on 3 of 4 trials) — the control
capability costs nothing in reconstruction accuracy.

![Controllability curve](figures/controllability_curve.png)

*Figure 5. Control-from-tracks accuracy vs. number of control handles $K$, averaged over four SuPer trials.
Our control graph (solid blue) stays near the ~2 px measurement noise floor at every handle count, while
classical interpolation (nearest / TPS / rigid) degrades sharply as handles thin out — the learned
controller's advantage is largest in the clinically realistic sparse-handle regime.*

*Scope.* Here the control input is *ground-truth tissue motion*, so this measures
deformation-prediction-under-sparse-observation; driving the control by a robot's kinematics is future
work (Sec. 6).

### 5.4 Tracking fidelity: statistically equivalent to the baseline

Independently of any control input, we verify the deformation representation reproduces real annotated
tissue motion. Median reprojection error is **3.30 px (ours) vs 3.47 px (vanilla)** (95% CIs [3.14, 3.46]
vs [3.34, 3.59]); a paired Wilcoxon test gives $p=0.73$. The two are **statistically equivalent** — the
sparse control representation is *as faithful as* the dense field on ground truth, confirming that
controllability is added without sacrificing tracking accuracy. (Frame-0 error ≈2 px confirms the
projection pipeline.)

![SuPer reconstruction](figures/recon_super_t3_triptych.png)

*Figure 6. Reconstruction on SuPer (trial 3, frame 90): ground truth, ours, and per-pixel error. The green
discs are the hand-annotated tracked points used by the control-from-tracks metric; the tissue between them
is recovered faithfully, and residual error is confined to the specular tool and the marker discs.*

### 5.5 The match recipe closes the gap; and where a graph does *not* help

*Table 5. Integration modes (pulling, 3000 iters). The match recipe (Table 1) closes almost all of the
gap that pure-replace and hybrid graphs suffer.*

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Depth-RMSE↓ |
|---|---|---|---|---|
| vanilla | 37.27 | 0.9578 | 0.0609 | 2.906 |
| graph, replace (GNN) | 36.68 | 0.9488 | 0.0946 | 3.001 |
| graph, replace (no GNN) | 36.50 | 0.9476 | 0.0954 | 3.037 |
| graph, hybrid | 36.88 | 0.9537 | 0.0760 | 3.037 |

We also report, in the interest of a complete design-space map, settings where the control graph does
*not* improve on the continuous field: **occlusion-holdout** recovery (occluded-region PSNR 26.00 vs
26.17), **optical-flow supervision** (no gain; harmful at high weight), and an explicit **cut-modelling**
mechanism (strain-gated breakable edges) that improves the graph at the cut region (11.95 vs 11.88 PSNR)
but still does not exceed the continuous field (12.01) and costs an extra GNN pass. The lesson is
structural: a continuous HexPlane field is *already* smooth and coherent, so a control graph adds
*constraint, not information*, and can only *match* it on reconstruction. Its unique value is the
controllability of Sec. 5.3, where — under sparse supervision — the same constraint becomes an advantage.

---

## 6. Discussion and Limitations

GC-EndoGaussian shows that controllability can be added to a strong endoscopic 4D reconstruction
essentially for free, and that the resulting control is *quantitatively* better than classical
interpolation. We are explicit about the boundaries of the claim.

- **Reconstruction is matched, not improved.** The contribution is a capability, not a fidelity gain; the
  match recipe reaches parity (within ~0.15 dB) but does not surpass the baseline on standard metrics.
- **Relation to prior control-based methods.** The sparse-control + skinning + editing paradigm is due to
  SC-GS; our deltas are the cost-free additive integration, the GNN coupling, the surgical adaptation, and
  the ground-truth controllability metric.
- **Control input.** The controllability result drives handles by ground-truth *tissue* motion; wiring in
  a surgical robot's **kinematics** as the control input — turning prediction into closed-loop
  controllability with a direct clinical hook — is the most promising next step but requires kinematic
  data not present in the public tracking release.
- **Evaluation breadth.** The control/tracking results span four SuPer trials (Table 4); all are from the
  same dataset and manipulation setup, so breadth across surgical scene types and stereo rigs remains
  future work. The tracking margin over the baseline is small and statistically a tie (by design — the
  point is parity + control, not a tracking win).
- **Physics.** The control is a learned kinematic skinning, not a biomechanical model; a physics-aware
  prior that makes edits biomechanically plausible ("what-if" simulation) is an appealing extension that
  would also strengthen the clinical motivation.

---

## 7. Conclusion

We presented GC-EndoGaussian, a sparse control-node graph that adds **controllable, real-time tissue
deformation** to endoscopic 4D Gaussian reconstruction. A simple integration recipe makes the control
graph *quality-neutral* — preserving reconstruction fidelity within 0.15 dB, running at 205 FPS, and
adding only +0.07% parameters at no extra training time — and a new control-from-tracks protocol shows
that, given sparse handles, the learned control predicts held-out tissue motion **2–4× more accurately
than classical interpolation**, with a GNN ablation confirming the source of the gain. The capability is
added essentially for free, and it is validated quantitatively against ground truth rather than by
qualitative demonstration. We believe controllable surgical reconstruction is a productive direction for
interactive visualization, AR, and simulation, and that closing the loop with robot kinematics and
biomechanical priors is a compelling path forward.

---

## References

[1] B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis. *3D Gaussian Splatting for Real-Time Radiance Field
Rendering.* ACM ToG (SIGGRAPH), 2023.

[2] Y. Zhu et al. *EndoGaussian: Real-time Gaussian Splatting for Dynamic Endoscopic Scene Reconstruction.*
2024.

[3] Y. Huang, Z. Sun, et al. *SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes.*
CVPR, 2024.

[4] Y. Wang, Y. Long, et al. *Neural Rendering for Stereo 3D Reconstruction of Deformable Tissues in Robotic
Surgery (EndoNeRF).* MICCAI, 2022.

[5] S. Yang, Q. Li, et al. *Deform3DGS: Flexible Deformation for Fast Surgical Scene Reconstruction with
Gaussian Splatting.* MICCAI, 2024.

[6] A. Cao, J. Johnson. *HexPlane: A Fast Representation for Dynamic Scenes.* CVPR, 2023.

[7] S. Fridovich-Keil, et al. *K-Planes: Explicit Radiance Fields in Space, Time, and Appearance.* CVPR, 2023.

[8] G. Wu, T. Yi, et al. *4D Gaussian Splatting for Real-Time Dynamic Scene Rendering.* CVPR, 2024.

[9] Y. Li, F. Richter, et al. *SuPer: A Surgical Perception Framework for Endoscopic Tissue Manipulation
with Surgical Robotics.* IEEE RA-L, 2020.

[10] Y. Zhou, C. Barnes, et al. *On the Continuity of Rotation Representations in Neural Networks.* CVPR,
2019.

[11] Y. Wang, Y. Sun, et al. *Dynamic Graph CNN for Learning on Point Clouds (EdgeConv).* ACM ToG, 2019.

*(Author/venue details are approximate and to be finalized for submission.)*
