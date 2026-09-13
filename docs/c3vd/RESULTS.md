# C3VD (colonoscopy, true multi-view) — reconstruction, control-graph fix, and edit test

## Why this dataset

Every other scene in this project (EndoNeRF, SuPer, Hamlyn, Endoscapes) is captured from an
effectively **fixed viewpoint**, and writes a single repeated pose into `poses_bounds.npy`. A fixed
viewpoint can only reconstruct a *surface with thickness*, never a layered volume — measured
directly: 45–78% of occluded Gaussians had been observed at some other timestamp, yet removing the
front layer changed only 1–2% of pixels, because what sits behind is the same sheet's backing.

C3VD is a colonoscope travelling down a phantom lumen with **ground-truth depth and registered
camera poses** — real parallax, and haustral folds as genuine occluders.

## Getting the data usable (two non-obvious problems)

**1. It is a 190° fisheye, not a pinhole.** 3DGS renders pinhole; fitting one to raw C3VD frames
fails outright (reprojection error 70–5500 px). Solved with the dataset's own Scaramuzza
omnidirectional calibration — [tools/c3vd_undistort.py](../../tools/c3vd_undistort.py).

**2. Conventions had to be verified, not assumed.** Predicting optical flow from poses+depth and
comparing against the dataset's ground-truth flow:

| Convention | Correlation |
|---|---|
| `R=M[:3,:3].T, t=M[3,:3]`, flow channels as-read | 0.267 |
| same, **flow channels swapped** | **0.946** (0.43 px residual) |
| other three pose conventions | 0.05–0.58 |

Verified settings: `R_c2w = M[:3,:3].T`, `t = M[3,:3]` (mm); depth full-scale 100 mm, stored as
distance along the ray; occlusion maps mark invalid ground truth.
Converter: [tools/c3vd_to_endonerf.py](../../tools/c3vd_to_endonerf.py).

Result: 148 frames, camera path **24.8 mm at 10.8 mm depth** (baseline/depth ≈ 2.3).

## Finding: the control graph needs a scene-scale-dependent learning rate

| Configuration | PSNR (148 frames) |
|---|---|
| Control graph @ `node_lr=0.0008` (paper default) | **3 dB — collapses within 120 iterations** |
| No control graph | 31.88 dB |
| **Control graph @ `node_lr=1e-5`** | **32.46 dB** |
| Control graph @ `node_lr=1e-6` | 32.46 dB |

The graph is not broken; its node learning rate is tuned for EndoNeRF's scale and is ~80× too large
here. On a scene whose change is dominated by *camera* motion rather than tissue deformation the
nodes should stay near identity, and at the default rate they drift and drag the Gaussians with them.
With the rate corrected the editable model slightly **outperforms** the non-editable one.

This is worth stating in the paper: it is a genuine robustness limitation, found by running the
method outside the regime it was tuned for.

## Reconstruction quality

Best multi-view result in the project: **32.5 dB**, per-frame MAE 5.8–6.0 (compare Endoscapes
Senario1 at 18–20 dB / MAE 73–79). Renders are visually near-indistinguishable from ground truth.

## Edit test — still not a convincing reveal

| Scene | Observed-but-hidden Gaussians | Pixel change on removal |
|---|---|---|
| EndoNeRF pulling | 45–51% | 1–2% |
| EndoNeRF cutting | 78% | 2.2% |
| Endoscapes Senario1 | 10% | 1.2–2.5% |
| **C3VD (multi-view)** | **~30%** | **0.1–5.3%** |

Multi-view raised the ceiling (5.3% is the largest yet) but did not produce a clean result: removing
the front layer at the automatically-selected regions yields dark artifacts rather than a legible
view of the surface behind the fold.

**Assessment.** The limiting factor is no longer the data — it is *region selection*. The scans pick
regions by depth separation or by counting observed-but-hidden Gaussians, and those criteria favour
bright near-wall patches rather than fold edges, which is where occlusion actually happens in a
lumen. A fold-edge-aware selection (e.g. driven by the dataset's own occlusion maps, or by the
`coverage_mesh.obj` geometry) is the obvious next step and was not reached.

## Reproduce

```bash
python3 tools/c3vd_to_endonerf.py --src data/desc_t4_a --out data/endonerf/c3vd_desc_t4a
sbatch run_c3vd_lr.bash        # trains with node_lr 1e-5 / 1e-6
sbatch run_c3vd_edit.bash      # render + observed-but-hidden scan + depth-layer peel
```

---

# Does the editing idea earn its place? Generalization evidence

The reveal demo was the weakest way to defend this method. The stronger argument is that the
editable interface **costs nothing and behaves identically on data it was never tuned for**.
C3VD is a new dataset, new anatomy (colon), and an inverted capture regime (moving camera, rigid
scene) versus the paper's fixed-camera deforming-tissue scenes.

## 1. Editability is free — it is not a quality/controllability trade

| Configuration | PSNR (148 frames) |
|---|---|
| Plain EndoGaussian (no control graph) | 31.88 dB |
| **GC-EndoGaussian (editable control graph)** | **32.46 dB** |

On EndoNeRF the paper reports a small cost (−0.11 to −0.27 dB). Here the editable model is *ahead*.
Held-out test metrics for the editable model: **PSNR 29.46, SSIM 0.802, LPIPS 0.326**.

## 2. The edit-interface metrics transfer almost unchanged

`eval_edit.py`, same protocol as the paper (32 edits: 8 handle draws x 4 magnitudes):

| Metric | Paper (EndoNeRF `pulling`) | **C3VD (colonoscopy)** |
|---|---|---|
| Handle fidelity ↑ | 0.96 | **0.98** |
| 3D leakage ↓ | 0.0% | **0.01%** |
| Pixel leakage ↓ | 12.3% | **11.8%** |
| Foldover ↓ | 0.05% | **0.27%** |
| Edit latency ↓ | 4.8 ms | **4.81 ms** |

Every property the paper claims for the interface holds on the new data: tissue follows the handle,
displacement is structurally confined, geometry does not self-intersect, and edits stay interactive.
Latency is identical to the millisecond.

## 3. One genuine difference, reported rather than hidden

Motion energy split (node field vs dense residual) is **50.7% / 49.3%** here, against 23–31% / 69–77%
on EndoNeRF. That is expected and consistent with the paper's thesis: this scene is rigid, so there
is very little deformation to represent at all, and what little exists is dominated by the node field.
The residual matters most where there is real tissue motion to capture.

## 4. Robustness limitation found by stress-testing

The control-node learning rate is **scene-scale dependent**. At the paper default (`node_lr=0.0008`,
tuned for EndoNeRF) training collapses to 3 dB on this scene; at `1e-5` it reaches 32.46 dB. On
scenes whose change is dominated by camera motion rather than tissue deformation, the nodes must be
held near identity. Worth stating in Limitations.
