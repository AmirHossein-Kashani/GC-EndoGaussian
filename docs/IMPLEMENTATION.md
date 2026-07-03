# GC-EndoGaussian — Implementation & Experiments Report

A reproducibility-focused companion to the paper ([paper.md](paper.md)). It documents the **exact
method internals, configuration, data pipeline, training/eval commands, and results**, so a reader can
re-run every number and figure. Tables and figures are shared with the paper; commands and file paths are
the live tree.

- **Overview:** [RESEARCH_OVERVIEW.md](RESEARCH_OVERVIEW.md) — high-level narrative and innovation.
- **Paper:** [paper.md](paper.md) — the polished write-up.
- **This file:** the "how it actually works and how to reproduce it" reference.

---

## 1. Environment

| | |
|---|---|
| Python / PyTorch / CUDA | **3.12 · torch 2.5.1 · CUDA 12.1** (12.6 toolchain on the cluster) |
| GPU | single **NVIDIA H100** (trains in a few minutes per scene) |
| CUDA submodules | `depth-diff-gaussian-rasterization` (depth-aware fork), `simple-knn` — rebuilt against torch 2.5 |
| Cluster | SLURM GPU cluster, modules `gcc/12.3 cuda/12.6 python/3.12.4 opencv/4.11.0` |

Setup is in [CLAUDE.md](../CLAUDE.md) (§ "Environment setup"). The venv lives in `.venv/` (git-ignored);
login nodes have internet but no GPU, compute nodes have `opencv` (cv2) but no internet.

---

## 2. Code map

| Path | Role |
|---|---|
| [train.py](../train.py) | two-stage training (coarse→fine), loss, densification, **gradient clipping**, `--seed` |
| [render.py](../render.py) | renders train/test/video sets + depth; `--reconstruct` exports point clouds |
| [metrics.py](../metrics.py) | PSNR / SSIM / LPIPS / depth-RMSE / FLIP on rendered outputs |
| [scene/gaussian_model.py](../scene/gaussian_model.py) | per-Gaussian state + `_deformation = deform_network(args)` |
| [scene/hexplane.py](../scene/hexplane.py) | HexPlane / k-planes 4D feature grid (base deformation field) |
| **[scene/node_deformation.py](../scene/node_deformation.py)** | **the control-node graph** (this work): seeding, binding, GNN, SE(3), LBS, edit handle |
| [scene/dataset_readers.py](../scene/dataset_readers.py) | scene loaders (EndoNeRF / SCARED / Hamlyn / COLMAP) |
| [tools/super_to_endonerf.py](../tools/super_to_endonerf.py) | **SuPer → EndoNeRF format** converter (stereo depth, tool masks, poses) |
| **[eval_tracking.py](../eval_tracking.py)** | tracking fidelity (reprojection error + bootstrap CI + drift) |
| **[eval_control.py](../eval_control.py)** | **control-from-tracks** controllability metric + baselines + CV + figure dump |
| [eval_paired.py](../eval_paired.py) | paired Wilcoxon + bootstrap CI of the median difference |
| `arguments/endonerf/*.py` | per-scene config dicts (merged into the argparse Namespace) |

---

## 3. Method — implementation details

The base reconstruction is unchanged EndoGaussian: ~30k canonical 3D Gaussians warped to each timestamp by
a **HexPlane** (k-planes over `x,y,z,t`) feeding small MLP heads that emit `dx, ds, dr, do`. The control
graph is added on top and lives entirely in [scene/node_deformation.py](../scene/node_deformation.py).

### 3.1 Node seeding — motion-weighted farthest-point sampling
`seed_nodes(xyz, motion_weight)` selects `M` nodes via `weighted_fps` (FPS guarantees spatial coverage; the
per-point weight is the **accumulated deformation magnitude** tracked by the base field, so nodes
concentrate where motion is complex). A node–node KNN graph of degree `node_knn` is built with
`chunked_knn`. Node identity is encoded by **position only** (no per-node free parameters) → re-seeding
never changes the learnable parameter set.

### 3.2 Gaussian→node binding
`compute_binding(xyz)`: each Gaussian is soft-bound to its `gauss_knn_K` nearest nodes with a
distance-softmax `w = softmax(-d² / σ²)` (σ from the per-Gaussian nearest-node distance). Bindings are
per-Gaussian and rebuilt when densification changes the Gaussian set.

### 3.3 Message-passing GNN → per-node SE(3)
`_run_gnn(t)`: node inputs `h⁰ = MLP([γ(n), γ(t)])` with positional encoding `γ` (`node_pe` freqs). Then
`gnn_layers` EdgeConv-style residual layers
`h^{l+1} = h^l + φ(h^l, AGG_{n∈N(m)} ψ(h^l, hⁿ, γ(nⁿ−n)))`. A head emits **3 translation + 6D rotation**
per node (`rotation_6d_to_matrix`), **initialised to identity** (`init_identity_head`: zeroed weights,
bias `[1,0,0,0,1,0]`) so training starts as an exact no-op. `gnn_layers=0` reduces to an independent
per-node MLP (the **SC-GS-style ablation**).

### 3.4 Linear blend skinning + the edit handle
`forward(...)` applies LBS over each Gaussian's `K` bound nodes:
`p_i = Σ_k w_ik [ R_k (x_i − n_k) + n_k + t_k ]`; rotation is a weighted node-quaternion blend composed
with the canonical rotation. A per-node **`edit_translation`** buffer (zero in training) is added to each
node's translation in `_run_gnn` — setting it at inference drags those nodes and the bound Gaussians
follow. Non-finite outputs fall back to the canonical value so degeneracies never reach the rasterizer.

### 3.5 The `control_only` eval guard (decontamination)
For the controllability metric we must predict **purely from the control input**, not any learned field.
Two eval-only guards, both gated on `control_only`:
1. `_run_gnn` ([node_deformation.py](../scene/node_deformation.py)) returns `R = I`, `trans = edit_translation`
   — freezes the graph's learned node motion.
2. `forward_dynamic` ([deformation.py](../scene/deformation.py)) **also skips the hybrid per-Gaussian
   residual** `pos_deform` — without this, the residual leaks learned reconstruction into the "control"
   prediction and inflates the score by ~4 px (see §9.3). This second guard is the decontamination.

### 3.6 The *match* recipe (what makes it cost-free)
Config [arguments/endonerf/pulling_graph_match_3k.py](../arguments/endonerf/pulling_graph_match_3k.py):

- `node_translation_only=True` — graph drives **position only**; rotation/scale/opacity from the full MLP
  (avoids a lossy quaternion-LBS blend).
- `node_hybrid=True` — small additive **per-Gaussian MLP residual** recovers high-frequency detail.
- `lambda_arap = lambda_isometric = lambda_node_temporal = 0` — **all coherence regularizers off** (real
  tissue is non-rigid; rigidity priors bias position away from the photometric optimum).
- `node_refresh_interval = 999999` — **nodes frozen** after the initial seed (no mid-training re-seed
  disruption).

---

## 4. Configuration reference

Config `.py` files are plain dicts keyed by `ModelParams / OptimizationParams / ModelHiddenParams`, merged
over the class defaults in [arguments/__init__.py](../arguments/__init__.py) via `merge_hparams`.

**Control-graph knobs** (default → *match* value):

| Knob | Default | *match* | Meaning |
|---|---|---|---|
| `use_node_graph` | `False` | **`True`** | enable the control graph |
| `num_nodes` (M) | 1024 | **2048** | control nodes |
| `node_knn` | 8 | 8 | node–node graph degree |
| `gauss_knn_K` | 4 | 4 | nodes each Gaussian binds to |
| `gnn_layers` | 2 | 2 | message-passing layers (**0 = SC-GS-style ablation**) |
| `gnn_width` | 64 | 64 | GNN hidden width |
| `node_pe` | 4 | 4 | positional-encoding freqs (xyz / edges / time) |
| `node_hybrid` | `False` | **`True`** | graph low-freq + per-Gaussian residual |
| `node_translation_only` | `False` | **`True`** | position-only control |
| `lambda_arap` | 0.01 | **0.0** | ARAP coherence prior |
| `lambda_isometric` | 0.0 | 0.0 | as-isometric edge-length prior |
| `node_refresh_interval` | 1000 | **999999** | re-seed cadence (frozen) |
| `node_lr_init / final` | 8e-4 / 8e-5 | 8e-4 / 8e-5 | GNN learning rate |
| `grad_clip` | 10.0 | 10.0 | deformation grad-norm clip (**0 = off**, for the stability study) |

Shared base: `coarse_iterations=1000`, `iterations=3000` (fine), HexPlane
`resolution=[64,64,64,100]`, `multires=[1,2,4,8]`, `net_width=32`, `defor_depth=0`.

**SC-GS learned baseline** ([pulling_graph_scgs.py](../arguments/endonerf/pulling_graph_scgs.py)): a faithful
SC-GS proxy via `gnn_layers=0` (independent control points), `node_translation_only`/`node_hybrid` at their
`False` defaults (full SE(3), points-are-deformation), `lambda_arap=0.01` + `lambda_node_temporal=0.001`
(ARAP coherence), `node_refresh_interval=1000` (adaptive nodes), `num_nodes=2048` (budget parity). All SC-GS
ingredients are pre-existing, functional knobs — no new code.

---

## 5. Data pipeline

### 5.1 EndoNeRF (`pulling`, `cutting`)
Real clips live only on the authors' Google Drive; fetch `pulling_soft_tissues` (63 frames, binocular)
into the EndoGaussian layout with pinned file IDs:
```
bash tools/download_endonerf_pulling.bash   # -> data/endonerf/pulling/{images,depth,masks}/*.png + poses_bounds.npy
```

### 5.2 SuPer → EndoNeRF ([tools/super_to_endonerf.py](../tools/super_to_endonerf.py))
The controllability dataset. SuPer provides stereo da Vinci video of tissue manipulation with
hand-annotated tracked points. The converter, per trial:

- **Resolution:** SuPer is 640×480; EndoNeRF expects 640×512 → **pad 16 px top + 16 px bottom** (the
  GT-track `v` coordinate therefore needs `+16`, passed as `--pad_v 16` to the eval scripts).
- **Depth:** stereo `cv2.StereoSGBM` (`numDisparities=96, blockSize=7`), `depth_mm = FOCAL·BASELINE/disp`
  with `FOCAL=768.98551924`, `BASELINE_MM=5.306` (superv2 intrinsics).
- **Masks:** tool = segmentation class 2 → EndoNeRF convention (tool=255, loader inverts to tissue=1).
- **Poses:** static endoscope — `R=I, t=0`, `hwf=[512,640,FOCAL]`, bounds from depth percentiles.

```
python tools/super_to_endonerf.py data/super/v2_data/trial_3 data/endonerf/super_trial3
```
Trials 3/4/8/9 (151 frames each; 32/26/36/51 annotated points) are the four used in the multi-trial study.

---

## 6. Training

Two-stage (`scene_reconstruction` called twice): **coarse** (1000 iters, static geometry, viewpoint idx 0)
then **fine** (3000 iters, random viewpoints, deformation + control graph active).

**Loss** ([train.py](../train.py) `L136+`):
`L = L1(img, tool-masked) + depth_loss + 0.03·(TV_img + TV_depth)` (+ `lambda_dssim·(1−SSIM)` if enabled;
+ `compute_regulation` in fine stage only when `time_smoothness_weight≠0`, which *match* sets to 0).
Depth loss: **binocular** = masked (inverse-)depth L1; **monocular** = `0.001·(1 − Pearson)` on masked pixels.

**Gradient clipping** ([train.py:298](../train.py#L298)): clips the grad-norm of the `deformation`, `grid`,
and `node_gnn` parameter groups to `max_norm=grad_clip` (default 10; set `grad_clip=0` to disable — used
only for the stability study).

```
# vanilla EndoGaussian
python train.py -s data/endonerf/pulling --expname endonerf/pulling --configs arguments/endonerf/pulling.py --port 6017
# GC-EndoGaussian (match recipe)
python train.py -s data/endonerf/super_trial3 --expname endonerf/super_match \
    --configs arguments/endonerf/pulling_graph_match_3k.py --save_iterations 1000 3000 --port 6017
```
Outputs land in `output/<expname>/` (`cfg_args`, `point_cloud/iteration_<n>/`, per-set render dumps).

---

## 7. Rendering

`render.py` reloads the persisted `cfg_args` from `--model_path`, so `--configs` must match the training
config. `--skip_train/--skip_test/--skip_video` suppress sets.
```
python render.py --model_path output/endonerf/super_match \
    --configs arguments/endonerf/pulling_graph_match_3k.py --iteration 3000 --skip_train --skip_test
```
Produces `video/ours_3000/{renders,gt,depth,gt_depth}/*.png` + `ours_video.mp4`, `gt_video.mp4`.

---

## 8. Evaluation

### 8.1 Reconstruction — `metrics.py`
`python metrics.py --model_path output/endonerf/super_match` → PSNR/SSIM/LPIPS/depth-RMSE on the test set.

### 8.2 Tracking fidelity — [eval_tracking.py](../eval_tracking.py)
Reprojects the deformed Gaussians and measures pixel error vs the GT tracks; reports **median RPE + 95%
bootstrap CI**, frame-0 error (projection sanity ≈2 px), and drift across the sequence.
```
python eval_tracking.py --model_path output/endonerf/super_match \
    --configs arguments/endonerf/pulling_graph_match_3k.py \
    --tracks data/super/v2_data/trial_3/rgb/trial_3_l_pts.npy --pad_v 16 --iteration 3000
```

### 8.3 Control-from-tracks (main metric) — [eval_control.py](../eval_control.py)
**Protocol** (per frame transition 0→f): split the GT points into `K` **handles** + held-out **targets**
(handles chosen by FPS). Each handle's observed 3D motion (back-projected via the static-camera geometry)
drives the `splat_k` control nodes nearest it; **all learned motion is frozen** (`control_only`: node SE(3)
*and* the hybrid residual — §3.5) so the prediction is a pure function of the control. The graph propagates
the sparse control through LBS; we **predict the held-out points** and score reprojection error against GT.
- **Baselines** through the same harness: `rigid`, `nearest` (nearest-handle copy), `tps`, and the retrained
  **SC-GS-style** model (`super_t*_scgs`, [pulling_graph_scgs.py](../arguments/endonerf/pulling_graph_scgs.py)).
- **Sweep + CV:** `K ∈ {4,8,16}`, **4-fold** leave-groups-out (rotate which points are handles).
- **Decontamination is load-bearing:** the residual guard (§3.5) is what makes this measure control, not
  reconstruction. Without it the number is ~4 px too optimistic (§9.3).
```
python eval_control.py --model_path output/endonerf/super_match \
    --configs arguments/endonerf/pulling_graph_match_3k.py \
    --tracks data/super/v2_data/trial_3/rgb/trial_3_l_pts.npy --pad_v 16 --iteration 3000
```
Writes `control_results.json`. `--dump_frame -2 --dump_K 8` dumps per-point predictions across all frames
into `control_viz.json` (used to build Figure 4). *Scope: the control input is GT tissue motion, i.e.
deformation-prediction-under-sparse-observation — not closed-loop robot control.*

### 8.4 Paired significance — [eval_paired.py](../eval_paired.py)
`python eval_paired.py A_per_point.npy B_per_point.npy` → median diff, bootstrap 95% CI, Wilcoxon `p`.

---

## 9. Experiments & results

All numbers below are the archived runs in `results_archive/` and match the paper tables.

### 9.1 Reconstruction is preserved (iteration-matched, 6000 fine iters)
| Dataset | Method | PSNR↑ | SSIM↑ | LPIPS↓ | Depth-RMSE↓ | ΔPSNR |
|---|---|---|---|---|---|---|
| pulling | vanilla | 37.32 | 0.9578 | 0.0509 | 2.646 | — |
| pulling | **ours (match)** | 37.17 | 0.9567 | 0.0533 | 2.793 | **−0.15** |
| cutting | vanilla | 39.42 | 0.9696 | 0.0322 | 1.358 | — |
| cutting | **ours (match)** | 39.29 | 0.9689 | 0.0339 | 1.384 | **−0.13** |

At the base 3000-iter budget the gap is ~0.27 dB and the extra iters recover only ~0.1 dB → capability at
**no extra training time**. See Figure 3 ([figures/recon_pulling_triptych.png](figures/recon_pulling_triptych.png)).

### 9.2 Efficiency (pulling)
| | EndoGaussian | Ours (match) |
|---|---|---|
| Render speed | 285 FPS | **205 FPS** |
| Deformation params | 85.29 M | 85.35 M (**+0.07%**) |
| Training time | baseline | **unchanged** |

### 9.3 Controllability — control-from-tracks (decontaminated; a negative finding)

**⚠️ Decontamination is essential.** A naïve control-from-tracks metric leaves the hybrid per-Gaussian
residual active, which leaks *learned reconstruction* into the "control" prediction. Freezing it
([deformation.py](../scene/deformation.py), `control_only` also skips `pos_deform`) is the honest metric.
Effect on our own *match* model (cross-trial mean px):

| K | Naïve (residual active) | **Decontaminated (control only)** | change |
|---|---|---|---|
| 4 | 2.86 | **6.82** | +3.96 |
| 8 | 2.77 | **6.80** | +4.03 |
| 16 | 2.92 | **8.09** | +5.17 |

**Decontaminated comparison (cross-trial mean over trials 3/4/8/9, px; lower is better):**
| K | Ours (control only) | SC-GS (learned) | Rigid | **Nearest** | TPS |
|---|---|---|---|---|---|
| 4 | 6.82 | 6.71 | 6.89 | **5.69** | 11.61\* |
| 8 | 6.80 | 6.74 | 6.03 | **4.73** | 5.87 |
| 16 | 8.09 | 8.06 | 6.24 | 3.97 | **3.45** |

<sub>\*TPS undefined at K=4 on 2/4 trials (degenerate with 4 control points).</sub>

**Reading (honest):** under the decontaminated metric, learned sparse control (ours ≈ SC-GS) does **not**
beat classical interpolation — nearest-handle wins at every K, and the learned methods are *worst* at K=16.
The GNN gives no control advantage because `control_only` bypasses the message passing (the edit is a
post-hoc node translation). The earlier "~2× over classical" was the residual leak, not a real property of
the control. See Figure 4 (naïve vs decontaminated, same frame) and Figure 5 (decontaminated curve). The
naïve numbers are preserved per-model in `control_results_residual.json`.

### 9.4 Tracking fidelity — statistically equivalent to baseline
Median RPE **3.30 (ours) vs 3.47 (vanilla)** px, 95% CIs [3.14, 3.46] vs [3.34, 3.59], paired Wilcoxon
**p=0.73**. Across four trials: mean median **2.76 (ours) vs 2.80 (vanilla)** px; ours lower on 3/4.

| Trial | Vanilla | Ours |
|---|---|---|
| 3 | 3.47 | 3.30 |
| 4 | 2.80 | 2.72 |
| 8 | 3.29 | 3.25 |
| 9 | 1.62 | 1.79 |

### 9.5 Residual isolation — the residual-matched SC-GS ablation (key attribution)
Which part of the recipe keeps editing reconstruction-neutral? Train SC-GS-style control (`gnn_layers=0`,
full SE(3), ARAP) **with and without** the per-Gaussian residual ([pulling_graph_scgs.py](../arguments/endonerf/pulling_graph_scgs.py)
vs [pulling_graph_scgs_hybrid.py](../arguments/endonerf/pulling_graph_scgs_hybrid.py)), 3000 iters
([run_scgs_residual.bash](../run_scgs_residual.bash)):

| Method | PSNR↑ (pulling) | SSIM↑ | LPIPS↓ | Track RPE↓ (SuPer t3) |
|---|---|---|---|---|
| vanilla (no editing) | 37.27 | 0.9578 | 0.0609 | 3.47 |
| SC-GS-style, **no** residual | 36.80 | 0.9505 | 0.0885 | 7.02 |
| SC-GS-style **+ residual** | **37.29** | 0.9570 | 0.0649 | 3.41 |
| Ours (match) | 37.00 | 0.9559 | 0.0638 | 3.30 |

**Finding:** the residual moves SC-GS-style from 36.80→37.29 dB and 7.02→3.41 px — on par with our match and
with vanilla. The **per-Gaussian residual**, not the GNN or the specific integration choices, is what
preserves fidelity; it transfers to either control architecture. So the honest claim is a *residual-centered
recipe*, not a superiority over SC-GS. (Earlier drafts' "~0.7 dB / ~2× advantage over SC-GS" reflected a
residual-free baseline — corrected here.)

### 9.6 Integration modes & negative results
| Method (pulling, 3000 iters) | PSNR↑ | SSIM↑ | LPIPS↓ | Depth-RMSE↓ |
|---|---|---|---|---|
| vanilla | 37.27 | 0.9578 | 0.0609 | 2.906 |
| graph, replace (GNN) | 36.68 | 0.9488 | 0.0946 | 3.001 |
| graph, replace (no GNN) | 36.50 | 0.9476 | 0.0954 | 3.037 |
| graph, hybrid | 36.88 | 0.9537 | 0.0760 | 3.037 |

The *match* recipe (§3.6) closes almost all of this gap. **Where the graph does *not* help:**
occlusion-holdout (26.00 vs 26.17 PSNR), optical-flow supervision (no gain), explicit cut-modelling
(11.95 vs 11.88 at the cut, still below the 12.01 continuous field). A continuous HexPlane is already
smooth/coherent, so the graph adds *constraint, not information* on reconstruction, and (§9.3) not a
controllability advantage either — its value is a cheap **editable handle at reconstruction parity**.

---

## 10. Figures & videos

| Artifact | File | Made by |
|---|---|---|
| Fig. 2 drag-to-edit | `figures/edit_{before,after,diff}.png` | gentle edit (`after_0`) + magnitude heatmap |
| Fig. 3 pulling reconstruction | `figures/recon_pulling_triptych.png` | GT\|Ours\|error from render dumps |
| Fig. 4 decontamination (qual) | `figures/control_from_tracks_qual.png` | naïve (residual-active) vs decontaminated (control-only) at trial 3 f57; `control_viz{,_residual}.json` |
| Fig. 5 decontaminated curve | `figures/controllability_curve.png` | 4-trial mean; learned (ours/SC-GS) vs classical + the naïve leak (dashed) |
| Fig. 6 SuPer reconstruction | `figures/recon_super_t3_triptych.png` | GT\|Ours\|error |
| GT-vs-Ours videos | `figures/recon_super_trial{3,4,8,9}_gt_vs_ours.mp4`, `recon_pulling_gt_vs_ours.mp4` | ffmpeg hstack of `ours_video.mp4` \| `gt_video.mp4` |

---

## 11. End-to-end reproduction (one trial)

```bash
# 0. data
python tools/super_to_endonerf.py data/super/v2_data/trial_3 data/endonerf/super_trial3
# 1. train vanilla + graph
python train.py -s data/endonerf/super_trial3 --expname endonerf/super_vanilla \
    --configs arguments/endonerf/pulling.py --save_iterations 1000 3000
python train.py -s data/endonerf/super_trial3 --expname endonerf/super_match \
    --configs arguments/endonerf/pulling_graph_match_3k.py --save_iterations 1000 3000
# 2. evaluate
T=data/super/v2_data/trial_3/rgb/trial_3_l_pts.npy
python eval_tracking.py --model_path output/endonerf/super_vanilla --configs arguments/endonerf/pulling.py --tracks $T --pad_v 16 --iteration 3000
python eval_tracking.py --model_path output/endonerf/super_match   --configs arguments/endonerf/pulling_graph_match_3k.py --tracks $T --pad_v 16 --iteration 3000
python eval_control.py  --model_path output/endonerf/super_match   --configs arguments/endonerf/pulling_graph_match_3k.py --tracks $T --pad_v 16 --iteration 3000
# 3. render + demo video
python render.py --model_path output/endonerf/super_match --configs arguments/endonerf/pulling_graph_match_3k.py --iteration 3000 --skip_train --skip_test
```
The multi-trial study is scripted in [run_gc_multitrial.bash](../run_gc_multitrial.bash) (trials 4/8/9); the
**SC-GS baseline** in [run_gc_scgs.bash](../run_gc_scgs.bash); the **decontaminated re-eval** in
[run_control_fair.bash](../run_control_fair.bash); rendering in
[run_super_render.bash](../run_super_render.bash); the Fig. 4 dumps in
[run_control_viz.bash](../run_control_viz.bash). SLURM jobs use a single H100.

> **Note on decontamination.** `control_results.json` holds the **decontaminated** (control-only) numbers;
> the naïve residual-active numbers are preserved as `control_results_residual.json` per model. The guard
> lives in [scene/deformation.py](../scene/deformation.py) (§3.5).

---

## 12. Artifacts

`results_archive/endonerf/` holds the archived JSONs per run (`results.json`, `tracking_results.json`,
`control_results.json`, `cfg_args`) for `super_match`, `super_match_nognn`, `super_vanilla`,
`super_t{4,8,9}_{match,vanilla}`, and the SC-GS baseline `super_t{3,4,8,9}_scgs`. `output/` and `data/` are
git-ignored and regenerated by the commands above.
