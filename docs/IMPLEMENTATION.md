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

### 3.5 The `control_only` eval guard
For the controllability metric we must predict **purely from the control input**, not the learned field.
`_run_gnn` checks `self.control_only`: when set, it returns `R = I` and `trans = edit_translation` (skips
the learned SE(3)). This is the only behavioural change the metric introduces, and it is eval-only.

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
drives the `splat_k` control nodes nearest it; the learned field is **frozen** (`control_only`) so the
prediction is a pure function of the control. The graph propagates the sparse control through LBS; we
**predict the held-out points** and score reprojection error against their GT tracks.
- **Baselines** through the same harness: `rigid` (single mean translation), `nearest` (nearest-handle
  copy), `tps` (thin-plate-spline interpolation of handle displacements).
- **Sweep + CV:** `K ∈ {4,8,16}`, **4-fold** leave-groups-out (rotate which points are handles).
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

### 9.3 Controllability — control-from-tracks (main result)
**Per-trial (trial 3), median ± std over folds (px):**
| K | **Ours (graph)** | Rigid | Nearest | TPS |
|---|---|---|---|---|
| 4 | **3.27 ± 0.14** | 7.36 | 6.68 | 12.34 |
| 8 | **3.34 ± 0.20** | 7.01 | 5.81 | 7.39 |
| 16 | **2.95** | 7.03 | 3.86 | 3.75 |

**GNN ablation (trial 3):** removing message passing (`gnn_layers=0`) degrades at every K — K4 3.27→3.54,
K8 3.34→3.45, K16 **2.95→4.27**.

**Cross-trial mean (trials 3/4/8/9), median held-out error (px):**
| K | **Ours (graph)** | Rigid | Nearest | TPS |
|---|---|---|---|---|
| 4 | **2.86** | 6.89 | 5.69 | 11.61\* |
| 8 | **2.77** | 6.03 | 4.73 | 5.87 |
| 16 | **2.92** | 6.24 | 3.97 | 3.45 |

<sub>\*TPS undefined at K=4 on 2/4 trials (degenerate with 4 control points).</sub>

**Reading:** the controller is nearly flat in K (2.77–2.92 px) while classical baselines degrade sharply as
handles thin out → advantage largest in the clinically realistic sparse regime (~2× over nearest at K=4).
As handles densify, the gap narrows; on the densest trial (51 pts) nearest-copy edges it. See Figure 4
([figures/control_from_tracks_qual.png](figures/control_from_tracks_qual.png)) and Figure 5
([figures/controllability_curve.png](figures/controllability_curve.png)).

### 9.4 Tracking fidelity — statistically equivalent to baseline
Median RPE **3.30 (ours) vs 3.47 (vanilla)** px, 95% CIs [3.14, 3.46] vs [3.34, 3.59], paired Wilcoxon
**p=0.73**. Across four trials: mean median **2.76 (ours) vs 2.80 (vanilla)** px; ours lower on 3/4.

| Trial | Vanilla | Ours |
|---|---|---|
| 3 | 3.47 | 3.30 |
| 4 | 2.80 | 2.72 |
| 8 | 3.29 | 3.25 |
| 9 | 1.62 | 1.79 |

### 9.5 Integration modes & negative results
| Method (pulling, 3000 iters) | PSNR↑ | SSIM↑ | LPIPS↓ | Depth-RMSE↓ |
|---|---|---|---|---|
| vanilla | 37.27 | 0.9578 | 0.0609 | 2.906 |
| graph, replace (GNN) | 36.68 | 0.9488 | 0.0946 | 3.001 |
| graph, replace (no GNN) | 36.50 | 0.9476 | 0.0954 | 3.037 |
| graph, hybrid | 36.88 | 0.9537 | 0.0760 | 3.037 |

The *match* recipe (§3.6) closes almost all of this gap. **Where the graph does *not* help:**
occlusion-holdout (26.00 vs 26.17 PSNR), optical-flow supervision (no gain), explicit cut-modelling
(11.95 vs 11.88 at the cut, still below the 12.01 continuous field). A continuous HexPlane is already
smooth/coherent, so the graph adds *constraint, not information* on reconstruction — its unique value is
controllability under sparse supervision.

---

## 10. Figures & videos

| Artifact | File | Made by |
|---|---|---|
| Fig. 2 drag-to-edit | `figures/edit_{before,after,diff}.png` | gentle edit (`after_0`) + magnitude heatmap |
| Fig. 3 pulling reconstruction | `figures/recon_pulling_triptych.png` | GT\|Ours\|error from render dumps |
| Fig. 4 control-from-tracks (qual) | `figures/control_from_tracks_qual.png` | `eval_control --dump_frame -2`, representative frame (trial 3, f57) |
| Fig. 5 controllability curve | `figures/controllability_curve.png` | 4-trial mean of `control_results.json` |
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
The multi-trial study is scripted in [run_gc_multitrial.bash](../run_gc_multitrial.bash) (trials 4/8/9);
rendering in [run_super_render.bash](../run_super_render.bash); the Fig. 4 dump in
[run_control_viz.bash](../run_control_viz.bash). SLURM jobs use a single H100.

---

## 12. Artifacts

`results_archive/endonerf/` holds the archived JSONs per run (`results.json`, `tracking_results.json`,
`control_results.json`, `cfg_args`) for `super_match`, `super_match_nognn`, `super_vanilla`, and
`super_t{4,8,9}_{match,vanilla}`. `output/` and `data/` are git-ignored and regenerated by the commands
above.
