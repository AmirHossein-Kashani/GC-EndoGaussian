# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

EndoGaussian — 4D Gaussian Splatting for dynamic endoscopic scene reconstruction. Built on 3DGS / 4DGaussians / EndoNeRF. Supports EndoNeRF, SCARED, and Hamlyn datasets, in `binocular` (stereo + depth) or `monocular` modes.

## Environment setup

Requires CUDA 11.7 + PyTorch 1.13.1 (Python 3.7 conda env per README). Two CUDA submodules must be installed editable:

```
pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

`render.py` calls `os.sched_setaffinity` ([render.py:32](render.py#L32)) which is Linux-only — running the renderer on Windows requires patching that line out.

## Common commands

Training (two-stage coarse → fine, configs are merged via `mmcv.Config.fromfile` into the argparse Namespace):

```
python train.py -s data/endonerf/pulling --port 6017 --expname endonerf/pulling --configs arguments/endonerf/pulling.py
```

Rendering (uses the saved `cfg_args` in `--model_path`; `--skip_train/--skip_test/--skip_video` to suppress sets; `--reconstruct` for point cloud export):

```
python render.py --model_path output/endonerf/pulling --skip_train --skip_video --configs arguments/endonerf/pulling.py
```

Metrics (PSNR / SSIM / LPIPS / RMSE / FLIP on the rendered outputs):

```
python metrics.py --model_path output/endonerf/pulling
```

Outputs land in `./output/<expname>/` containing `cfg_args`, `point_cloud/{coarse_iteration_,iteration_}<n>/`, and per-set render dumps. `output/` and `data/` are git-ignored.

## Architecture

### Two-stage training ([train.py:238](train.py#L238))

`training()` calls `scene_reconstruction()` twice with `stage="coarse"` (uses `opt.coarse_iterations`, always picks viewpoint idx 0) then `stage="fine"` (uses `opt.iterations`, random viewpoints, deformation field active). `--no_fine` skips the fine stage and loads coarse checkpoints. SH degree ramps up every 500 iters via `oneupSHdegree()`.

### Scene dispatch ([scene/__init__.py:39](scene/__init__.py#L39))

`Scene.__init__` picks a loader by inspecting the source dir + the `extra_mark` field from the config:

- `sparse/` + `extra_mark=None` → `readColmapSceneInfo`
- `poses_bounds.npy` + `extra_mark='endonerf'` → `readEndoNeRFInfo`
- `poses_bounds.npy` + `extra_mark='hamlyn'` → `readHamlynInfo`
- `point_cloud.obj` or `left_point_cloud.obj` → `readScaredInfo`

All four are registered in `sceneLoadTypeCallbacks` ([scene/dataset_readers.py:318](scene/dataset_readers.py#L318)). The loader returns train/test/video camera lists, a point cloud, and `nerf_normalization`; for endonerf, `args.camera_extent` from the config overrides the auto-derived radius.

### Loss composition ([train.py:136](train.py#L136))

`L = L1(img, masked) + depth_loss + 0.03 * (TV(img) + TV(depth))`, plus optional dssim/lpips, plus `gaussians.compute_regulation(...)` in fine stage when `hyper.time_smoothness_weight != 0`. Depth loss differs by mode: binocular uses inverse-depth L1; monocular uses Pearson correlation on masked pixels.

### Gaussian model + 4D deformation

`GaussianModel` ([scene/gaussian_model.py](scene/gaussian_model.py)) owns the per-Gaussian state (`_xyz`, `_features_dc/rest`, `_scaling`, `_rotation`, `_opacity`) plus `_deformation = deform_network(args)`. The deformation MLP is fed by a `HexPlaneField` ([scene/hexplane.py](scene/hexplane.py)) parameterised by `ModelHiddenParams.kplanes_config` (4D grid: x,y,z,t) and `multires`. AABB is set from the loaded point cloud in `Scene.__init__` ([scene/__init__.py:66](scene/__init__.py#L66)).

Densification & pruning are driven by `opacity_threshold_*` / `densify_grad_threshold_*` knobs from `OptimizationParams`; thresholds anneal linearly between `_fine_init` and `_fine_after` over `densify_until_iter` ([train.py:213](train.py#L213)). `opacity_reset_interval` periodically resets opacities.

### Config merge ([arguments/__init__.py](arguments/__init__.py))

`ModelParams`, `PipelineParams`, `OptimizationParams`, `ModelHiddenParams` are argparse-style classes. Per-scene `.py` files under `arguments/<dataset>/` are plain dicts named after these classes (e.g. `ModelParams = dict(extra_mark='endonerf', camera_extent=10)`) — they override defaults via `utils.params_utils.merge_hparams`. `get_combined_args` additionally reloads the persisted `cfg_args` from `--model_path` so rendering/eval matches the training run.

### Rasterizer

`gaussian_renderer/__init__.py` wraps the `depth-diff-gaussian-rasterization` CUDA kernel and additionally returns rendered depth. The vanilla `diff-gaussian-rasterization` submodule is listed in `.gitmodules` but the codebase uses the depth-aware fork.

## Repo-specific notes

- `journal/` is an in-tree snapshot copy of an older variant of the project (configs include `arguments/hamlyn/seq{1..7}.py` that the top-level `arguments/` is missing). Hamlyn training therefore needs a config file copied from `journal/arguments/hamlyn/` even though the scene loader already supports `extra_mark='hamlyn'`. Treat `journal/` as reference only — do not edit it expecting changes to be picked up by `train.py`.
- `full_eval.py` is inherited from upstream 3DGS and references mipnerf360 / tanks-and-temples / deep-blending scene names; it is not wired to the endoscopic datasets used here.
- TensorBoard logging in `training_report` only emits scalar loss/timing; the image/PSNR test block is commented out ([train.py:275](train.py#L275)).
- `arguments/<scene>_mono.py` variants switch `mode='monocular'` which changes the depth-loss branch.
