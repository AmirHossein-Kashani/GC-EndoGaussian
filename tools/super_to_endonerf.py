"""Convert a SuPer (Semantic-SuPer v2) trial into EndoNeRF on-disk format so it reuses the existing
EndoGaussian binocular pipeline. Produces data/endonerf/super_<trial>/{images,depth,masks}/*.png +
poses_bounds.npy (static endoscope, centered intrinsics).

Decisions (documented for the tracking metric in eval_tracking.py):
- SuPer frames are 640x480; EndoNeRF hardcodes 640x512, so we PAD 16 px top + 16 px bottom (no
  distortion). The original image center (320,240) lands at the padded center (320,256), so the
  centered-intrinsics assumption stays self-consistent. GT track v-coords must be offset by +16.
- Depth is computed from stereo (cv2 SGBM): depth_mm = focal * baseline / disparity. Padded region
  and invalid disparity -> depth 0 (ignored by the loader's percentile clip + back-projection).
- Masks follow the EndoNeRF convention (tool=255 -> loader inverts to tissue=1). Tool = seg class 2.
- poses_bounds: identity static camera; hwf=[512,640,focal]; bounds from depth percentiles.
"""
import os, glob, sys
import numpy as np
import cv2
from PIL import Image

TRIAL = sys.argv[1] if len(sys.argv) > 1 else "data/super/v2_data/trial_3"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/endonerf/super_trial3"
FOCAL = 768.98551924          # superv2 intrinsics at 640x480
BASELINE_MM = 5.306           # |T[0]| from camera_matrices.yaml
PAD_TOP, PAD_BOT = 16, 16     # 480 -> 512
W, H = 640, 512
TOOL_CLASS = 2

for s in ("images", "depth", "masks"):
    os.makedirs(os.path.join(OUT, s), exist_ok=True)

lefts = sorted(glob.glob(os.path.join(TRIAL, "rgb", "*-left.png")))
N = len(lefts)
assert N > 0, f"no left frames in {TRIAL}/rgb"

sgbm = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=96, blockSize=7,
    P1=8 * 3 * 7 ** 2, P2=32 * 3 * 7 ** 2, disp12MaxDiff=1,
    uniquenessRatio=10, speckleWindowSize=100, speckleRange=2)

def pad(img, val=0):
    if img.ndim == 2:
        return np.pad(img, ((PAD_TOP, PAD_BOT), (0, 0)), constant_values=val)
    return np.pad(img, ((PAD_TOP, PAD_BOT), (0, 0), (0, 0)), constant_values=val)

medians = []
for i, lf in enumerate(lefts):
    rf = lf.replace("-left", "-right")
    L = np.asarray(Image.open(lf).convert("RGB"))
    Rr = np.asarray(Image.open(rf).convert("RGB"))
    Image.fromarray(pad(L)).save(os.path.join(OUT, "images", f"{i:05d}.png"))

    disp = sgbm.compute(cv2.cvtColor(L, cv2.COLOR_RGB2GRAY),
                        cv2.cvtColor(Rr, cv2.COLOR_RGB2GRAY)).astype(np.float32) / 16.0
    depth = np.zeros_like(disp)
    valid = disp > 0.5
    depth[valid] = FOCAL * BASELINE_MM / disp[valid]            # mm
    depth[(depth > 300) | (depth < 5)] = 0
    if (depth > 0).any():
        medians.append(float(np.median(depth[depth > 0])))
    Image.fromarray(pad(depth.astype(np.uint16)).astype(np.uint16)).save(
        os.path.join(OUT, "depth", f"{i:05d}.png"))

    seg = np.asarray(Image.open(os.path.join(TRIAL, "seg", "png_masks", os.path.basename(lf))))
    tool = (seg == TOOL_CLASS).astype(np.uint8) * 255
    Image.fromarray(pad(tool, val=255)).save(os.path.join(OUT, "masks", f"{i:05d}.png"))  # pad=tool

med = float(np.median(medians)) if medians else 100.0
near = max(5.0, np.percentile(medians, 5) * 0.5)
far = np.percentile(medians, 95) * 1.5
pose = np.zeros((3, 5), np.float32)
pose[0, 0] = pose[1, 1] = pose[2, 2] = 1.0                       # R=I, t=0 (static endoscope)
pose[:, 4] = [H, W, FOCAL]
row = np.concatenate([pose.reshape(-1), [near, far]])           # 17
np.save(os.path.join(OUT, "poses_bounds.npy"), np.tile(row, (N, 1)))

print(f"[super->endonerf] wrote {N} frames to {OUT}")
print(f"  depth median ~{med:.1f} mm  | near/far {near:.1f}/{far:.1f}")
print(f"  -> set camera_extent ~= {med:.0f} in the super config; GT-track v offset = +{PAD_TOP}")
