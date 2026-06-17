"""Generate a minimal but valid EndoNeRF-format scene for smoke-testing the
training pipeline on the cluster. NOT real data — only used to verify that the
data loader -> CUDA rasterizer -> optimizer -> checkpoint path runs end-to-end.

Layout produced (binocular mode):
    data/endonerf/synth/
        poses_bounds.npy      (N, 17)
        images/*.png          RGB 512x640
        depth/*.png           16-bit depth, non-zero in valid region
        masks/*.png           255 = tool (excluded), 0 = tissue (kept)
"""
import os
import numpy as np
from PIL import Image

OUT = "data/endonerf/synth"
N = 12               # frames
H, W = 512, 640
FOCAL = 560.0
NEAR, FAR = 50.0, 300.0

os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
os.makedirs(os.path.join(OUT, "depth"), exist_ok=True)
os.makedirs(os.path.join(OUT, "masks"), exist_ok=True)

rng = np.random.default_rng(0)

# pixel grid
jj, ii = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
cx, cy = W / 2.0, H / 2.0

# a central elliptical tissue region is "valid"; the rest emulates surgical tool
ellipse = (((ii - cx) / (0.42 * W)) ** 2 + ((jj - cy) / (0.42 * H)) ** 2) <= 1.0

poses = []
for k in range(N):
    t = k / N
    # ---- depth: smooth bowl ~150mm with a moving bump, only inside ellipse ----
    base = 150.0 + 40.0 * (((ii - cx) / W) ** 2 + ((jj - cy) / H) ** 2)
    bump = 25.0 * np.exp(-(((ii - cx - 80 * np.sin(2 * np.pi * t)) ** 2 +
                            (jj - cy) ** 2) / (2 * (60.0 ** 2))))
    depth = base + bump
    depth_png = np.where(ellipse, depth, 0.0).astype(np.uint16)
    Image.fromarray(depth_png, mode="I;16").save(os.path.join(OUT, "depth", f"{k:04d}.png"))

    # ---- mask: 0 over tissue (kept), 255 over tool (excluded) ----
    mask = np.where(ellipse, 0, 255).astype(np.uint8)
    Image.fromarray(mask, mode="L").save(os.path.join(OUT, "masks", f"{k:04d}.png"))

    # ---- color: smooth reddish tissue texture + slight per-frame motion ----
    r = (0.6 + 0.25 * np.sin(0.02 * ii + 3 * t)) * 255
    g = (0.3 + 0.15 * np.cos(0.02 * jj + 3 * t)) * 255
    b = (0.3 + 0.10 * np.sin(0.015 * (ii + jj))) * 255
    rgb = np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)
    rgb[~ellipse] = 30  # dark tool region
    Image.fromarray(rgb, mode="RGB").save(os.path.join(OUT, "images", f"{k:04d}.png"))

    # ---- pose: small orbit so views differ slightly ----
    ang = 0.05 * np.sin(2 * np.pi * t)
    R = np.array([[np.cos(ang), 0, np.sin(ang)],
                  [0, 1, 0],
                  [-np.sin(ang), 0, np.cos(ang)]])
    T = np.array([0.0, 0.0, 0.0])
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = T
    pose_3x5 = np.zeros((3, 5))
    pose_3x5[:3, :4] = c2w[:3, :4]
    pose_3x5[:, 4] = [H, W, FOCAL]
    row = np.concatenate([pose_3x5.reshape(-1), [NEAR, FAR]])
    poses.append(row)

poses = np.stack(poses).astype(np.float64)
np.save(os.path.join(OUT, "poses_bounds.npy"), poses)
print(f"wrote {N} frames to {OUT}; poses_bounds.npy shape {poses.shape}")
