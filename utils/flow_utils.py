"""Offline optical-flow supervision helpers (cv2 Farneback — no network, Compute-Canada safe).

Builds, for each consecutive training-frame pair (i, i+1), a grid_sample grid that warps the
render of frame i into frame i+1's view using the observed flow. A flow-consistency loss then
asks: the model's rendered motion between i and i+1 should match the real (GT) motion.

Because GT renders are already flow-consistent, this loss is ~0 where the model's motion is
correct and only bites where it deviates — so it preferentially corrects an over-smooth /
under-constrained motion field (the graph's failure mode) while barely touching an already
accurate one. Includes a self-check: it only enables flow if warping GT_i actually reduces the
photometric error to GT_{i+1} (guards against a flipped flow convention).
"""
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def build_flow_grids(images, device, sample_check=8):
    """images: list of (3,H,W) float tensors in [0,1] (train order).

    Returns (grids, enabled, (improved, tested)):
      grids[i] : (H,W,2) normalized grid that warps render_i -> frame i+1 (None for last / failures)
      enabled  : True iff warping GT with these grids reduces GT L1 on a majority of checked pairs
    """
    import cv2
    N = len(images)
    H, W = images[0].shape[-2:]
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32), indexing="ij")
    grids = [None] * N
    improved, tested = 0, 0

    def to_gray_u8(t):
        a = (t.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)

    for i in range(N - 1):
        ga, gb = to_gray_u8(images[i]), to_gray_u8(images[i + 1])
        # backward flow (b -> a): for each pixel of frame i+1, where did it come from in frame i
        fb = cv2.calcOpticalFlowFarneback(gb, ga, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fx = torch.from_numpy(fb[..., 0]); fy = torch.from_numpy(fb[..., 1])
        sx, sy = xs + fx, ys + fy
        gx = 2.0 * sx / max(W - 1, 1) - 1.0
        gy = 2.0 * sy / max(H - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).float().to(device)   # (H,W,2)
        grids[i] = grid
        # self-check on a sparse subset
        if (sample_check <= 0) or (i % max(1, (N // max(sample_check, 1))) == 0):
            a = images[i].unsqueeze(0).to(device)
            b = images[i + 1].unsqueeze(0).to(device)
            warped = F.grid_sample(a, grid.unsqueeze(0), align_corners=True, padding_mode="border")
            if (warped - b).abs().mean() < (a - b).abs().mean():
                improved += 1
            tested += 1
    enabled = tested > 0 and improved / tested > 0.5
    return grids, enabled, (improved, tested)


def flow_warp(image_chw, grid_hw2):
    """Warp a (3,H,W) render by a (H,W,2) grid -> (1,3,H,W) aligned to the next frame."""
    return F.grid_sample(image_chw.unsqueeze(0), grid_hw2.unsqueeze(0),
                         align_corners=True, padding_mode="border")
