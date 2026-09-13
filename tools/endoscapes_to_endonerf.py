"""Convert an Endoscapes-CVS201 clip (monocular laparoscopy) into the EndoNeRF on-disk layout so it
runs through the existing EndoGaussian *monocular* pipeline.

Produces  data/endonerf/<out>/{images,monodepth,masks}/*.png + poses_bounds.npy

Why this differs from tools/super_to_endonerf.py: SuPer/EndoNeRF are stereo, so depth comes from
cv2 SGBM for free. Endoscapes is a single moving laparoscope -- there is no stereo pair, so depth
must come from a learned monocular estimator (Depth-Anything V2). `mode='monocular'` in the loader
reads `monodepth/` and converts stored *disparity* (bright = near) into depth via `(1/d)*0.4`, so we
store the normalized disparity the network predicts, not metric depth.

Geometry decisions (all documented because they are assumptions, not measurements):
- The endoscope image is a circle vignetted into an 854x480 frame; content bbox is x=[64,765],
  y=[0,479] (measured over the clip). We crop that bbox (702x480), resize to 640x438 preserving
  aspect, and pad 37px top/bottom to the 640x512 the loader hardcodes. No stretching.
- Poses are STATIC (R=I, t=0), the same assumption the EndoNeRF/SuPer scenes use. The Endoscapes
  laparoscope actually moves; with static poses that apparent motion is absorbed by the 4D
  deformation field instead of by the camera. This is what makes the clip trainable without SfM
  (COLMAP is unreliable on textureless, specular, deforming tissue), but it means the reconstruction
  is view-dependent: geometry is only meaningful for rendering near the training viewpoint.
- Focal is unknown for this dataset; we assume a 60-degree horizontal FOV, typical for a
  laparoscope, giving f = (W/2)/tan(fov/2).
- Masks follow the EndoNeRF convention: WHITE(255) = excluded from the photometric loss. We exclude
  only the vignette, the letterbox padding, and near-black instrument shafts -- NOT the metallic
  instruments; see invalid_mask() for why color-based tool segmentation is rejected here.

  python3 tools/endoscapes_to_endonerf.py --src data/Endoscapes-CVS201/selected \
      --out data/endonerf/endoscapes_83 [--no-depth]
"""
import os, glob, argparse
import numpy as np
from PIL import Image, ImageFilter

W, H = 640, 512                 # what the loader hardcodes
CROP = (64, 0, 767, 480)        # measured content bbox (left, top, right, bottom) over all 73 frames
FOV_DEG = 60.0                  # assumed laparoscope horizontal FOV
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DISP_MIN = 0.30                 # -> depth span 0.4 .. 1.33 units (~3.3:1, realistic for endoscopy)


def letterbox(im, resample, crop=None):
    """Crop the vignette bbox, scale to fit (W,H) preserving aspect, centre-pad. No stretching."""
    im = im.crop(crop or CROP)
    w, h = im.size
    s = min(W / w, H / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    im = im.resize((nw, nh), resample)
    px, pad = (W - nw) // 2, (H - nh) // 2
    canvas = Image.new(im.mode, (W, H), 0)
    canvas.paste(im, (px, pad))
    return canvas, pad, nh


def invalid_mask(rgb, pad, nh):
    """White(255) = excluded from the photometric loss: vignette, padding, and dark instrument shafts.

    We deliberately do NOT try to segment the metallic instruments by color. Endoscapes ships no tool
    annotations, and every saturation/brightness rule that covers the graspers also swallows the pale
    fat and connective tissue of the hepatocystic triangle -- i.e. exactly the anatomy this clip is
    for. Masking that away would be far more damaging than leaving the instruments in, so the tools
    stay in the reconstruction and are modeled as moving structures by the 4D field.
    """
    a = rgb.astype(np.float32)
    val = a.max(-1) / 255.0
    m = (val < 0.10).astype(np.uint8) * 255         # vignette + near-black shafts
    m[:pad, :] = 255                                # padding rows are never valid content
    m[pad + nh:, :] = 255
    im = Image.fromarray(m)
    for _ in range(2):                              # close the soft vignette edge
        im = im.filter(ImageFilter.MaxFilter(3))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/Endoscapes-CVS201/selected")
    ap.add_argument("--out", default="data/endonerf/endoscapes_83")
    ap.add_argument("--no-depth", action="store_true", help="skip monodepth (masks/images only)")
    ap.add_argument("--camera-extent-hint", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--crop", type=int, nargs=4, default=None, metavar=("L", "T", "R", "B"),
                    help="vignette content bbox for this clip; measure it, do not assume")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*.jpg")))
    assert files, f"no jpgs in {args.src}"
    for s in ("images", "monodepth", "masks"):
        os.makedirs(os.path.join(args.out, s), exist_ok=True)

    # ---- images + masks -------------------------------------------------
    rgbs = []
    for i, f in enumerate(files):
        im, pad, nh = letterbox(Image.open(f).convert("RGB"), Image.LANCZOS, tuple(args.crop) if args.crop else None)
        im.save(os.path.join(args.out, "images", f"{i:05d}.png"))
        rgb = np.asarray(im)
        rgbs.append(rgb)
        invalid_mask(rgb, pad, nh).save(os.path.join(args.out, "masks", f"{i:05d}.png"))
    N = len(files)
    excl = np.mean([np.asarray(Image.open(os.path.join(args.out, "masks", f"{i:05d}.png"))) > 127
                    for i in range(N)])
    print(f"[endoscapes] {N} frames -> {args.out}  (excluded {excl*100:.1f}% of pixels)")

    # ---- monocular depth ------------------------------------------------
    if not args.no_depth:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
        net = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(dev).eval()
        for i, rgb in enumerate(rgbs):
            inp = proc(images=Image.fromarray(rgb), return_tensors="pt").to(dev)
            with torch.no_grad():
                d = net(**inp).predicted_depth[0]                    # disparity-like: bright = near
            d = torch.nn.functional.interpolate(d[None, None], size=(H, W),
                                                mode="bicubic", align_corners=False)[0, 0]
            d = d.float().cpu().numpy()
            lo, hi = np.percentile(d, 1), np.percentile(d, 99)
            d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
            # Compress into a physically sensible near/far ratio. The loader maps stored disparity
            # s in (0,1] to depth (1/s)*0.4, so s in [DISP_MIN,1] gives depth [0.4, 0.4/DISP_MIN].
            # Using the full 8-bit range would give a 255:1 depth span -- absurd for endoscopy, where
            # tissue spans roughly 3:1 -- and it produces an elongated point cloud that destabilises
            # densification (observed: NaN loss and collapse to ~8 dB).
            d8 = np.round((DISP_MIN + d * (1.0 - DISP_MIN)) * 255).astype(np.uint8)
            Image.fromarray(np.repeat(d8[..., None], 3, -1)).save(
                os.path.join(args.out, "monodepth", f"{i:05d}.png"))
        print(f"[endoscapes] monodepth written with {DEPTH_MODEL} on {dev}")

    # ---- static poses ---------------------------------------------------
    focal = (W / 2) / np.tan(np.deg2rad(FOV_DEG) / 2)
    pose = np.zeros((3, 5), np.float32)
    pose[0, 0] = pose[1, 1] = pose[2, 2] = 1.0        # R=I, t=0 (static-camera assumption)
    pose[:, 4] = [H, W, focal]
    # loader maps stored disparity d in (0,1] to depth (1/d)*0.4 -> range [0.4, 102]
    row = np.concatenate([pose.reshape(-1), [0.4, 4.0]])
    np.save(os.path.join(args.out, "poses_bounds.npy"), np.tile(row, (N, 1)).astype(np.float32))
    print(f"[endoscapes] poses_bounds.npy: static, focal={focal:.1f}px (assumed {FOV_DEG:.0f}deg FOV)")


if __name__ == "__main__":
    main()
