"""Convert a C3VD colonoscopy sequence into the EndoNeRF layout, with REAL camera poses.

Why C3VD: every other scene here (EndoNeRF, SuPer, Hamlyn, Endoscapes) is captured from an
effectively fixed viewpoint, which can only ever reconstruct a *surface with thickness* -- never a
layered volume. Removing the front layer therefore exposes that same surface's backing (measured:
45-78% of occluded Gaussians were observed elsewhere, yet removal changed only 1-2% of pixels).
C3VD has a colonoscope travelling down the lumen with ground-truth poses and depth, so surfaces are
seen from multiple angles and genuinely occluded volume can be reconstructed. Haustral folds are
real occluders -- the clinical motivation being polyps missed behind folds.

Conventions, all verified rather than assumed (see the validation below):
- Camera is a cfhq190l ~190-degree fisheye described by the Scaramuzza omnidirectional model, NOT a
  pinhole. 3DGS renders pinhole, so frames are undistorted first (tools/c3vd_undistort.py).
- pose.txt rows are 4x4 row-vector matrices: R_c2w = M[:3,:3].T and t = M[3,:3], in millimetres.
- *_depth.tiff is uint16 with full scale = 100 mm, measured as distance ALONG THE RAY, so it is
  converted to z-depth for the pinhole camera.
- *_occlusion.png marks pixels with no valid ground truth; they are excluded from the loss.

Validation (tools/... /tmp/c3vd_corr2.py): predicting optical flow from poses+depth and comparing
against the dataset's ground-truth flow gives correlation 0.946 and 0.43 px residual. Two earlier
conventions failed this test outright, which is why it is worth running before training.

  python3 tools/c3vd_to_endonerf.py --src data/desc_t4_a --out data/endonerf/c3vd_desc_t4a
"""
import os, glob, argparse
import numpy as np
import cv2

W_OUT, H_OUT = 640, 512
CAL = dict(cx=679.544839263292, cy=543.975887548343,
           a0=769.243600037458, a2=-0.000812770624150226,
           a3=6.25674244578925e-07, a4=-1.19662182144280e-09,
           e=0.999986882249990, f=0.00288273829525059, g=-0.00296316513429569)
DEPTH_FULL_MM = 100.0


def _tables():
    rho = np.linspace(0, 900, 6000)
    z = CAL["a0"] + CAL["a2"] * rho ** 2 + CAL["a3"] * rho ** 3 + CAL["a4"] * rho ** 4
    th = np.arctan2(rho, z)
    o = np.argsort(th)
    return th[o], rho[o]


TH_T, RHO_T = _tables()


def undistort_maps(fov_deg, out):
    """Sampling map taking the virtual pinhole image back to fisheye pixels."""
    ow, oh = out
    fv = (ow / 2) / np.tan(np.radians(fov_deg) / 2)
    ys, xs = np.mgrid[0:oh, 0:ow].astype(np.float64)
    dx, dy = xs - ow / 2, ys - oh / 2
    r = np.hypot(dx, dy)
    th = np.arctan2(r, fv)
    rho = np.interp(th, TH_T, RHO_T, left=0, right=np.nan)
    s = np.divide(rho, np.maximum(r, 1e-9))
    u, v = dx * s, dy * s
    mx = u * CAL["e"] + v * CAL["f"] + CAL["cx"]
    my = u * CAL["g"] + v * 1.0 + CAL["cy"]
    ok = np.isfinite(mx) & np.isfinite(my)
    return (np.where(ok, mx, -1).astype(np.float32),
            np.where(ok, my, -1).astype(np.float32), fv, th)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fov", type=float, default=100.0)
    ap.add_argument("--step", type=int, default=1)
    args = ap.parse_args()

    colors = sorted(glob.glob(os.path.join(args.src, "*_color.png")))[::args.step]
    assert colors, f"no *_color.png in {args.src}"
    M = np.loadtxt(os.path.join(args.src, "pose.txt"), delimiter=",").reshape(-1, 4, 4)

    mx, my, fv, th = undistort_maps(args.fov, (W_OUT, H_OUT))
    cos_th = np.cos(th).astype(np.float32)          # ray distance -> z depth

    for s in ("images", "depth", "masks"):
        os.makedirs(os.path.join(args.out, s), exist_ok=True)

    meds, rows = [], []
    for k, cp in enumerate(colors):
        i = int(os.path.basename(cp).split("_")[0])
        col = cv2.remap(cv2.imread(cp), mx, my, cv2.INTER_LINEAR, borderValue=0)
        draw = cv2.imread(cp.replace("_color.png", "_depth.tiff"), cv2.IMREAD_UNCHANGED).astype(np.float32)
        dep = cv2.remap(draw, mx, my, cv2.INTER_NEAREST, borderValue=0)
        dep = dep / 65535.0 * DEPTH_FULL_MM * cos_th          # -> z depth in mm
        occ = cv2.remap(cv2.imread(cp.replace("_color.png", "_occlusion.png"), 0),
                        mx, my, cv2.INTER_NEAREST, borderValue=255)
        invalid = (occ > 127) | (dep <= 0.5) | (mx < 0)       # no GT here -> exclude from loss
        dep[invalid] = 0

        cv2.imwrite(os.path.join(args.out, "images", f"{k:05d}.png"), col)
        cv2.imwrite(os.path.join(args.out, "depth", f"{k:05d}.png"), dep.astype(np.uint16))
        cv2.imwrite(os.path.join(args.out, "masks", f"{k:05d}.png"),
                    (invalid.astype(np.uint8) * 255))          # EndoNeRF: white = excluded
        if (dep > 0).any():
            meds.append(float(np.median(dep[dep > 0])))

        R = M[i][:3, :3].T                                     # verified convention
        t = M[i][3, :3]
        p = np.concatenate([np.concatenate([R, t.reshape(3, 1)], 1),
                            np.array([[H_OUT], [W_OUT], [fv]])], 1)   # 3x5
        rows.append(p.reshape(-1))

    med = float(np.median(meds))
    near = max(1.0, np.percentile(meds, 5) * 0.4)
    far = np.percentile(meds, 95) * 2.0
    pb = np.stack([np.concatenate([r, [near, far]]) for r in rows]).astype(np.float32)
    np.save(os.path.join(args.out, "poses_bounds.npy"), pb)

    t_all = pb[:, :15].reshape(-1, 3, 5)[:, :, 3]
    print(f"[c3vd] {len(colors)} frames -> {args.out}")
    print(f"[c3vd] undistorted to {W_OUT}x{H_OUT}, {args.fov:.0f} deg FOV, focal {fv:.1f} px")
    print(f"[c3vd] depth median {med:.1f} mm | near/far {near:.1f}/{far:.1f}")
    print(f"[c3vd] camera translation span (mm): {np.round(t_all.max(0)-t_all.min(0),2)} "
          f"| path {np.linalg.norm(np.diff(t_all,axis=0),axis=1).sum():.1f}")
    print(f"[c3vd] -> set camera_extent ~= {med:.0f}")


if __name__ == "__main__":
    main()
