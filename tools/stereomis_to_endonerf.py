"""Convert a StereoMIS window into the EndoNeRF on-disk layout, keeping the REAL camera poses.

Why this dataset: every scene used so far (EndoNeRF, SuPer, Hamlyn, Endoscapes) writes a single
repeated pose into poses_bounds.npy -- a fixed viewpoint. A fixed viewpoint can only ever produce a
surface, never a layered volume, so "remove this tissue and see what is behind" has nothing to show.
StereoMIS is stereo video from a da Vinci Xi *with camera forward kinematics*, so the camera really
moves and surfaces are observed from several angles. scene/endo_loader.py already reads per-frame
poses, so no loader change is needed -- this is the first scene to use that path for real.

Layout: video is left/right stacked vertically (1280x2048 -> two 1280x1024 eyes), one tool mask per
frame, groundtruth.txt in TUM form `idx tx ty tz qx qy qz qw`, intrinsics in StereoCalibration.ini.

Depth comes from stereo SGBM in millimetres (focal*baseline/disparity), exactly as
tools/super_to_endonerf.py does.

**Pose scale.** The kinematics translations are in an unknown unit relative to our millimetre depth,
and getting it wrong makes the poses inconsistent with the geometry. We therefore *calibrate* it:
back-project frame A's depth, move it by the relative pose with a trial scale, project into frame B,
and pick the scale minimising disagreement with measured optical flow. --pose_scale overrides.

  python3 tools/stereomis_to_endonerf.py --seq data/stereomis/StereoMIS/P2_6 \
      --start 6050 --count 120 --step 3 --out data/endonerf/stereomis_p26
"""
import os, argparse, configparser
import numpy as np
import cv2

W_OUT, H_OUT = 640, 512          # what the loader hardcodes


def quat_to_R(q):
    x, y, z, w = q
    n = np.linalg.norm([x, y, z, w])
    x, y, z, w = np.array([x, y, z, w]) / max(n, 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def read_calib(path):
    cp = configparser.ConfigParser()
    cp.read(path)
    L = cp["StereoLeft"]
    fx, fy = float(L["fc_x"]), float(L["fc_y"])
    res_x, res_y = float(L["res_x"]), float(L["res_y"])
    # stereo translation lives in the right-camera section; first component is the baseline
    R = cp["StereoRight"]
    base = abs(float(R.get("T_0", -4.16)))
    return fx, fy, res_x, res_y, base


def grab(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    if not ok:
        return None, None
    h = fr.shape[0] // 2
    return fr[:h], fr[h:]                      # left (top), right (bottom), BGR


def sgbm_depth(l, r, focal, base):
    m = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=7,
                              P1=8 * 3 * 49, P2=32 * 3 * 49, disp12MaxDiff=1,
                              uniquenessRatio=10, speckleWindowSize=100, speckleRange=2)
    d = m.compute(cv2.cvtColor(l, cv2.COLOR_BGR2GRAY),
                  cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16.0
    depth = np.zeros_like(d)
    ok = d > 0.5
    depth[ok] = focal * base / d[ok]
    depth[(depth < 5) | (depth > 400)] = 0
    return depth


def calibrate_scale(capA, idxs, poses, focal, base, fx_out, cx, cy):
    """Pick the translation scale that best matches measured optical flow."""
    a, b = idxs[0], idxs[min(len(idxs) - 1, 8)]
    la, ra = grab(capA, a)
    lb, _ = grab(capA, b)
    if la is None or lb is None:
        return 1.0
    dep = sgbm_depth(la, ra, focal, base)
    dep = cv2.resize(dep, (W_OUT, H_OUT), interpolation=cv2.INTER_NEAREST)
    ga = cv2.cvtColor(cv2.resize(la, (W_OUT, H_OUT)), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(lb, (W_OUT, H_OUT)), cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 25, 3, 5, 1.2, 0)

    Ra, ta = poses[0]
    Rb, tb = poses[min(len(poses) - 1, 8)]
    ys, xs = np.mgrid[0:H_OUT, 0:W_OUT]
    m = dep > 0
    xs, ys, z = xs[m], ys[m], dep[m]
    X = (xs - cx) * z / fx_out
    Y = (ys - cy) * z / fx_out
    P = np.stack([X, Y, z], 1)
    fu, fv = flow[..., 0][m], flow[..., 1][m]

    best, bs = None, 1.0
    for s in np.concatenate([np.geomspace(1e-2, 1e4, 60), [1.0]]):
        Rrel = Rb.T @ Ra
        trel = (Rb.T @ ((ta - tb) * s))
        Q = P @ Rrel.T + trel
        zz = np.clip(Q[:, 2], 1e-3, None)
        u = fx_out * Q[:, 0] / zz + cx
        v = fx_out * Q[:, 1] / zz + cy
        err = np.median(np.hypot(u - (xs + fu), v - (ys + fv)))
        if best is None or err < best:
            best, bs = err, s
    print(f"[calib] pose scale {bs:.4g} (median reprojection error {best:.2f} px)")
    return bs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--start", type=int, default=6050)
    ap.add_argument("--count", type=int, default=120)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pose_scale", type=float, default=None)
    args = ap.parse_args()

    vids = [f for f in os.listdir(args.seq) if f.endswith(".mp4")]
    vid = os.path.join(args.seq, vids[0])
    fx, fy, res_x, res_y, base = read_calib(os.path.join(args.seq, "StereoCalibration.ini"))
    gt = np.loadtxt(os.path.join(args.seq, "groundtruth.txt"))
    idxs = [args.start + i * args.step for i in range(args.count)]

    sc = W_OUT / res_x                       # 1280 -> 640
    fx_out, cx, cy = fx * sc, W_OUT / 2, H_OUT / 2

    poses = []
    for i in idxs:
        row = gt[i]
        poses.append((quat_to_R(row[4:8]), row[1:4].astype(np.float64)))

    cap = cv2.VideoCapture(vid)
    scale = args.pose_scale if args.pose_scale is not None else \
        calibrate_scale(cap, idxs, poses, fx, base, fx_out, cx, cy)

    for s in ("images", "depth", "masks"):
        os.makedirs(os.path.join(args.out, s), exist_ok=True)

    meds = []
    for k, i in enumerate(idxs):
        l, r = grab(cap, i)
        if l is None:
            raise RuntimeError(f"could not read frame {i}")
        dep = sgbm_depth(l, r, fx, base)
        img = cv2.resize(l, (W_OUT, H_OUT), interpolation=cv2.INTER_AREA)
        dep = cv2.resize(dep, (W_OUT, H_OUT), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(args.out, "images", f"{k:05d}.png"), img)
        cv2.imwrite(os.path.join(args.out, "depth", f"{k:05d}.png"), dep.astype(np.uint16))
        mp = os.path.join(args.seq, "masks", f"{i:06d}l.png")
        if os.path.exists(mp):
            mk = cv2.imread(mp, 0)
            mk = cv2.resize(mk, (W_OUT, H_OUT), interpolation=cv2.INTER_NEAREST)
            mk = (mk < 127).astype(np.uint8) * 255      # StereoMIS: 0 = tool -> EndoNeRF: 255 = tool
        else:
            mk = np.zeros((H_OUT, W_OUT), np.uint8)
        cv2.imwrite(os.path.join(args.out, "masks", f"{k:05d}.png"), mk)
        if (dep > 0).any():
            meds.append(float(np.median(dep[dep > 0])))

    # ---- poses_bounds.npy with REAL per-frame poses -------------------------
    rows = []
    near = max(5.0, np.percentile(meds, 5) * 0.5)
    far = np.percentile(meds, 95) * 1.5
    R0, t0 = poses[0]
    for (R, t) in poses:
        # express every pose relative to the first, so the scene sits near the origin
        Rr = R0.T @ R
        tr = (R0.T @ ((t - t0) * scale)).reshape(3, 1)
        p = np.concatenate([Rr, tr], 1)                       # 3x4 c2w
        p = np.concatenate([p, np.array([[H_OUT], [W_OUT], [fx_out]])], 1)   # 3x5
        rows.append(np.concatenate([p.reshape(-1), [near, far]]))
    np.save(os.path.join(args.out, "poses_bounds.npy"), np.stack(rows).astype(np.float32))

    tt = np.stack([r[:15].reshape(3, 5)[:, 3] for r in rows])
    print(f"[stereomis] {len(idxs)} frames -> {args.out}")
    print(f"[stereomis] depth median {np.median(meds):.1f} mm | near/far {near:.1f}/{far:.1f}")
    print(f"[stereomis] camera translation span (mm): {np.round(tt.max(0)-tt.min(0),2)}")
    print(f"[stereomis] set camera_extent ~= {np.median(meds):.0f} in the config")


if __name__ == "__main__":
    main()
