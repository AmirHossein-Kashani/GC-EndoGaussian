"""Convert a STIR Challenge 2024 sequence into the EndoNeRF on-disk layout.

Why this dataset (chosen for the inference-time reveal experiment): STIR clips are stereo-rectified
1280x1024 video from a da Vinci Xi with a *static* camera, showing tissue pieces being grasped and
rolled/flipped off the bed they cover. That is exactly the configuration in which the removal edit
demonstrably works (EndoNeRF `pulling`/`cutting`): real stereo depth gives the occluder its own
depth stratum, and because the occluder moves away mid-clip, the bed underneath is directly observed
at other timestamps -- so the canonical Gaussian set holds real content behind the occluder, and GT
frames verify the reveal.

Layout of a STIR sequence (per collection calib.json):
  <coll>/left/seqNN/frames/*-visible.mp4      1280x1024 left eye
  <coll>/right/seqNN/frames/*-visible.mp4     1280x1024 right eye
  <coll>/calib.json                            rectified intrinsics; principal points DIFFER per eye

Depth from stereo SGBM. The rectification uses different principal points per eye, so the measured
disparity d relates to depth by  Z = fx*B / (d - (cx_l - cx_r)):  at 06/calib.json the offset
cx_l - cx_r ~= -90.6 px, i.e. true disparity = d + 90.6 and measured disparities are mostly
NEGATIVE -- SGBM must run with a negative minDisparity. Everything is computed at 640x512 (exactly
half resolution), the size the EndoNeRF loader hardcodes, so no padding is needed.

No tool masks exist; masks are written empty (tools stay in the scene and are modelled as moving
structures -- the Endoscapes-83 precedent).

  python3 tools/stir_to_endonerf.py --seq data/stir/STIRChallenge_2024/06 --name seq23 \
      --out data/endonerf/stir06_seq23
"""
import os, glob, json, argparse
import numpy as np
import cv2

W_OUT, H_OUT = 640, 512          # exactly half of 1280x1024; what the loader hardcodes


def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="collection dir containing calib.json + left/ right/")
    ap.add_argument("--name", required=True, help="sequence name, e.g. seq23")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=1)
    args = ap.parse_args()

    calib = json.load(open(os.path.join(args.seq, "calib.json")))
    fx = calib["leftcameramat"][0][0]
    cx_l = calib["leftcameramat"][0][2]
    cx_r = calib["rightcameramat"][0][2]
    base_mm = abs(calib["translation"][0]) * 1000.0
    # half resolution: focal, principal-point offset and disparity all scale by 0.5
    fx_h = fx * 0.5
    doff = (cx_l - cx_r) * 0.5              # true_disp = measured_disp - doff  (doff is negative)

    lvid = glob.glob(os.path.join(args.seq, "left", args.name, "frames", "*.mp4"))
    rvid = glob.glob(os.path.join(args.seq, "right", args.name, "frames", "*.mp4"))
    assert lvid and rvid, f"missing videos for {args.name}"
    L = read_video(lvid[0])[:: args.step]
    R = read_video(rvid[0])[:: args.step]
    n = min(len(L), len(R))
    L, R = L[:n], R[:n]
    print(f"[stir] {args.name}: {n} stereo frames | fx/2={fx_h:.1f} base={base_mm:.2f}mm doff={doff:.1f}px")

    # true disparity fx_h*B/Z for Z in [25, 400] mm -> [5.3, 85]; measured = true + doff (~ -45.3)
    sgbm = cv2.StereoSGBM_create(
        minDisparity=-64, numDisparities=112, blockSize=7,
        P1=8 * 3 * 49, P2=32 * 3 * 49, disp12MaxDiff=1,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=2)

    for s in ("images", "depth", "masks"):
        os.makedirs(os.path.join(args.out, s), exist_ok=True)

    meds, cover = [], []
    for i in range(n):
        l = cv2.resize(L[i], (W_OUT, H_OUT), interpolation=cv2.INTER_AREA)
        r = cv2.resize(R[i], (W_OUT, H_OUT), interpolation=cv2.INTER_AREA)
        d = sgbm.compute(cv2.cvtColor(l, cv2.COLOR_BGR2GRAY),
                         cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16.0
        true_disp = d - doff
        depth = np.zeros_like(d)
        ok = (d > -63.5) & (true_disp > 2.0)
        depth[ok] = fx_h * base_mm / true_disp[ok]
        depth[(depth < 20) | (depth > 400)] = 0
        cv2.imwrite(os.path.join(args.out, "images", f"{i:05d}.png"), l)
        cv2.imwrite(os.path.join(args.out, "depth", f"{i:05d}.png"), depth.astype(np.uint16))
        cv2.imwrite(os.path.join(args.out, "masks", f"{i:05d}.png"), np.zeros((H_OUT, W_OUT), np.uint8))
        if (depth > 0).any():
            meds.append(float(np.median(depth[depth > 0])))
            cover.append(float((depth > 0).mean()))

    near = max(5.0, np.percentile(meds, 5) * 0.5)
    far = np.percentile(meds, 95) * 1.5
    pose = np.zeros((3, 5), np.float32)
    pose[0, 0] = pose[1, 1] = pose[2, 2] = 1.0           # static camera, R=I t=0
    pose[:, 4] = [H_OUT, W_OUT, fx_h]
    row = np.concatenate([pose.reshape(-1), [near, far]])
    np.save(os.path.join(args.out, "poses_bounds.npy"), np.tile(row, (n, 1)))

    print(f"[stir] wrote {n} frames -> {args.out}")
    print(f"[stir] depth median {np.median(meds):.1f} mm | coverage {np.mean(cover)*100:.1f}% | near/far {near:.1f}/{far:.1f}")


if __name__ == "__main__":
    main()
