"""Detect, with no manual input, the frames in which a selected region stands exposed.

Used to remove the last hand-tuned parameter from the STIR reveals (the reference frame). Signals
derived from the RECONSTRUCTION do not work: the deformation field recycles primitives instead of
transporting them, so the object's own Gaussians never leave the footprint (coverage bottoms out at
48%) and the model's rendered depth there swings only ~11mm across a whole clip. The INPUT stereo
depth does record it -- the region reads shallow while the object rests on it and steps back once
the object is dragged away -- so base-frame detection is done on the dataset's depth maps.

Validated: the frame this picks reproduces (seq23: 168 vs the hand-picked 170, 25.66 vs 25.9 dB) or
improves on (seq33: 26 vs 90, 25.12 vs 24.5 dB) the manual choice in tools/stir_object_reveal.py.

  python3 tools/detect_base_frames.py --scene data/endonerf/stir06_seq23 \
      --mask data/endonerf/stir06_seq23/organ_mask_f10.png
"""
import argparse, glob
import numpy as np
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="EndoNeRF-layout scene dir (needs depth/)")
    ap.add_argument("--mask", required=True, help="footprint of the object to look underneath")
    ap.add_argument("--min_px", type=int, default=50)
    args = ap.parse_args()

    m = cv2.imread(args.mask, 0) > 127
    med = []
    for p in sorted(glob.glob(f"{args.scene}/depth/*.png")):
        d = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32)
        v = d[m & (d > 0)]
        med.append(float(np.median(v)) if v.size > args.min_px else 0.0)
    med = np.array(med)
    ok = med > 0
    thr = (med[ok].min() + med[ok].max()) / 2          # midpoint of the covered/exposed swing
    exposed = np.where(med > thr)[0]
    best = int(np.argmax(med))
    print(f"footprint depth {med[ok].min():.0f}-{med[ok].max():.0f}mm (swing {med[ok].ptp():.0f}mm)")
    print(f"exposed frames (> {thr:.0f}mm): {exposed.tolist()}")
    print(f"BASE FRAME (most exposed) = {best}  depth {med[best]:.0f}mm")
    if med[ok].ptp() < 20:
        print("WARNING: weak depth swing -- the object may not clear the region in this clip")


if __name__ == "__main__":
    main()
