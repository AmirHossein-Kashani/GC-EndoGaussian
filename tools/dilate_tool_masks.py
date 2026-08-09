"""Grow (dilate) the tool masks for SuPer trials so the whole tool + a margin is
excluded from training — the lever for cleaner tool removal (there is no scalar
'tool-removal' coefficient; the masking is binary, so coverage is what matters).

Tool masks are white(255)=tool, black=tissue; the loader does `1 - mask/255`, so a
larger white region excludes more. We dilate the white region by `--radius` pixels
(repeated 3x3 max-filter, PIL only — runs on the login node, no GPU/cv2).

For each trial it writes a sibling data dir `<trial>_tooldil/` with dilated `masks/`
and symlinks to the original `images/`, `depth/`, `poses_bounds.npy` so training reads
it as a normal EndoNeRF scene.

  python3 tools/dilate_tool_masks.py --root data/endonerf \
      --trials super_trial3 super_trial4 super_trial8 super_trial9 --radius 10
"""
import os, argparse, glob
from PIL import Image, ImageFilter


def dilate(im, r):
    for _ in range(r):                      # each 3x3 max-filter grows white by ~1px
        im = im.filter(ImageFilter.MaxFilter(3))
    return im


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/endonerf")
    ap.add_argument("--trials", nargs="+", required=True)
    ap.add_argument("--radius", type=int, default=10)
    ap.add_argument("--suffix", default="_tooldil")
    args = ap.parse_args()

    for tr in args.trials:
        src = os.path.join(args.root, tr)
        dst = os.path.join(args.root, tr + args.suffix)
        os.makedirs(os.path.join(dst, "masks"), exist_ok=True)
        # symlink everything except masks
        for name in os.listdir(src):
            if name == "masks":
                continue
            link(os.path.join(src, name), os.path.join(dst, name))
        # dilate masks
        mpaths = sorted(glob.glob(os.path.join(src, "masks", "*.png")))
        for p in mpaths:
            im = Image.open(p)
            dilate(im, args.radius).save(os.path.join(dst, "masks", os.path.basename(p)))
        print(f"[{tr}] {len(mpaths)} masks dilated r={args.radius} -> {dst}")


if __name__ == "__main__":
    main()
