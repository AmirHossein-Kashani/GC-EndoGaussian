"""Final figure: remove a region of tissue at inference time and show what the model holds behind it.

What this demonstrates, and how it differs from the baseline. EndoGaussian already renders tissue
where a *tool* was, but only because tool pixels are painted out of the loss by a mask supplied
before training -- no editing is involved and nothing is removed at inference. Here we remove
ordinary, un-masked tissue at inference time through the control-graph interface, and the
reconstruction composites whatever it holds behind that tissue. That is the editable-interface
contribution, not the base model's masking.

The panel is built from a saved peel run (tools/endoscapes_peel.py --steps N writes frames_<tag>/),
so this script is CPU-only.

  python3 tools/make_peel_figure.py \
      --frames output/endonerf/cutting_match/peel/frames_flap_video \
      --rect 240 60 420 240 --zoom 250 70 415 235 \
      --out docs/figures/peel_reveal.png
"""
import os, glob, argparse
import numpy as np
import cv2

G = (70, 235, 70)


def cap(im, txt, color=(255, 255, 255), h=34):
    im = im.copy()
    cv2.rectangle(im, (0, im.shape[0] - h), (im.shape[1], im.shape[0]), (0, 0, 0), -1)
    cv2.putText(im, txt, (10, im.shape[0] - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="frames_<tag> dir from endoscapes_peel.py")
    ap.add_argument("--rect", type=int, nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--zoom", type=int, nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--out", default="docs/figures/peel_reveal.png")
    args = ap.parse_args()

    fs = sorted(glob.glob(os.path.join(args.frames, "*.png")))
    if len(fs) < 2:
        raise SystemExit(f"need at least 2 frames in {args.frames}")
    before = cv2.imread(fs[0])[:, :, ::-1].copy()
    after = cv2.imread(fs[-1])[:, :, ::-1].copy()

    x0, y0, x1, y1 = args.rect
    diff = np.abs(after.astype(np.float32) - before.astype(np.float32)).mean(-1)
    changed = float((diff > 8).mean() * 100)

    b = before.copy(); cv2.rectangle(b, (x0, y0), (x1, y1), G, 2)
    a = after.copy();  cv2.rectangle(a, (x0, y0), (x1, y1), G, 2)
    heat = cv2.applyColorMap(np.clip(diff / max(diff.max(), 1e-6) * 255, 0, 255).astype(np.uint8),
                             cv2.COLORMAP_INFERNO)[:, :, ::-1].copy()

    top = np.concatenate([
        cap(b, "1. reconstruction  -  tissue selected for removal (green box)"),
        cap(a, "2. after removing the selected tissue at inference time", (110, 255, 110)),
        cap(heat, "3. what changed  -  confined to the selection"),
    ], 1)

    zx0, zy0, zx1, zy1 = args.zoom
    zw = top.shape[1] // 2
    zh = int(zw * (zy1 - zy0) / (zx1 - zx0))
    z = lambda im: cv2.resize(im[zy0:zy1, zx0:zx1], (zw, zh), interpolation=cv2.INTER_CUBIC)
    bot = np.concatenate([
        cap(z(before), "ZOOM  -  before: front tissue layer present"),
        cap(z(after), "ZOOM  -  after: the surface behind it, held by the model", (110, 255, 110)),
    ], 1)

    fig = np.concatenate([top, bot], 0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, fig[:, :, ::-1])
    print(f"[fig] {len(fs)} peel steps | edit changed {changed:.1f}% of pixels -> {args.out}")


if __name__ == "__main__":
    main()
