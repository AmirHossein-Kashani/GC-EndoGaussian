"""Append the STIR seq33 ground-truth-verified reveal to the presentation demo video.

demoF2_presentation.mp4 (63 s, 1280x1024 @30) + [intro card -> captioned reveal footage ->
verification card] -> demoF2_presentation_v2.mp4. Cards reuse the dark research-talk style of
tools/make_presentation_cards.py so the appended chapter looks native. CPU only.
"""
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

FDIR = "/usr/share/fonts/dejavu-sans-fonts"
W, H, FPS = 1280, 1024, 30
BG = (13, 17, 23)
WHITE = (236, 239, 241)
MUTE = (150, 162, 173)
GREEN = (57, 255, 20)
CYAN = (0, 209, 255)
MARGIN = 112

SRC = "docs/supplementary/demoF2_presentation.mp4"
FRAMES = "output/stir/seq33_6k/timefill/frames_f8_ref90"
GT90 = "data/endonerf/stir06_seq33/images/00090.png"
OUT = "docs/supplementary/demoF2_presentation_v2.mp4"
ZOOM = (170, 150, 470, 370)          # hole crop used in the paper figure (640x512 coords)


def font(bold, size):
    return ImageFont.truetype(os.path.join(FDIR, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)


def tw(d, s, f):
    b = d.textbbox((0, 0), s, font=f)
    return b[2] - b[0]


def wrap(d, text, f, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if tw(d, t, f) <= maxw:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def base(accent=GREEN):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 14, H], fill=accent)
    d.rectangle([0, H - 8, W, H], fill=accent)
    return im, d


def bullets(d, items, y0, f, fb, gap=96):
    y = y0
    for it in items:
        d.text((MARGIN, y + 4), "▸", font=fb, fill=GREEN)
        x0 = MARGIN + 46
        for ln in wrap(d, it, f, W - x0 - MARGIN):
            d.text((x0, y), ln, font=f, fill=WHITE)
            y += 50
        y += gap - 50
    return y


def to_bgr(im):
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def card_intro():
    im, d = base()
    d.text((MARGIN, 128), "Bonus result  ·  STIR dataset (real da Vinci stereo video)", font=font(0, 30), fill=CYAN)
    d.text((MARGIN, 196), "Seeing what the tissue hides", font=font(1, 62), fill=WHITE)
    d.text((MARGIN, 288), "a ground-truth-verified removal edit", font=font(1, 38), fill=MUTE)
    bullets(d, [
        "An organ covers the tissue bed; later in the clip it is dragged aside, "
        "so the hidden region becomes visible.",
        "We remove the organ's Gaussians at an early frame — an inference-time edit "
        "of the reconstruction, no retraining.",
        "The revealed region is re-shown at the time it was observed — and matches "
        "the real footage at 24.8 dB.",
    ], 400, font(0, 32), font(1, 32), gap=120)
    d.text((MARGIN, H - 60), "STIR: Schmidt et al., IEEE TMI 2024  ·  collection 06 / seq33", font=font(0, 24), fill=MUTE)
    return to_bgr(im)


def card_verify(fill_crop, gt_crop):
    im, d = base(CYAN)
    d.text((MARGIN, 110), "Verification", font=font(1, 64), fill=WHITE)
    d.text((MARGIN, 210), "revealed region (model)  vs.  ground truth, later in the same clip",
           font=font(0, 34), fill=MUTE)
    ch, cw = 480, 520
    fc = cv2.resize(fill_crop, (cw, ch), interpolation=cv2.INTER_CUBIC)
    gc = cv2.resize(gt_crop, (cw, ch), interpolation=cv2.INTER_CUBIC)
    arr = np.array(im)
    y0 = 300
    x1 = MARGIN
    x2 = W - MARGIN - cw
    arr[y0:y0 + ch, x1:x1 + cw] = cv2.cvtColor(fc, cv2.COLOR_BGR2RGB)
    arr[y0:y0 + ch, x2:x2 + cw] = cv2.cvtColor(gc, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    d.rectangle([x1 - 3, y0 - 3, x1 + cw + 2, y0 + ch + 2], outline=(57, 255, 20), width=3)
    d.rectangle([x2 - 3, y0 - 3, x2 + cw + 2, y0 + ch + 2], outline=(0, 209, 255), width=3)
    d.text((x1, y0 + ch + 18), "revealed by the edit", font=font(1, 30), fill=GREEN)
    d.text((x2, y0 + ch + 18), "ground truth (later frame)", font=font(1, 30), fill=CYAN)
    d.text((MARGIN, y0 + ch + 92), "24.8 dB in the revealed region —", font=font(1, 34), fill=WHITE)
    d.text((MARGIN, y0 + ch + 142), "the model re-shows what it truly observed; it does not hallucinate.",
           font=font(0, 30), fill=MUTE)
    return to_bgr(im)


def caption_bar(fr, headline, sub):
    im = fr.copy()
    cv2.rectangle(im, (0, H - 130), (W, H), (17, 13, 10), -1)
    cv2.rectangle(im, (0, H - 130), (W, H - 126), (20, 255, 57), -1)
    pim = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pim)
    d.text((40, H - 112), headline, font=font(1, 36), fill=WHITE)
    d.text((40, H - 58), sub, font=font(0, 28), fill=MUTE)
    d.text((W - 470, 20), "STIR 06/seq33 · real surgical stereo video", font=font(1, 24), fill=(230, 230, 230))
    return cv2.cvtColor(np.array(pim), cv2.COLOR_RGB2BGR)


def main():
    import glob
    tf = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(FRAMES, "f*.png")))]
    assert len(tf) == 21, f"expected 21 timefill frames, got {len(tf)}"
    up = [cv2.resize(f, (W, H), interpolation=cv2.INTER_CUBIC) for f in tf]

    seg = []
    seg += [caption_bar(up[0], "Before the edit (frame 8 of the clip)",
                        "The organ covers the tissue bed — what is underneath is hidden at this moment.")] * (3 * FPS)
    for f in up[1:13]:
        seg += [caption_bar(f, "Removing the organ's Gaussians, front to back",
                            "Inference-time edit on the reconstruction — no retraining, under 5 ms per re-render.")] * 6
    for f in up[13:21]:
        seg += [caption_bar(f, "Filling the reveal with what the model actually observed",
                            "The bed was seen later in the clip; it is re-shown here at its observed time.")] * 6
    seg += [caption_bar(up[20], "The hidden tissue bed — recovered from the reconstruction",
                        "Verified next against real footage from later in the same clip.")] * (4 * FPS)

    x0, y0, x1, y1 = ZOOM
    fill_crop = tf[20][y0:y1, x0:x1]
    gt = cv2.imread(GT90)
    gt_crop = gt[y0:y1, x0:x1]

    cards = [(card_intro(), 8 * FPS), (card_verify(fill_crop, gt_crop), 7 * FPS)]

    vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    cap = cv2.VideoCapture(SRC)
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        vw.write(fr); n += 1
    cap.release()
    for im, cnt in [cards[0]]:
        for _ in range(cnt):
            vw.write(im)
    for f in seg:
        vw.write(f)
    for im, cnt in [cards[1]]:
        for _ in range(cnt):
            vw.write(im)
    vw.release()
    total = n + cards[0][1] + len(seg) + cards[1][1]
    print(f"[v2] {OUT}: {n} original + {cards[0][1] + len(seg) + cards[1][1]} new = {total} frames "
          f"({total / FPS:.1f} s)")


if __name__ == "__main__":
    main()
