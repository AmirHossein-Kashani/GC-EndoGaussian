"""Rebuild demoF2_presentation_v2.mp4 from the ORIGINAL demo plus the object-removal chapter.

Supersedes tools/make_presentation_v2.py, whose chapter showed the earlier ellipse-selection reveal
with a hand-picked reference frame. This version starts again from the reviewed
demoF2_presentation.mp4 and appends the current result: an object selected by its own depth
extent, removed at inference, and the exposed region shown at a timestamp the system detects for
itself -- verified against held-out real frames on both STIR clips.

Structure of the appended chapter (~34 s):
  intro card -> seq23 footage with captions -> seq23 verification card
              -> seq33 footage with captions -> closing card
Cards reuse the dark research-talk style of tools/make_presentation_cards.py. CPU only.
"""
import os, glob
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
OUT = "docs/supplementary/demoF2_presentation_v3.mp4"
SCENES = [
    dict(tag="seq23", frames="output/stir/seq23_6k_tm/autoframe/frames_f10_ref168_AUTO",
         gt="data/endonerf/stir06_seq23/images/00168.png", edit=10, ref=168, db="25.7",
         zoom=(140, 60, 430, 280), banner="STIR 06/seq23"),
    dict(tag="seq33", frames="output/stir/seq33_6k/autoframe/frames_f8_ref26_AUTO",
         gt="data/endonerf/stir06_seq33/images/00026.png", edit=8, ref=26, db="25.1",
         zoom=(170, 150, 470, 370), banner="STIR 06/seq33"),
]


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


def bullets(d, items, y0, f, fb, gap=120):
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
    d.text((MARGIN, 118), "STIR dataset  ·  real da Vinci stereo video", font=font(0, 30), fill=CYAN)
    d.text((MARGIN, 182), "Removing a structure to see", font=font(1, 60), fill=WHITE)
    d.text((MARGIN, 250), "what it hides", font=font(1, 60), fill=WHITE)
    d.text((MARGIN, 338), "an inference-time edit, verified against real footage", font=font(1, 34), fill=MUTE)
    bullets(d, [
        "The selection is an object, not a region: inside a loose footprint the "
        "depths split into the covering structure and the surface below.",
        "Its Gaussians are removed at render time — no retraining, under 5 ms.",
        "The exposed region is shown at a timestamp when it was observed; the "
        "system detects that timestamp itself from the depth in the footprint.",
    ], 430, font(0, 30), font(1, 30), gap=112)
    d.text((MARGIN, H - 58), "Schmidt et al., IEEE TMI 2024  ·  collections 06/seq23 and 06/seq33",
           font=font(0, 24), fill=MUTE)
    return to_bgr(im)


def card_verify(fill_crop, gt_crop, db, ref, tag):
    im, d = base(CYAN)
    d.text((MARGIN, 104), "Verification", font=font(1, 60), fill=WHITE)
    d.text((MARGIN, 196), f"{tag}: revealed region vs. the real frame, later in the same clip",
           font=font(0, 32), fill=MUTE)
    ch, cw = 470, 510
    fc = cv2.resize(fill_crop, (cw, ch), interpolation=cv2.INTER_CUBIC)
    gc = cv2.resize(gt_crop, (cw, ch), interpolation=cv2.INTER_CUBIC)
    arr = np.array(im)
    y0, x1 = 290, MARGIN
    x2 = W - MARGIN - cw
    arr[y0:y0 + ch, x1:x1 + cw] = cv2.cvtColor(fc, cv2.COLOR_BGR2RGB)
    arr[y0:y0 + ch, x2:x2 + cw] = cv2.cvtColor(gc, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    d.rectangle([x1 - 3, y0 - 3, x1 + cw + 2, y0 + ch + 2], outline=(57, 255, 20), width=3)
    d.rectangle([x2 - 3, y0 - 3, x2 + cw + 2, y0 + ch + 2], outline=(0, 209, 255), width=3)
    d.text((x1, y0 + ch + 16), "revealed by the edit", font=font(1, 30), fill=GREEN)
    d.text((x2, y0 + ch + 16), f"real frame {ref}", font=font(1, 30), fill=CYAN)
    d.text((MARGIN, y0 + ch + 86), f"{db} dB in the revealed region", font=font(1, 36), fill=WHITE)
    d.text((MARGIN, y0 + ch + 140),
           "the model re-shows what it observed; nothing is hallucinated", font=font(0, 30), fill=MUTE)
    return to_bgr(im)


def card_close():
    im, d = base()
    d.text((MARGIN, 150), "Both clips, no manual tuning", font=font(1, 54), fill=WHITE)
    bullets(d, [
        "seq23: 25.7 dB in the revealed region · seq33: 25.1 dB — against held-out real frames.",
        "Removal without the temporal step scores 10–13 dB: what it exposes at the edit "
        "timestamp was never constrained by any photograph.",
        "Reference timestamps 168 and 26 were detected automatically, matching or beating "
        "the frames a person picked by hand.",
        "Where the camera saw nothing at any time, the render stays soft — by design.",
    ], 270, font(0, 30), font(1, 30), gap=108)
    return to_bgr(im)


def object_mask(tf):
    """Footprint of the selected object, recovered from the frames themselves: the pixels that
    change between the un-edited render and the fully-removed one are exactly where it stood."""
    d = np.abs(tf[0].astype(np.float32) - tf[12].astype(np.float32)).mean(-1)
    m = (d > 10).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > 800:
            keep[lab == i] = 1
    return keep


def dashed_contour(img, mask_small, seg=18, on=10, color=(60, 220, 60), width=3):
    """Green dashed selection outline, matching the paper figures."""
    m = cv2.resize(mask_small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    cont, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = img.copy()
    for c in cont:
        pts = c[:, 0, :]
        for k in range(0, len(pts), seg):
            piece = pts[k:k + on]
            if len(piece) > 1:
                cv2.polylines(out, [piece.reshape(-1, 1, 2)], False, color, width, cv2.LINE_AA)
    return out


def caption_bar(fr, headline, sub, banner):
    im = fr.copy()
    cv2.rectangle(im, (0, H - 130), (W, H), (17, 13, 10), -1)
    cv2.rectangle(im, (0, H - 130), (W, H - 126), (20, 255, 57), -1)
    pim = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pim)
    d.text((40, H - 112), headline, font=font(1, 36), fill=WHITE)
    d.text((40, H - 58), sub, font=font(0, 28), fill=MUTE)
    d.text((W - 470, 20), f"{banner} · real surgical stereo video", font=font(1, 24), fill=(230, 230, 230))
    return cv2.cvtColor(np.array(pim), cv2.COLOR_RGB2BGR)


def scene_segment(sc, hold_first=3, hold_last=4):
    tf = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(sc["frames"], "f*.png")))]
    assert len(tf) == 21, f"expected 21 frames in {sc['frames']}, got {len(tf)}"
    up = [cv2.resize(f, (W, H), interpolation=cv2.INTER_CUBIC) for f in tf]
    b = sc["banner"]
    om = object_mask(tf)                       # the selected object's footprint
    seg = [caption_bar(dashed_contour(up[0], om), f"Before the edit (frame {sc['edit']})",
                       "The selected object is outlined; what lies under it is hidden at this moment.",
                       b)] * (hold_first * FPS)
    # the object dissolves quickly: what is momentarily exposed underneath was never constrained
    # at this timestamp, so it is a transition rather than a result and should not linger
    for i, f in enumerate(up[1:13]):
        a = 1.0 - i / 11.0                      # outline fades out as the object dissolves
        fr = cv2.addWeighted(dashed_contour(f, om), a, f, 1.0 - a, 0) if a > 0.02 else f
        seg += [caption_bar(fr, "Removing the selected object",
                            "Its Gaussians are switched off at render time — no retraining.", b)] * 3
    for f in up[13:21]:
        seg += [caption_bar(f, "Showing the region at the time it was observed",
                            f"Frame {sc['ref']}, found automatically from the depth in the footprint.", b)] * 6
    seg += [caption_bar(up[20], "The surface underneath, recovered from the reconstruction",
                        "Verified next against the real frame.", b)] * (hold_last * FPS)
    return seg, tf


def main():
    vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    cap = cv2.VideoCapture(SRC)
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        vw.write(fr); n += 1
    cap.release()
    print(f"[v3] carried over {n} frames of the reviewed demo ({n/FPS:.1f} s)")

    added = 0
    for f in [card_intro()] * (8 * FPS):
        vw.write(f); added += 1
    for i, sc in enumerate(SCENES):
        seg, tf = scene_segment(sc)
        for f in seg:
            vw.write(f); added += 1
        x0, y0, x1, y1 = sc["zoom"]
        gt = cv2.imread(sc["gt"])
        card = card_verify(tf[20][y0:y1, x0:x1], gt[y0:y1, x0:x1], sc["db"], sc["ref"], sc["banner"])
        for f in [card] * (6 * FPS):
            vw.write(f); added += 1
    for f in [card_close()] * (8 * FPS):
        vw.write(f); added += 1
    vw.release()
    print(f"[v3] {OUT}: {n} original + {added} new = {n+added} frames ({(n+added)/FPS:.1f} s)")


if __name__ == "__main__":
    main()
