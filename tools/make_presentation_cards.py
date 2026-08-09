"""Render presentation slide cards for the GC-EndoGaussian demo video.

Produces 1280x1024 title / definition / caption / results cards (dark research-talk
style) that frame the raw demo footage so the video explains *what the paper does* and
*why it is strong*, not just a sequence of renders. CPU only (PIL). Numbers match the
paper (docs/paper_v2.md). Output PNGs -> <out>/{title,what,editcap,results}.png
"""
import os, argparse
from PIL import Image, ImageDraw, ImageFont

FDIR = "/usr/share/fonts/dejavu-sans-fonts"
W, H = 1280, 1024
BG = (13, 17, 23)
WHITE = (236, 239, 241)
MUTE = (150, 162, 173)
GREEN = (57, 255, 20)
CYAN = (0, 209, 255)
MARGIN = 112


def font(bold, size):
    p = os.path.join(FDIR, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    if not os.path.exists(p):
        p = os.path.join(FDIR, "DejaVuSans-Bold.ttf")
    return ImageFont.truetype(p, size)


def tw(draw, s, f):
    b = draw.textbbox((0, 0), s, font=f)
    return b[2] - b[0]


def wrap(draw, text, f, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if tw(draw, t, f) <= maxw:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def base(accent=GREEN):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 14, H], fill=accent)                 # left accent spine
    d.rectangle([0, H - 8, W, H], fill=accent)              # bottom accent
    return im, d


def draw_segments(d, x, y, segments, f):
    for text, color in segments:
        d.text((x, y), text, font=f, fill=color)
        x += tw(d, text, f)


def bullets(d, items, y0, f, fb, gap=118, dot=GREEN):
    y = y0
    for it in items:
        d.text((MARGIN, y + 6), "▸", font=fb, fill=dot)
        x0 = MARGIN + 46
        if isinstance(it, list):                            # rich line: [(text,color),...]
            draw_segments(d, x0, y, it, f)
            y += gap
        else:
            for i, ln in enumerate(wrap(d, it, f, W - x0 - MARGIN)):
                d.text((x0, y), ln, font=f, fill=WHITE)
                y += 52
            y += gap - 52
    return y


def card_title(out):
    im, d = base()
    d.text((MARGIN, 150), "Surgical Vision  ·  4D Gaussian Splatting", font=font(0, 30), fill=CYAN)
    d.text((MARGIN, 236), "GC-EndoGaussian", font=font(1, 96), fill=WHITE)
    st = font(1, 40)
    d.text((MARGIN, 372), "Residual-Centered Sparse Control for", font=st, fill=WHITE)
    d.text((MARGIN, 424), "Editable Endoscopic 4D Gaussian Reconstruction", font=st, fill=WHITE)
    d.rectangle([MARGIN, 520, MARGIN + 620, 524], fill=GREEN)
    body = font(0, 34)
    d.text((MARGIN, 570), "Reconstruct deforming surgical tissue in 4D —", font=body, fill=MUTE)
    d.text((MARGIN, 616), "then edit it at inference time, with no retraining.", font=body, fill=MUTE)
    im.save(os.path.join(out, "title.png"))


def card_what(out):
    im, d = base()
    d.text((MARGIN, 110), "What this paper does", font=font(1, 58), fill=WHITE)
    d.rectangle([MARGIN, 192, MARGIN + 380, 196], fill=GREEN)
    f, fb = font(0, 33), font(1, 33)
    bullets(d, [
        "Reconstructs deforming tissue from endoscopic video in 4D (EndoGaussian dense-deformation base).",
        "Adds a sparse control-node layer (SC-GS-style): drag nodes to edit the tissue at inference — no retraining.",
        "Key insight: a dense per-Gaussian residual — not the graph — is what preserves reconstruction fidelity.",
        "Net effect: editing handles at near-zero cost to reconstruction quality.",
    ], 244, f, fb, gap=126)
    im.save(os.path.join(out, "what.png"))


def card_editcap(out):
    im, d = base()
    d.text((MARGIN, 150), "Inference-time editing", font=font(1, 62), fill=GREEN)
    d.rectangle([MARGIN, 244, MARGIN + 420, 248], fill=GREEN)
    f = font(0, 40)
    lines = [
        "A local group of control nodes is dragged laterally",
        "to retract the tissue and reveal the region behind it.",
        "",
        "Smooth · localized · no retraining —",
        "and the 4D scene keeps playing throughout.",
    ]
    y = 340
    for ln in lines:
        d.text((MARGIN, y), ln, font=f, fill=WHITE if ln and not ln.startswith("Smooth") else MUTE)
        y += 74 if ln else 40
    im.save(os.path.join(out, "editcap.png"))


def card_howread(out):
    im, d = base(CYAN)
    d.text((MARGIN, 110), "How to read the demo", font=font(1, 58), fill=WHITE)
    d.rectangle([MARGIN, 192, MARGIN + 380, 196], fill=CYAN)
    f, fb = font(0, 33), font(1, 33)
    bullets(d, [
        "Each pair: the real endoscope video (left) vs. our 4D reconstruction of the same instant (right).",
        "Matching tissue means a faithful reconstruction — the two should look the same.",
        "The surgical tool is masked out during training, so the reconstruction shows the tissue behind it (a tool-free view).",
        "Later panels show the pixel error vs. ground truth — darker is smaller error.",
    ], 244, f, fb, gap=126, dot=CYAN)
    im.save(os.path.join(out, "howread.png"))


def card_results(out):
    im, d = base()
    d.text((MARGIN, 108), "Why it's strong", font=font(1, 58), fill=WHITE)
    d.rectangle([MARGIN, 190, MARGIN + 300, 194], fill=GREEN)
    f, fb = font(0, 32), font(1, 32)
    bullets(d, [
        [("Editability nearly free: within ", WHITE), ("0.13–0.27 dB", GREEN), (" of baseline", WHITE)],
        [("205 FPS", GREEN), (" real-time  ·  only ", WHITE), ("+0.07%", GREEN), (" parameters", WHITE)],
        [("Tissue tracking on par with baseline  ", WHITE), ("(p = 0.73)", CYAN)],
        [("Residual isolation: ", WHITE), ("+0.49 dB", GREEN), (" PSNR, ", WHITE), ("7.02 → 3.41 px", GREEN), (" tracking", WHITE)],
    ], 256, f, fb, gap=124)
    d.text((MARGIN, 812), "A transferable recipe: the residual is the load-bearing ingredient,",
           font=font(0, 30), fill=MUTE)
    d.text((MARGIN, 852), "independent of the sparse-control architecture.", font=font(0, 30), fill=MUTE)
    im.save(os.path.join(out, "results.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/pres_cards")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    card_title(args.out); card_what(args.out); card_howread(args.out)
    card_editcap(args.out); card_results(args.out)
    print("cards ->", args.out, os.listdir(args.out))


if __name__ == "__main__":
    main()
