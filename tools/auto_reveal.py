"""Automatic click-to-reveal (v3): click an object in one frame, see what is beneath it.

The v2 pipeline (tools/stir_object_reveal.py) needed a hand-drawn region hint, a hand-picked
reference frame, and a 2D composite between two renders. This tool automates all three:

  Stage 1  click -> object: flood fill on the rendered depth map from the click (the object's rim
           is a depth cliff), then the near-depth Otsu cluster inside that support = the object's
           Gaussians; exact silhouette from their alpha-blended contribution.
  Stage 2  the system detects the BASE FRAMES by itself: it tracks how much of the object's
           footprint the object still covers over the sequence, and the frames where that coverage
           collapses are the ones where the region stands exposed. Reconstruction then draws only
           on those frames -- each occluded Gaussian takes its best-observed timestamp restricted
           to the detected base set. No frame is named by hand anywhere.
  Stage 3  single-pass reveal: the deformation network is evaluated at the edit time plus a few
           bucketed best-observed times (the node graph collapses time to a scalar per call --
           scene/node_deformation.py:299 -- so per-row mixed times must be assembled outside);
           under-object Gaussians take their observed-time state, the object's rows are removed,
           and ONE rasterizer pass renders the reveal. No image-space compositing anywhere.
  Stage 4  honesty: hole pixels dominated by never-observed Gaussians are flagged in a diagnostic.

  python3 tools/auto_reveal.py --model_path output/stir/seq33_6k --configs arguments/stir/seq33_6k.py \
      --iteration 6000 --frame 8 --click 300 180 --out output/stir/seq33_6k/autoreveal
"""
import os, sys
from argparse import ArgumentParser
from collections import deque
import torch
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render
from tools.endoscapes_peel import to8, label


def psnr_np(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))


class FixedDeform:
    """Stands in for gaussians._deformation during one render: returns pre-assembled outputs.
    Every other attribute is delegated to the real module -- the render path also reaches into
    `_deformation.deformation_net` (node bindings), which must keep working."""
    def __init__(self, orig, outs):
        object.__setattr__(self, "_orig", orig)
        object.__setattr__(self, "outs", outs)

    def __call__(self, *a, **k):
        return self.outs

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_orig"), name)


@torch.no_grad()
def eval_deform(gaussians, cam):
    """Evaluate the deformation exactly as render() does; returns per-row (pts, scales, rots, opac)
    for the _deformation_table subset, plus the projection helpers for the FULL set."""
    means3D = gaussians.get_xyz
    time = torch.tensor(cam.time).to(means3D.device).repeat(means3D.shape[0], 1)
    dp = gaussians._deformation_table
    binding_idx = binding_w = None
    if getattr(gaussians, "use_node_graph", False) and gaussians.node_seeded:
        if gaussians._binding_idx.shape[0] != means3D.shape[0]:
            gaussians.compute_node_bindings()
        binding_idx = gaussians._binding_idx.to(means3D.device)[dp]
        binding_w = gaussians._binding_w.to(means3D.device)[dp]
    outs = gaussians._deformation(means3D[dp], gaussians._scaling[dp], gaussians._rotation[dp],
                                  gaussians._opacity[dp], time[dp], binding_idx, binding_w)
    full = means3D.clone()
    full[dp] = outs[0]
    return outs, full


def project(pts, cam):
    ph = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
    clip = ph @ cam.full_proj_transform.to(pts.device)
    ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-7)
    sx = (ndc[:, 0] * .5 + .5) * cam.image_width
    sy = (ndc[:, 1] * .5 + .5) * cam.image_height
    vz = (ph @ cam.world_view_transform.to(pts.device))[:, 2]
    return sx, sy, vz


@torch.no_grad()
def visible_at(gaussians, cam, pipe, bg, tol=0.05):
    depth = render(cam, gaussians, pipe, bg, stage="fine")["depth"]
    depth = depth[0] if depth.dim() == 3 else depth
    H, W = depth.shape[-2], depth.shape[-1]
    _, full = eval_deform(gaussians, cam)
    sx, sy, vz = project(full, cam)
    xi = sx.long().clamp(0, W - 1)
    yi = sy.long().clamp(0, H - 1)
    surf = depth[yi, xi]
    inside = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H) & (surf > 0)
    return inside & ((vz - surf).abs() <= tol * surf.clamp_min(1e-6))


def otsu_split(z, bins=64, min_side=0.05):
    """Threshold maximising between-class variance; both sides must hold >= min_side of the mass."""
    z = np.asarray(z, np.float64)
    hist, edges = np.histogram(z, bins=bins)
    w = hist.astype(np.float64); tot = w.sum()
    mids = 0.5 * (edges[:-1] + edges[1:])
    best, thr = -1.0, float(np.median(z))
    for k in range(1, bins):
        w0, w1 = w[:k].sum(), w[k:].sum()
        if w0 < tot * min_side or w1 < tot * min_side:
            continue
        m0 = (w[:k] * mids[:k]).sum() / w0
        m1 = (w[k:] * mids[k:]).sum() / w1
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best:
            best, thr = v, float(edges[k])
    return thr


def flood_fill_depth(depth, cx, cy, grad_thresh, max_frac):
    """Local growth across smooth depth. Used only as a refinement: on its own it leaks wherever
    the object rests on the surface behind it (the contact rim is locally smooth)."""
    H, W = depth.shape
    seen = np.zeros((H, W), np.uint8)
    if depth[cy, cx] <= 0:
        return seen
    q = deque([(cy, cx)])
    seen[cy, cx] = 1
    count, cap = 1, int(max_frac * H * W)
    while q and count < cap:
        y, x = q.popleft()
        d0 = depth[y, x]
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx]:
                dn = depth[ny, nx]
                if dn > 0 and abs(dn - d0) < grad_thresh:
                    seen[ny, nx] = 1
                    q.append((ny, nx))
                    count += 1
    return seen


def segment_object_2d(depth, cx, cy, grad_thresh, max_frac):
    """Click -> object support. The depth cliff is defined GLOBALLY (the frame's depth histogram is
    bimodal: object in front, surface behind), then the object is the connected component of the
    near layer containing the click. A purely local flood fill leaks across the contact rim, so it
    is used only to refine an oversized component."""
    valid = depth > 0
    thr = otsu_split(depth[valid])
    seed_d = float(depth[cy, cx])
    layer = (valid & (depth < thr)) if seed_d < thr else (valid & (depth >= thr))
    layer = cv2.morphologyEx(layer.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    layer = cv2.morphologyEx(layer, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    nl, lab = cv2.connectedComponents(layer, 8)
    if lab[cy, cx] == 0:                      # click fell in a gap: take the nearest labelled pixel
        ys, xs = np.where(lab > 0)
        k = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        comp = (lab == lab[ys[k], xs[k]]).astype(np.uint8)
    else:
        comp = (lab == lab[cy, cx]).astype(np.uint8)
    print(f"[stage1] depth layer split at {thr:.1f}mm (seed {seed_d:.1f}mm) -> "
          f"component {100*comp.mean():.1f}% of frame")
    if comp.mean() > max_frac:                # still too big: refine locally inside the component
        comp = flood_fill_depth(np.where(comp > 0, depth, 0), cx, cy, grad_thresh, max_frac)
        print(f"[stage1] refined by local growth -> {100*comp.mean():.1f}% of frame")
    return cv2.morphologyEx(comp, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


@torch.no_grad()
def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=6000, type=int)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--click", type=int, nargs=2, default=None, metavar=("X", "Y"))
    parser.add_argument("--obj_mask", type=str, default=None,
                        help="use this mask as the object support instead of click segmentation")
    parser.add_argument("--grad_thresh", type=float, default=2.5, help="flood-fill depth step (mm)")
    parser.add_argument("--max_frac", type=float, default=0.45)
    parser.add_argument("--scan_step", type=int, default=2)
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--buckets", type=int, default=8)
    parser.add_argument("--base_cov", type=float, default=0.20,
                        help="a frame is a BASE frame when the object still covers less than this "
                             "fraction of its own footprint there")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--tag", type=str, default="")
    args = get_combined_args(parser)
    for k in ("rect", "px", "py", "mask", "click", "obj_mask"):
        if not hasattr(args, k):
            setattr(args, k, None)
    if args.configs:
        from utils.config_loader import load_config
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, load_config(args.configs))

    dataset, pipe, hyper = model.extract(args), pipeline.extract(args), hp.extract(args)
    os.makedirs(args.out, exist_ok=True)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    vcams = scene.getVideoCameras()
    cam = vcams[min(args.frame, len(vcams) - 1)]
    H, W = cam.image_height, cam.image_width
    # ================= Stage 1: click (or mask) -> object =================
    depth = render(cam, gaussians, pipe, bg, stage="fine")["depth"]
    depth = (depth[0] if depth.dim() == 3 else depth).cpu().numpy()
    if args.obj_mask:
        m = cv2.imread(args.obj_mask, 0)
        if m is None:
            sys.exit(f"ERROR: could not read {args.obj_mask}")
        support = (cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
        ys, xs = np.where(support > 0)
        cx, cy = (int(xs.mean()), int(ys.mean())) if args.click is None else args.click
        print(f"[stage1] object support from mask {os.path.basename(args.obj_mask)}")
    else:
        if args.click is None:
            sys.exit("ERROR: give either --click X Y or --obj_mask PATH")
        cx, cy = args.click
        support = segment_object_2d(depth, cx, cy, args.grad_thresh, args.max_frac)
    frac = support.mean()
    if support.sum() < 500:
        sys.exit(f"ERROR: object support too small ({int(support.sum())} px) -- check the click")
    print(f"[stage1] click ({cx},{cy}) depth {depth[cy,cx]:.1f}mm -> support {100*frac:.1f}% of frame")

    outs_e, full_e = eval_deform(gaussians, cam)
    sx, sy, vz = project(full_e, cam)
    xi = sx.long().clamp(0, W - 1)
    yi = sy.long().clamp(0, H - 1)
    sup_t = torch.from_numpy(support).to(sx.device)
    in_sup = sup_t[yi, xi] > 0
    z = vz[in_sup].cpu().numpy()
    z = z[(z > 1) & (z < np.percentile(z, 99.5))]
    hist, edges = np.histogram(z, bins=64)
    w = hist.astype(np.float64); tot = w.sum()
    mids = 0.5 * (edges[:-1] + edges[1:])
    best, thr = -1.0, float(np.median(z))
    for k in range(1, 64):
        w0, w1 = w[:k].sum(), w[k:].sum()
        if w0 < tot * 0.05 or w1 < tot * 0.05:
            continue
        m0 = (w[:k] * mids[:k]).sum() / w0
        m1 = (w[k:] * mids[k:]).sum() / w1
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best:
            best, thr = v, float(edges[k])
    thr -= 8.0
    sel = in_sup & (vz < thr)
    n = full_e.shape[0]
    print(f"[stage1] depth split {thr:.1f}mm -> object = {int(sel.sum())} Gaussians")

    ocol = torch.zeros(n, 3, device="cuda"); ocol[sel] = 1.0
    omap = render(cam, gaussians, pipe, bg, override_color=ocol, stage="fine")["render"][0]
    sil = (omap > 0.5).float().cpu().numpy().astype(np.uint8)
    sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    sil_t = torch.from_numpy(sil).to(sx.device)

    # ================= Stage 2: detect the base frames, then use only those =================
    scan = list(range(0, len(vcams), args.scan_step))
    # Base frames are detected from DEPTH inside the footprint, not from the object's Gaussians:
    # the labelled Gaussians barely travel (the field recycles primitives instead of transporting
    # them), so they always overlap the footprint and can never signal "the object has left".
    # Depth can: the region reads shallow while the object sits on it and steps back once it is
    # gone. Otsu splits that curve into covered / exposed.
    sup_t = torch.from_numpy(support).to("cuda") > 0
    vis = torch.zeros(n, len(scan), dtype=torch.bool, device="cuda")
    dmed = []
    for j, s in enumerate(scan):
        vis[:, j] = visible_at(gaussians, vcams[s], pipe, bg, args.tol)
        d = render(vcams[s], gaussians, pipe, bg, stage="fine")["depth"]
        d = d[0] if d.dim() == 3 else d
        dv = d[sup_t & (d > 0)]
        dmed.append(float(dv.median()) if dv.numel() else 0.0)
    dmed = np.array(dmed)
    seen_cnt = vis.sum(1)

    dthr = otsu_split(dmed[dmed > 0])
    base_j = np.where(dmed > dthr)[0]                 # deeper than the split => region exposed
    if base_j.size == 0:
        base_j = np.argsort(-dmed)[:max(1, len(scan) // 10)]
    base_frames = [scan[j] for j in base_j]
    print(f"[stage2] footprint depth over time: {dmed.min():.0f}-{dmed.max():.0f}mm, "
          f"exposed when > {dthr:.0f}mm | detected {len(base_frames)} BASE frames: "
          f"{sorted(base_frames)[:14]}{' ...' if len(base_frames) > 14 else ''}")

    # each Gaussian's best-observed time, restricted to the detected base frames
    win = torch.nn.functional.conv1d(vis.float().unsqueeze(1),
                                     torch.ones(1, 1, 5, device="cuda"), padding=2).squeeze(1)
    bj = torch.from_numpy(base_j).to("cuda")
    win_base = win[:, bj]
    best_in_base = bj[win_base.argmax(1)]
    has_base = win_base.max(1).values > 0
    best_j = torch.where(has_base, best_in_base, win.argmax(1))
    print(f"[stage2] never-observed: {int((seen_cnt==0).sum())}/{n} | "
          f"{int(has_base.sum())} Gaussians observed within the base frames")

    # ================= Stage 3: assemble per-Gaussian states, render once =================
    # bucket the candidates' best times
    qs = torch.linspace(0, 1, args.buckets + 1, device="cuda")[1:-1]
    cuts = torch.quantile(best_j.float(), qs)
    bucket_of = torch.bucketize(best_j.float(), cuts)
    bucket_frames = []
    for b in range(args.buckets):
        rows = best_j[bucket_of == b]
        bucket_frames.append(int(scan[int(rows.float().median())]) if rows.numel() else int(scan[len(scan)//2]))

    dp = gaussians._deformation_table
    pts_f, scl_f, rot_f, opa_f = [t.clone() for t in outs_e]
    candidate_sub = None
    cand_full = torch.zeros(n, dtype=torch.bool, device="cuda")
    for b in range(args.buckets):
        fb = bucket_frames[b]
        outs_b, full_b = eval_deform(gaussians, vcams[fb])
        bx, by, _ = project(full_b, vcams[fb])
        bxi = bx.long().clamp(0, W - 1)
        byi = by.long().clamp(0, H - 1)
        inside_sil_at_best = sil_t[byi, bxi] > 0
        cand = (~sel) & inside_sil_at_best & (seen_cnt > 0) & (bucket_of == b)
        cand_full |= cand
        cs = cand[dp]
        pts_f[cs] = outs_b[0][cs]
        scl_f[cs] = outs_b[1][cs]
        rot_f[cs] = outs_b[2][cs]
        opa_f[cs] = outs_b[3][cs]
    print(f"[stage3] under-object candidates re-timed: {int(cand_full.sum())} over buckets "
          f"{sorted(set(bucket_frames))}")

    # verification frame: the base frame that exposes the most of the revealed content
    cov_c = vis[cand_full][:, bj].sum(0)
    auto_ref = int(scan[int(bj[int(cov_c.argmax())])])
    print(f"[stage2] verification frame (auto) = {auto_ref}, a detected base frame covering "
          f"{int(cov_c.max())}/{int(cand_full.sum())} revealed Gaussians")

    orig_deform = gaussians._deformation
    orig_opac = gaussians._opacity.clone()
    before = to8(render(cam, gaussians, pipe, bg, stage="fine")["render"])

    gaussians._deformation = FixedDeform(orig_deform, (pts_f, scl_f, rot_f, opa_f))
    gaussians._opacity[sel] = -50.0
    reveal = to8(render(cam, gaussians, pipe, bg, stage="fine")["render"])

    # progressive dissolve for the video (bed already holds observed-time state throughout)
    frames = [before]
    for s in range(1, args.steps + 1):
        gaussians._opacity.copy_(orig_opac)
        gaussians._opacity[sel] = orig_opac[sel] - 8.0 * s / args.steps
        if s == args.steps:
            gaussians._opacity[sel] = -50.0
        frames.append(to8(render(cam, gaussians, pipe, bg, stage="fine")["render"]))
    gaussians._deformation = orig_deform
    gaussians._opacity.copy_(orig_opac)

    # ================= Stage 4: honesty diagnostic =================
    ucol = torch.zeros(n, 3, device="cuda")
    ucol[(seen_cnt == 0) & (~sel)] = 1.0
    gaussians._deformation = FixedDeform(orig_deform, (pts_f, scl_f, rot_f, opa_f))
    gaussians._opacity[sel] = -50.0
    umap = render(cam, gaussians, pipe, bg, override_color=ucol, stage="fine")["render"][0]
    gaussians._deformation = orig_deform
    gaussians._opacity.copy_(orig_opac)
    unrel = ((umap > 0.3).float().cpu().numpy() * (sil > 0)).astype(np.uint8)
    diag = reveal.copy(); diag[unrel > 0] = (diag[unrel > 0] * 0.15).astype(np.uint8)

    # ================= outputs =================
    gt_ref = (vcams[auto_ref].original_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    hole = sil > 0
    p_rev = psnr_np(reveal[hole], gt_ref[hole])

    before_marked = before.copy()
    cont, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cont:
        pts = c[:, 0, :]
        for k in range(0, len(pts), 14):
            seg = pts[k:k + 8]
            if len(seg) > 1:
                cv2.polylines(before_marked, [seg.reshape(-1, 1, 2)], False, (60, 220, 60), 2, cv2.LINE_AA)
    cv2.drawMarker(before_marked, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 22, 3)

    panel = np.hstack([
        label(before_marked, f"before (frame {args.frame}); one click -> object segmented"),
        label(reveal, f"object removed -- beneath revealed in one render | {p_rev:.1f} dB"),
        label(gt_ref, f"GT frame {auto_ref} (auto-detected base frame)", (60, 255, 60)),
    ])
    tag = args.tag or f"f{args.frame}_click{cx}x{cy}"
    out_png = os.path.join(args.out, f"autoreveal_{tag}.png")
    cv2.imwrite(out_png, panel[:, :, ::-1])
    cv2.imwrite(os.path.join(args.out, f"autoreveal_{tag}_diag_unreliable.png"), diag[:, :, ::-1])
    fdir = os.path.join(args.out, f"frames_{tag}")
    os.makedirs(fdir, exist_ok=True)
    for i, im in enumerate(frames):
        cv2.imwrite(os.path.join(fdir, f"f{i:03d}.png"), im[:, :, ::-1])
    print(f"[autoreveal] object {int(sel.sum())} | candidates {int(cand_full.sum())} | "
          f"auto_ref {auto_ref} | hole PSNR {p_rev:.2f} dB -> {out_png}")


if __name__ == "__main__":
    main()
