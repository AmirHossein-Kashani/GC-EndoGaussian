"""Extract the OBJECTS that live under a selected organ, as entities in their own right.

This is object-centric, not frame-centric: we do not look for a frame to copy a region from. We
ask which Gaussians constitute the surface inside the organ's footprint at the times when that
surface was actually visible, cluster them into instances, and render those instances on their own.
Each instance is frozen at its own observed state, so it is a standing object in the scene, not a
picture borrowed from a timestamp.

Feasibility is scene-dependent and measurable (tools/measure_recruitment.py): where the deformation
field recycles the organ's Gaussians to form the surface beneath it, no distinct under-object exists
(seq33: 87% shared). Where it does not (seq23: 13% shared), the under-objects are real and separable.

Outputs
  under_objects_only.png    the extracted objects rendered alone (black elsewhere)
  under_objects_labelled.png  the same, per-instance false colour
  before_after.png          organ present | organ removed, under-objects shown
  stats printed: instance count and size, shared-with-organ fraction

  python3 tools/extract_under_objects.py --model_path output/stir/seq23_6k_tm \
      --configs arguments/stir/seq23_6k_tm.py --iteration 6000 --frame 10 \
      --obj_mask data/endonerf/stir06_seq23/organ_mask_f10.png \
      --out output/stir/seq23_6k_tm/under_objects
"""
import os, sys
from argparse import ArgumentParser
import torch
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render
from tools.endoscapes_peel import to8, label
from tools.auto_reveal import eval_deform, project, visible_at, otsu_split


@torch.no_grad()
def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=6000, type=int)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--obj_mask", type=str, required=True)
    parser.add_argument("--scan_step", type=int, default=2)
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--buckets", type=int, default=8)
    parser.add_argument("--min_instance", type=int, default=200)
    parser.add_argument("--link_mm", type=float, default=4.0, help="clustering link radius")
    parser.add_argument("--out", type=str, required=True)
    args = get_combined_args(parser)
    for k in ("rect", "px", "py", "mask", "click"):
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

    mk = cv2.imread(args.obj_mask, 0)
    mk = (cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    mt = torch.from_numpy(mk).cuda()

    # ---- the organ (near-depth cluster inside its footprint at the edit frame) --------------
    outs_e, full_e = eval_deform(gaussians, cam)
    sx, sy, vz = project(full_e, cam)
    in_m = mt[sy.long().clamp(0, H - 1), sx.long().clamp(0, W - 1)] > 0
    z = vz[in_m].cpu().numpy(); z = z[(z > 1) & (z < np.percentile(z, 99.5))]
    thr = otsu_split(z) - 8.0
    organ = in_m & (vz < thr)
    n = full_e.shape[0]

    # ---- the under-objects: what forms the visible surface inside the footprint, when visible
    scan = list(range(0, len(vcams), args.scan_step))
    vis = torch.zeros(n, len(scan), dtype=torch.bool, device="cuda")
    inside_when_vis = torch.zeros(n, dtype=torch.bool, device="cuda")
    for j, s in enumerate(scan):
        v = visible_at(gaussians, vcams[s], pipe, bg, args.tol)
        vis[:, j] = v
        _, full_s = eval_deform(gaussians, vcams[s])
        px, py, _ = project(full_s, vcams[s])
        ins = mt[py.long().clamp(0, H - 1), px.long().clamp(0, W - 1)] > 0
        inside_when_vis |= (v & ins)
    seen_cnt = vis.sum(1)
    under = inside_when_vis & (~organ) & (seen_cnt > 0)
    shared = float((inside_when_vis & organ).sum()) / max(float(inside_when_vis.sum()), 1) * 100
    print(f"[under] organ {int(organ.sum())} | under-object set {int(under.sum())} | "
          f"shared with organ {shared:.1f}%")
    if int(under.sum()) < args.min_instance:
        sys.exit("ERROR: no separable under-object content in this scene")

    # ---- freeze each under-Gaussian at its own observed state -------------------------------
    win = torch.nn.functional.conv1d(vis.float().unsqueeze(1),
                                     torch.ones(1, 1, 5, device="cuda"), padding=2).squeeze(1)
    best_j = win.argmax(1)
    dp = gaussians._deformation_table
    xyz_new = gaussians._xyz.detach().clone()
    scl_new = gaussians._scaling.detach().clone()
    rot_new = gaussians._rotation.detach().clone()
    qs = torch.linspace(0, 1, args.buckets + 1, device="cuda")[1:-1]
    cuts = torch.quantile(best_j.float(), qs)
    bucket_of = torch.bucketize(best_j.float(), cuts)
    for b in range(args.buckets):
        rows = best_j[bucket_of == b]
        if rows.numel() == 0:
            continue
        fb = int(scan[int(rows.float().median())])
        outs_b, _ = eval_deform(gaussians, vcams[fb])
        tgt = under & (bucket_of == b) & dp
        sub = tgt[dp]
        xyz_new[tgt] = outs_b[0][sub]
        scl_new[tgt] = outs_b[1][sub]
        rot_new[tgt] = outs_b[2][sub]
    gaussians._xyz.data.copy_(xyz_new)
    gaussians._scaling.data.copy_(scl_new)
    gaussians._rotation.data.copy_(rot_new)
    tbl = gaussians._deformation_table.clone(); tbl[under] = False
    gaussians._deformation_table = tbl

    # ---- cluster the under set into instances (grid-linked connected components) ------------
    P = gaussians._xyz.detach()[under].cpu().numpy()
    idx_under = torch.nonzero(under).squeeze(1).cpu().numpy()
    cell = args.link_mm
    keys = np.floor(P / cell).astype(np.int64)
    lut = {}
    for i, k in enumerate(map(tuple, keys)):
        lut.setdefault(k, []).append(i)
    lab = -np.ones(len(P), np.int64); cur = 0
    for i in range(len(P)):
        if lab[i] >= 0:
            continue
        stack, lab[i] = [i], cur
        while stack:
            j = stack.pop()
            kj = keys[j]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for m in lut.get((kj[0]+dx, kj[1]+dy, kj[2]+dz), ()):
                            if lab[m] < 0 and np.linalg.norm(P[m] - P[j]) <= cell:
                                lab[m] = cur; stack.append(m)
        cur += 1
    sizes = np.bincount(lab)
    keep = [c for c in range(cur) if sizes[c] >= args.min_instance]
    print(f"[under] {cur} raw clusters -> {len(keep)} instances >= {args.min_instance} Gaussians: "
          f"{sorted(sizes[keep], reverse=True)[:8]}")

    # ---- renders ----------------------------------------------------------------------------
    orig_opac = gaussians._opacity.detach().clone()
    before = to8(render(cam, gaussians, pipe, bg, stage="fine")["render"])
    gaussians._opacity.data[organ] = -50.0
    after = to8(render(cam, gaussians, pipe, bg, stage="fine")["render"])
    gaussians._opacity.data.copy_(orig_opac)

    hide_all_but_under = torch.ones(n, dtype=torch.bool, device="cuda"); hide_all_but_under[under] = False
    gaussians._opacity.data[hide_all_but_under] = -50.0
    only = to8(render(cam, gaussians, pipe, bg, stage="fine")["render"])
    col = torch.zeros(n, 3, device="cuda")
    palette = [(1,0.35,0.2),(0.2,0.8,1),(0.5,1,0.3),(1,0.9,0.2),(0.9,0.4,1),(0.3,0.5,1)]
    for t, c in enumerate(keep):
        rgb = palette[t % len(palette)]
        rows = torch.from_numpy(idx_under[lab == c]).cuda()
        col[rows] = torch.tensor(rgb, device="cuda")
    labelled = to8(render(cam, gaussians, pipe, bg, override_color=col, stage="fine")["render"])
    gaussians._opacity.data.copy_(orig_opac)

    cv2.imwrite(os.path.join(args.out, "under_objects_only.png"),
                label(only, f"the objects under the organ ({int(under.sum())} gaussians)")[:, :, ::-1])
    cv2.imwrite(os.path.join(args.out, "under_objects_labelled.png"),
                label(labelled, f"{len(keep)} instances, false colour")[:, :, ::-1])
    ba = np.hstack([label(before, f"before: organ present (frame {args.frame})"),
                    label(after, "after: organ removed -- the objects beneath")])
    cv2.imwrite(os.path.join(args.out, "before_after.png"), ba[:, :, ::-1])
    print(f"[under] wrote renders -> {args.out}")


if __name__ == "__main__":
    main()
