"""Bake a layered reconstruction: store the "what is underneath" knowledge IN the model file.

Motivation. The reveal tools (stir_object_reveal.py, auto_reveal.py) must look up another
timestamp at render time, because the deformation field only places occluded Gaussians correctly
at the timestamps where the camera saw them. This tool removes that lookup permanently: it writes a
new model in which the scene is split into layers and the base layer is already correct at EVERY
timestamp, so hiding the top layer directly shows what is beneath -- one render, any frame, no
search for a similar frame.

How the baking works (no retraining, no new machinery):
  * layer 1 = the object (selected by one click, same flood-fill + depth-cluster as auto_reveal);
    layer 0 = everything else, i.e. the base/bed layer.
  * every layer-0 Gaussian is evaluated at ITS best-observed timestamp, and that deformed state
    (position, pre-activation scale, rotation) is written into its CANONICAL parameters;
  * its `_deformation_table` entry is set False. The renderer leaves such Gaussians completely
    undeformed -- `means3D_final[~deformation_point] = means3D[~deformation_point]`
    (gaussian_renderer/__init__.py) -- so the baked state is what the rasterizer sees at every t.
  * `layer_id.pth` (0/1 per Gaussian) is saved next to the checkpoint, so any renderer/viewer can
    hide a layer by zeroing its opacity.
This is sound because the patch under the object is static in these scenes (only the organ
travels); a base surface that genuinely deforms under the object would need its frozen patch
expressed in the deformation field's frame -- see the note at the bottom.

Colour needs no baking (spherical harmonics are canonical), and opacity is already time-invariant
in this renderer (it activates pc._opacity directly).

  python3 tools/bake_layers.py --model_path output/stir/seq33_6k --configs arguments/stir/seq33_6k.py \
      --iteration 6000 --frame 8 --click 300 180 --out_model output/stir/seq33_6k_layered
"""
import os, sys, shutil
from argparse import ArgumentParser
import torch
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render
from tools.endoscapes_peel import to8, label
from tools.auto_reveal import eval_deform, project, visible_at, segment_object_2d, otsu_split


@torch.no_grad()
def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=6000, type=int)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--click", type=int, nargs=2, default=None, metavar=("X", "Y"))
    parser.add_argument("--obj_mask", type=str, default=None,
                        help="use this mask as the object support instead of click segmentation")
    parser.add_argument("--grad_thresh", type=float, default=2.5)
    parser.add_argument("--max_frac", type=float, default=0.45)
    parser.add_argument("--scan_step", type=int, default=2)
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--buckets", type=int, default=8)
    parser.add_argument("--out_model", type=str, required=True)
    args = get_combined_args(parser)
    for k in ("rect", "px", "py", "mask", "click", "obj_mask"):
        if not hasattr(args, k):
            setattr(args, k, None)
    if args.configs:
        from utils.config_loader import load_config
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, load_config(args.configs))

    dataset, pipe, hyper = model.extract(args), pipeline.extract(args), hp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    vcams = scene.getVideoCameras()
    cam = vcams[min(args.frame, len(vcams) - 1)]
    H, W = cam.image_height, cam.image_width
    # ---- layer 1: the clicked (or masked) object -------------------------------------------
    depth = render(cam, gaussians, pipe, bg, stage="fine")["depth"]
    depth = (depth[0] if depth.dim() == 3 else depth).cpu().numpy()
    if args.obj_mask:
        m = cv2.imread(args.obj_mask, 0)
        if m is None:
            sys.exit(f"ERROR: could not read {args.obj_mask}")
        support = (cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
        print(f"[bake] object support from mask {os.path.basename(args.obj_mask)}")
    else:
        if args.click is None:
            sys.exit("ERROR: give either --click X Y or --obj_mask PATH")
        cx, cy = args.click
        support = segment_object_2d(depth, cx, cy, args.grad_thresh, args.max_frac)
    outs_e, full_e = eval_deform(gaussians, cam)
    sx, sy, vz = project(full_e, cam)
    xi = sx.long().clamp(0, W - 1); yi = sy.long().clamp(0, H - 1)
    in_sup = torch.from_numpy(support).to(sx.device)[yi, xi] > 0
    z = vz[in_sup].cpu().numpy(); z = z[(z > 1) & (z < np.percentile(z, 99.5))]
    thr = otsu_split(z) - 8.0
    layer1 = in_sup & (vz < thr)
    n = full_e.shape[0]
    print(f"[bake] layer 1 (object) = {int(layer1.sum())}/{n} Gaussians")

    # ---- per-Gaussian best-observed timestamp ----------------------------------------------
    scan = list(range(0, len(vcams), args.scan_step))
    vis = torch.zeros(n, len(scan), dtype=torch.bool, device="cuda")
    for j, s in enumerate(scan):
        vis[:, j] = visible_at(gaussians, vcams[s], pipe, bg, args.tol)
    seen_cnt = vis.sum(1)
    win = torch.nn.functional.conv1d(vis.float().unsqueeze(1),
                                     torch.ones(1, 1, 5, device="cuda"), padding=2).squeeze(1)
    best_j = win.argmax(1)
    print(f"[bake] visibility scan over {len(scan)} frames | never observed: {int((seen_cnt==0).sum())}")

    # ---- bake layer 0 into canonical parameters at its best-observed time -------------------
    dp = gaussians._deformation_table.clone()
    xyz_new = gaussians._xyz.detach().clone()
    scl_new = gaussians._scaling.detach().clone()
    rot_new = gaussians._rotation.detach().clone()
    qs = torch.linspace(0, 1, args.buckets + 1, device="cuda")[1:-1]
    cuts = torch.quantile(best_j.float(), qs)
    bucket_of = torch.bucketize(best_j.float(), cuts)
    baked = torch.zeros(n, dtype=torch.bool, device="cuda")
    sup_t = torch.from_numpy(support).to(vz.device)
    for b in range(args.buckets):
        rows = best_j[bucket_of == b]
        if rows.numel() == 0:
            continue
        fb = int(scan[int(rows.float().median())])
        outs_b, full_b = eval_deform(gaussians, vcams[fb])
        # under-object test at the timestamp where the Gaussian was actually observed (its state
        # at the edit time is exactly the unreliable thing we are fixing)
        bx, by, _ = project(full_b, vcams[fb])
        bxi = bx.long().clamp(0, W - 1); byi = by.long().clamp(0, H - 1)
        under = sup_t[byi, bxi] > 0
        tgt = (~layer1) & under & (seen_cnt > 0) & (bucket_of == b) & dp
        sub = tgt[dp]
        xyz_new[tgt] = outs_b[0][sub]
        scl_new[tgt] = outs_b[1][sub]
        rot_new[tgt] = outs_b[2][sub]
        baked |= tgt
        print(f"[bake]   bucket {b}: {int(tgt.sum())} under-object Gaussians frozen at frame {fb}")

    gaussians._xyz.data.copy_(xyz_new)
    gaussians._scaling.data.copy_(scl_new)
    gaussians._rotation.data.copy_(rot_new)
    table_new = dp.clone(); table_new[baked] = False        # baked rows: never deformed again
    gaussians._deformation_table = table_new
    # Cull the unreliable Gaussians. A Gaussian never on the visible surface at ANY timestamp was
    # never constrained by a photograph; at the edit time such Gaussians drift into the space under
    # the object and are exactly the sparkle that swamps the revealed bed. They carry no verifiable
    # content anywhere, so the baked model retires them permanently (opacity -> 0).
    junk = (seen_cnt == 0) & (~layer1)
    gaussians._opacity.data[junk] = -50.0
    layer_id = torch.where(layer1, torch.ones_like(seen_cnt),
                torch.where(baked, torch.full_like(seen_cnt, 2),
                 torch.where(junk, torch.full_like(seen_cnt, 3), torch.zeros_like(seen_cnt))))
    print(f"[bake] culled {int(junk.sum())} never-observed Gaussians (layer 3)")
    # 0 = scene, 1 = object, 2 = frozen patch beneath, 3 = culled
    print(f"[bake] froze {int(baked.sum())} under-object Gaussians (layer 2); "
          f"{int(table_new.sum())} of {n} remain dynamic")

    # ---- save the layered model -------------------------------------------------------------
    pcd = os.path.join(args.out_model, "point_cloud", f"iteration_{args.iteration}")
    os.makedirs(pcd, exist_ok=True)
    gaussians.save_ply(os.path.join(pcd, "point_cloud.ply"))
    gaussians.save_deformation(pcd)
    torch.save(layer_id, os.path.join(pcd, "layer_id.pth"))
    src_cfg = os.path.join(args.model_path, "cfg_args")
    if os.path.exists(src_cfg):
        shutil.copy(src_cfg, os.path.join(args.out_model, "cfg_args"))
    print(f"[bake] wrote layered model -> {pcd} (+ layer_id.pth)")

    # ---- verification: hide layer 1 at several timestamps, no lookup anywhere ---------------
    orig_opac = gaussians._opacity.detach().clone()
    tiles = []
    for f in np.linspace(0, len(vcams) - 1, 5).astype(int):
        c = vcams[int(f)]
        gaussians._opacity.data.copy_(orig_opac)
        base_img = to8(render(c, gaussians, pipe, bg, stage="fine")["render"])
        gaussians._opacity.data[layer1] = -50.0
        hid = to8(render(c, gaussians, pipe, bg, stage="fine")["render"])
        gaussians._opacity.data.copy_(orig_opac)
        tiles.append(np.vstack([label(base_img, f"frame {f}: both layers"),
                                label(hid, f"frame {f}: layer 1 hidden -> beneath")]))
    sheet = np.hstack(tiles)
    vp = os.path.join(args.out_model, "layer_verification.png")
    cv2.imwrite(vp, sheet[:, :, ::-1])
    print(f"[bake] verification sheet -> {vp}")


if __name__ == "__main__":
    main()

# Note on deforming base layers: freezing every observed base Gaussian is correct only when the
# base is static. For a base that deforms, freeze just the Gaussians occluded by the object and
# keep the rest dynamic -- their canonical state must then be expressed in the deformation field's
# frame, otherwise the frozen patch tears away from its moving neighbours. That is the general
# version of this idea and is left as future work.
