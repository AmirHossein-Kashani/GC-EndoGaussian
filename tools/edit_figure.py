"""Generality edit-figure renderer for GC-EndoGaussian.

Loads a trained match-mode model and applies inference-time control-node drags in
several directions, over a small magnitude sweep, re-rendering the same view each
time. Also exports the 2D pixel projection of the dragged control nodes + the drag
vector so a drag-handle overlay can be composited offline (no GPU needed there).

Outputs, under <out>/:
  before.png
  after_<dir>_<mag>.png            one per (direction, magnitude)
  meta.json                        {W,H, per-dir: node px/py, centroid, arrow tip}

Usage (see run_edit_figure.bash):
  python tools/edit_figure.py --model_path output/endonerf/pulling_match \
      --configs arguments/endonerf/pulling_graph_match.py --iteration 6000 \
      --mags 0.08,0.12,0.16,0.20 --dirs +y,-y,+x,-x \
      --radius_frac 0.14 --cam 0 --out output/endonerf/pulling_match/edit_fig
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
import torch
import torchvision
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render

AXIS = {"x": 0, "y": 1, "z": 2}


def project(pts, cam):
    """World points (M,3) -> pixel (px,py) and clip-w (for in-front test), 3DGS convention."""
    ones = torch.ones(pts.shape[0], 1, device=pts.device, dtype=pts.dtype)
    p_hom = torch.cat([pts, ones], dim=-1)                      # (M,4)
    p_clip = p_hom @ cam.full_proj_transform.to(pts.device)    # (M,4)
    w = p_clip[:, 3:4].clamp_min(1e-6)
    ndc = p_clip[:, :3] / w
    px = (ndc[:, 0] * 0.5 + 0.5) * cam.image_width
    py = (ndc[:, 1] * 0.5 + 0.5) * cam.image_height
    return px, py, p_clip[:, 3]


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=6000, type=int)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--mags", type=str, default="0.08,0.12,0.16,0.20")   # fractions of node-cloud extent
    parser.add_argument("--dirs", type=str, default="+y,-y,+x,-x")           # signed axes to drag along
    parser.add_argument("--radius_frac", type=float, default=0.14)
    parser.add_argument("--region", type=str, default="median",
                        choices=["median", "lo", "hi"])   # which node cluster to grab (per primary axis)
    parser.add_argument("--region_axis", type=str, default="x", choices=["x", "y", "z"])
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = get_combined_args(parser)
    if args.configs:
        from utils.config_loader import load_config
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, load_config(args.configs))

    dataset, pipe, hyper = model.extract(args), pipeline.extract(args), hp.extract(args)
    out = getattr(args, "out", None) or os.path.join(args.model_path, "edit_fig")
    os.makedirs(out, exist_ok=True)
    mags = [float(m) for m in args.mags.split(",") if m.strip()]
    dirs = [d.strip() for d in args.dirs.split(",") if d.strip()]

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyper)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        if not getattr(gaussians, "use_node_graph", False) or not gaussians.node_seeded:
            print("ERROR: model has no seeded control graph; editing requires use_node_graph.")
            sys.exit(1)
        bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        cam = scene.getTrainCameras()[args.cam]
        nd = gaussians._deformation.deformation_net.node_deform

        def save(img, name):
            torchvision.utils.save_image(img.clamp(0, 1), os.path.join(out, name))

        base = render(cam, gaussians, pipe, bg, stage="fine")["render"]
        save(base, "before.png")

        nx = nd.node_xyz                                       # (M,3)
        ext = (nx.max(0).values - nx.min(0).values).norm()
        # region centre: median cluster, or the low/high extreme along region_axis
        ra = AXIS[args.region_axis]
        if args.region == "median":
            center = nx.median(0).values
        else:
            q = 0.15 if args.region == "lo" else 0.85
            thr = torch.quantile(nx[:, ra], q)
            pick = nx[:, ra] < thr if args.region == "lo" else nx[:, ra] > thr
            center = nx[pick].median(0).values if pick.any() else nx.median(0).values
        sel = (nx - center).norm(dim=1) < args.radius_frac * ext
        n_sel = int(sel.sum())

        meta = {"W": cam.image_width, "H": cam.image_height, "n_nodes": int(nx.shape[0]),
                "n_selected": n_sel, "ext": float(ext), "radius_frac": args.radius_frac,
                "region": args.region, "region_axis": args.region_axis,
                "cam": args.cam, "iteration": args.iteration, "edits": []}

        # projected pixels of the selected nodes (for the handle overlay) + region centroid
        px, py, _ = project(nx[sel], cam)
        cx, cy, _ = project(center.unsqueeze(0), cam)
        meta["sel_px"] = px.tolist()
        meta["sel_py"] = py.tolist()
        meta["centroid_px"] = float(cx.item())
        meta["centroid_py"] = float(cy.item())

        for d in dirs:
            sign = -1.0 if d[0] == "-" else 1.0
            axis = AXIS[d[-1]]
            direction = torch.zeros(3, device=nx.device); direction[axis] = sign
            # arrow tip pixel at the largest magnitude (world centroid + drag)
            tip_world = center + direction * (max(mags) * ext)
            tx, ty, _ = project(tip_world.unsqueeze(0), cam)
            for m in mags:
                disp = torch.zeros_like(nx)
                disp[sel] = direction * (m * ext)
                nd.edit_translation = disp
                img = render(cam, gaussians, pipe, bg, stage="fine")["render"]
                tag = f"{d.replace('+','p').replace('-','m')}_{m:.2f}"
                save(img, f"after_{tag}.png")
                meta["edits"].append({"dir": d, "mag": m, "file": f"after_{tag}.png",
                                      "arrow_tip_px": float(tx.item()), "arrow_tip_py": float(ty.item())})
            # reset before next direction
            nd.edit_translation = torch.zeros_like(nx)

        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[edit_figure] {n_sel}/{nx.shape[0]} nodes | dirs={dirs} mags={mags} "
              f"-> {len(meta['edits'])} renders in {out}")


if __name__ == "__main__":
    main()
