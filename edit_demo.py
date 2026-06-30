"""Controllable-deformation demo for GC-EndoGaussian.

Loads a trained graph model, grabs a local region of control nodes, drags them by a chosen
displacement (set node_deform.edit_translation), and re-renders the same view. The bound
Gaussians follow via LBS — tissue motion editing the per-Gaussian baseline cannot do.

Writes before / after / diff PNGs (and a small sweep) to <out>.

Usage:
  python edit_demo.py --model_path output/endonerf/<exp> --configs <config> [--edit_mag 0.3 --axis y]
"""
import os, sys
import torch
import torchvision
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=3000, type=int)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--edit_mag", type=float, default=0.3)   # fraction of node-cloud extent
    parser.add_argument("--radius_frac", type=float, default=0.18)
    parser.add_argument("--axis", type=str, default="y", choices=["x", "y", "z"])
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = get_combined_args(parser)
    if args.configs:
        from utils.config_loader import load_config
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, load_config(args.configs))

    dataset, pipe, hyper = model.extract(args), pipeline.extract(args), hp.extract(args)
    # get_combined_args drops cmdline args left at their None default, so read defensively
    out = getattr(args, "out", None) or os.path.join(args.model_path, "edit_demo")
    os.makedirs(out, exist_ok=True)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyper)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        if not getattr(gaussians, "use_node_graph", False) or not gaussians.node_seeded:
            print("ERROR: this model has no seeded control graph; editing requires use_node_graph.")
            sys.exit(1)
        bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        cam = scene.getTrainCameras()[args.cam]
        nd = gaussians._deformation.deformation_net.node_deform

        def save(img, name):
            torchvision.utils.save_image(img.clamp(0, 1), os.path.join(out, name))

        base = render(cam, gaussians, pipe, bg, stage="fine")["render"]
        save(base, "before.png")

        nx = nd.node_xyz                                   # (M,3)
        center = nx.median(0).values
        ext = (nx.max(0).values - nx.min(0).values).norm()
        sel = (nx - center).norm(dim=1) < args.radius_frac * ext
        axis = {"x": 0, "y": 1, "z": 2}[args.axis]
        direction = torch.zeros(3, device=nx.device); direction[axis] = 1.0

        # a small sweep so the edit is visibly a controllable handle, not a one-off
        for k, frac in enumerate([0.33, 0.66, 1.0]):
            disp = torch.zeros_like(nx)
            disp[sel] = direction * (args.edit_mag * ext * frac)
            nd.edit_translation = disp
            img = render(cam, gaussians, pipe, bg, stage="fine")["render"]
            save(img, f"after_{k}.png")
            save((img - base).abs(), f"diff_{k}.png")

        print(f"[edit_demo] dragged {int(sel.sum())}/{nx.shape[0]} nodes along +{args.axis} "
              f"(mag {args.edit_mag} x extent); wrote before/after/diff to {out}")


if __name__ == "__main__":
    main()
