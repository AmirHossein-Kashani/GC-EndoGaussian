"""Figure-2 drag edit with NEVER-OBSERVED Gaussians rendered black.

The paper's Figure 2 shows the region revealed by the drag as a soft blur: the model's rendering of
content it barely (or never) observed. This variant makes the observation boundary explicit: every
Gaussian that fails the surface-visibility test at ALL sampled timestamps is painted black, so the
black area in the edited view marks exactly where observation ends, and every colored pixel was
genuinely seen at some timestamp. Binary rule by user decision -- partially observed content keeps
its full color (it is real, not hallucinated).

Mechanics reuse existing pieces unchanged:
  - drag edit: identical to tools/edit_figure.py (node ball selection, nd.edit_translation buffer);
  - visibility: the seen_at loop of tools/find_verifiable_reveal.py (relative depth agreement
    within --tol at every --sample-th video camera), computed in the UNEDITED configuration --
    observedness is a property of training-time coverage, and blackened Gaussians then move with
    the edit;
  - painting black: _features_dc = -0.5/C0 and _features_rest = 0 in place (sh2rgb + 0.5 == 0),
    restored afterwards; same clone/restore pattern find_verifiable_reveal.py uses for _opacity.

  python3 tools/edit_figure_obsblack.py --model_path output/endonerf/cutting_match \
      --configs arguments/endonerf/cutting_graph_match.py --iteration 6000 \
      --cam 0 --dir -x --mag 0.16 --radius_frac 0.06 --region median \
      --sample 4 --tol 0.03 --out output/endonerf/cutting_match/edit_fig_obsblack
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torchvision
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render
from tools.endoscapes_peel import deformed_xyz

AXIS = {"x": 0, "y": 1, "z": 2}
C0 = 0.28209479177387814                     # SH DC basis constant (utils/sh_utils.py)


@torch.no_grad()
def visible_at(gaussians, cam, pipe, bg, tol):
    """True for Gaussians lying on the rendered visible surface at this camera/time.
    Inlined from tools/find_verifiable_reveal.py (which renders a module-global instead of pc)."""
    depth = render(cam, gaussians, pipe, bg, stage="fine")["depth"]
    depth = depth[0] if depth.dim() == 3 else depth
    H, W = depth.shape[-2], depth.shape[-1]
    xyz = deformed_xyz(gaussians, cam)
    ph = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
    clip = ph @ cam.full_proj_transform.to(xyz.device)
    ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-7)
    sx = (ndc[:, 0] * .5 + .5) * W
    sy = (ndc[:, 1] * .5 + .5) * H
    vz = (ph @ cam.world_view_transform.to(xyz.device))[:, 2]
    xi = sx.long().clamp(0, W - 1)
    yi = sy.long().clamp(0, H - 1)
    surf = depth[yi, xi]
    inside = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H) & (surf > 0)
    return inside & ((vz - surf).abs() <= tol * surf.clamp_min(1e-6))


@torch.no_grad()
def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=6000, type=int)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--dir", type=str, default="-x")
    parser.add_argument("--mag", type=float, default=0.16)
    parser.add_argument("--radius_frac", type=float, default=0.06)
    parser.add_argument("--region", type=str, default="median", choices=["median", "lo", "hi"])
    parser.add_argument("--region_axis", type=str, default="x", choices=["x", "y", "z"])
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--tols", type=str, default="0.03",
                        help="comma-separated visibility tolerances; one output per value")
    parser.add_argument("--restrict_reveal", action="store_true",
                        help="blacken only Gaussians projecting inside the drag-revealed 2D area, "
                             "so the rest of the frame stays identical to the plain edited render")
    parser.add_argument("--out", type=str, required=True)
    args = get_combined_args(parser)
    if args.configs:
        from utils.config_loader import load_config
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, load_config(args.configs))

    dataset, pipe, hyper = model.extract(args), pipeline.extract(args), hp.extract(args)
    os.makedirs(args.out, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree, hyper)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    if not getattr(gaussians, "use_node_graph", False) or not gaussians.node_seeded:
        sys.exit("ERROR: model has no seeded control graph")
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    cam = scene.getTrainCameras()[args.cam]
    vcams = scene.getVideoCameras()
    nd = gaussians._deformation.deformation_net.node_deform

    def save(img, name):
        torchvision.utils.save_image(img.clamp(0, 1), os.path.join(args.out, name))

    tols = [float(t) for t in args.tols.split(",") if t.strip()]

    # ---- observedness over the whole (unedited) sequence, one flag set per tolerance -------
    # A strict tolerance wrongly blackens the BACKING of observed surfaces (Gaussians a little
    # beneath the visible sheet never match the surface depth exactly), which inflates the black
    # area far beyond genuinely-unseen content. Looser shells count them as observed.
    seen_cnt = {t: visible_at(gaussians, cam, pipe, bg, t).int() for t in tols}
    n_checked = 1
    for s in range(0, len(vcams), args.sample):
        for t in tols:
            seen_cnt[t] += visible_at(gaussians, vcams[s], pipe, bg, t).int()
        n_checked += 1
    n = gaussians._xyz.shape[0]
    # confidence-graded rule: "unreliable" = observed in fewer than min_frac of checked frames.
    # min_frac=0 reproduces the strict never-observed rule.
    fracs = [0.0, 0.05, 0.10, 0.20]
    observed = {}
    for t in tols:
        for f in fracs:
            need = max(1, int(round(f * n_checked)))
            rel = seen_cnt[t] >= need
            observed[(t, f)] = rel
            print(f"[obsblack] tol={t} min_frac={f}: {n - int(rel.sum())}/{n} Gaussians below "
                  f"{need}/{n_checked} observations ({100 * (1 - rel.float().mean()):.1f}%)")

    # ---- the identical Figure-2 drag edit --------------------------------------------------
    nx = nd.node_xyz
    ext = (nx.max(0).values - nx.min(0).values).norm()
    ra = AXIS[args.region_axis]
    if args.region == "median":
        center = nx.median(0).values
    else:
        q = 0.15 if args.region == "lo" else 0.85
        thr = torch.quantile(nx[:, ra], q)
        pick = nx[:, ra] < thr if args.region == "lo" else nx[:, ra] > thr
        center = nx[pick].median(0).values if pick.any() else nx.median(0).values
    sel = (nx - center).norm(dim=1) < args.radius_frac * ext
    sign = -1.0 if args.dir[0] == "-" else 1.0
    direction = torch.zeros(3, device=nx.device); direction[AXIS[args.dir[-1]]] = sign
    disp = torch.zeros_like(nx)
    disp[sel] = direction * (args.mag * ext)
    nd.edit_translation = disp

    tag = f"{args.dir.replace('+', 'p').replace('-', 'm')}_{args.mag:.2f}"
    nd.edit_translation = torch.zeros_like(nx)
    before = render(cam, gaussians, pipe, bg, stage="fine")["render"]
    nd.edit_translation = disp
    after = render(cam, gaussians, pipe, bg, stage="fine")["render"]
    save(after, f"after_{tag}.png")

    # ---- the drag-revealed 2D area + which Gaussians land inside it ------------------------
    reveal2d = ((after - before).abs().mean(0) > 0.03).float()
    reveal2d = torch.nn.functional.max_pool2d(reveal2d[None, None], 15, 1, 7)[0, 0] > 0
    in_reveal = torch.ones(n, dtype=torch.bool, device="cuda")
    if args.restrict_reveal:
        xyz_e = deformed_xyz(gaussians, cam)                      # includes the active edit
        ph = torch.cat([xyz_e, torch.ones_like(xyz_e[:, :1])], dim=1)
        clip = ph @ cam.full_proj_transform.to(xyz_e.device)
        ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-7)
        H, W = reveal2d.shape
        xi = ((ndc[:, 0] * .5 + .5) * W).long().clamp(0, W - 1)
        yi = ((ndc[:, 1] * .5 + .5) * H).long().clamp(0, H - 1)
        in_reveal = reveal2d[yi, xi]

    # ---- same edited view, never-observed Gaussians painted black, one image per tol -------
    dc_orig = gaussians._features_dc.detach().clone()
    rest_orig = gaussians._features_rest.detach().clone()
    for (t, f) in observed:
        blk = (~observed[(t, f)]) & in_reveal
        gaussians._features_dc.data[blk] = -0.5 / C0
        gaussians._features_rest.data[blk] = 0.0
        after_black = render(cam, gaussians, pipe, bg, stage="fine")["render"]
        name = f"after_{tag}_obsblack_tol{t:g}_mf{f:g}{'_rr' if args.restrict_reveal else ''}.png"
        save(after_black, name)
        gaussians._features_dc.data.copy_(dc_orig)
        gaussians._features_rest.data.copy_(rest_orig)
        dark = (after - after_black).abs().mean(0) > 0.03
        inside = float((dark & reveal2d).float().sum() / reveal2d.float().sum().clamp_min(1) * 100)
        print(f"[obsblack] tol={t} mf={f}: blackened {int(blk.sum())} Gaussians | {inside:.1f}% of "
              f"the revealed area is black -> {name}")

        # ---- explicit per-pixel unseen mask: alpha-blended contribution of never-observed
        # Gaussians (override_color: unseen=1, observed=0), thresholded and painted SOLID black
        # on the edited view -- unmistakable at figure scale, no soft blending.
        ucol = torch.zeros(n, 3, device="cuda")
        ucol[blk] = 1.0
        umap = render(cam, gaussians, pipe, bg, override_color=ucol, stage="fine")["render"][0]
        save(umap.unsqueeze(0).repeat(3, 1, 1), f"umap_{tag}_tol{t:g}_mf{f:g}.png")
        for thr in (0.3, 0.5):
            hard = umap > thr
            panel = after.clone()
            panel[:, hard] = 0.0
            mname = f"after_{tag}_unseenmask_tol{t:g}_mf{f:g}_thr{thr:g}.png"
            save(panel, mname)
            frac = float((hard & reveal2d).float().sum() / reveal2d.float().sum().clamp_min(1) * 100)
            print(f"[obsblack] tol={t} mf={f} thr={thr}: hard mask covers {frac:.1f}% of the "
                  f"revealed area -> {mname}")
    nd.edit_translation = torch.zeros_like(nx)


if __name__ == "__main__":
    main()
