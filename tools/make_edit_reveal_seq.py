"""Edit-during-playback video: the Figure-2 cutting lateral retraction, applied
while the 4D reconstruction plays over time.

Unlike the static crossfade (tools/make_edit_reveal_video.py), this actually
renders the cutting_match model across its video time-sweep and ramps the
inference-time control-node edit in mid-sequence: the scene plays unedited, then
a local group of nodes is dragged laterally (-x) to retract the overlying tissue
and reveal the region behind, held while time keeps advancing, then released.

Handle region matches Figure 2 exactly (median cluster, radius_frac 0.06, -x drag,
peak magnitude 0.16 of node-cloud extent). A green dashed circle marks the grabbed
region, projected once on the (static) video pose. GPU required -- run via sbatch.

  python3 tools/make_edit_reveal_seq.py \
      --model_path output/endonerf/cutting_match \
      --configs arguments/endonerf/cutting_graph_match.py --iteration 6000 \
      --out output/endonerf/cutting_match/edit_seq
"""
import os, sys, math
from argparse import ArgumentParser
import torch
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render

AXIS = {"x": 0, "y": 1, "z": 2}


def smooth(u):
    return 0.5 * (1.0 - math.cos(math.pi * max(0.0, min(1.0, u))))


def to8(img):
    return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()


def dashed_circle(im, cx, cy, r, color=(60, 220, 60), seg=20, w=2):
    for k in range(seg):
        if k % 2 == 0:
            a0, a1 = 360.0 * k / seg, 360.0 * (k + 0.72) / seg
            cv2.ellipse(im, (int(cx), int(cy)), (int(r), int(r)), 0, a0, a1, color, w, cv2.LINE_AA)
    cv2.circle(im, (int(cx), int(cy)), 3, color, -1, cv2.LINE_AA)
    return im


@torch.no_grad()
def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=6000, type=int)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--drag", type=str, default="-x")          # Figure-2 lateral retraction
    parser.add_argument("--radius_frac", type=float, default=0.06)
    parser.add_argument("--peak", type=float, default=0.16)        # fraction of node-cloud extent
    # edit timeline as fractions of the sequence: play / ramp-in / hold / release
    parser.add_argument("--t_play", type=float, default=0.30)
    parser.add_argument("--t_ramp", type=float, default=0.22)
    parser.add_argument("--t_release", type=float, default=0.10)
    args = get_combined_args(parser)
    if args.configs:
        from utils.config_loader import load_config
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, load_config(args.configs))

    dataset, pipe, hyper = model.extract(args), pipeline.extract(args), hp.extract(args)
    os.makedirs(args.out, exist_ok=True)
    frames_dir = os.path.join(args.out, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for f in os.listdir(frames_dir):
        if f.endswith(".png"):
            os.remove(os.path.join(frames_dir, f))

    gaussians = GaussianModel(dataset.sh_degree, hyper)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    if not getattr(gaussians, "use_node_graph", False) or not gaussians.node_seeded:
        print("ERROR: model has no seeded control graph; editing requires use_node_graph.")
        sys.exit(1)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    nd = gaussians._deformation.deformation_net.node_deform
    vcams = scene.getVideoCameras()
    N = len(vcams)

    nx = nd.node_xyz
    M = nx.shape[0]
    ext = float((nx.max(0).values - nx.min(0).values).norm())
    center = nx.median(0).values                                   # region="median"
    sel = (nx - center).norm(dim=1) < args.radius_frac * ext
    n_sel = int(sel.sum())
    sign = -1.0 if args.drag[0] == "-" else 1.0
    axis = AXIS[args.drag[-1]]
    direction = torch.zeros(3, device=nx.device); direction[axis] = sign
    PEAK = args.peak * ext
    zero = torch.zeros(M, 3, device=nx.device)

    # green dashed handle circle, projected once on the (static) video pose
    cam0 = vcams[0]
    W, Hh = cam0.image_width, cam0.image_height
    ph = torch.cat([nx[sel], torch.ones_like(nx[sel][:, :1])], dim=1)
    clip = ph @ cam0.full_proj_transform.to(nx.device)
    ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-7)
    spx = (ndc[:, 0] * .5 + .5) * W; spy = (ndc[:, 1] * .5 + .5) * Hh
    cph = torch.cat([center.unsqueeze(0), torch.ones(1, 1, device=nx.device)], dim=1)
    cclip = cph @ cam0.full_proj_transform.to(nx.device)
    cndc = cclip[:, :3] / cclip[:, 3:4].clamp_min(1e-7)
    cpx = float((cndc[0, 0] * .5 + .5) * W); cpy = float((cndc[0, 1] * .5 + .5) * Hh)
    rp = float(torch.quantile(torch.hypot(spx - cpx, spy - cpy), 0.80))

    t_play, t_ramp, t_release = args.t_play, args.t_ramp, args.t_release

    def mag_at(u):                                                 # u in [0,1] over the sequence
        if u < t_play:
            return 0.0
        if u < t_play + t_ramp:
            return smooth((u - t_play) / t_ramp) * PEAK
        if u < 1.0 - t_release:
            return PEAK
        return (1.0 - smooth((u - (1.0 - t_release)) / t_release)) * PEAK

    for i, cam in enumerate(vcams):
        u = i / max(N - 1, 1)
        disp = zero.clone(); disp[sel] = direction * mag_at(u)
        nd.edit_translation = disp
        img = to8(render(cam, gaussians, pipe, bg, stage="fine")["render"])
        dashed_circle(img, cpx, cpy, rp)
        cv2.imwrite(os.path.join(frames_dir, f"f{i:04d}.png"), img[:, :, ::-1])   # RGB->BGR
    nd.edit_translation = zero

    print(f"[edit_seq] {N} frames | {n_sel}/{M} nodes | drag={args.drag} peak={args.peak} "
          f"| circle=({cpx:.0f},{cpy:.0f}) r={rp:.0f} -> {frames_dir}")


if __name__ == "__main__":
    main()
