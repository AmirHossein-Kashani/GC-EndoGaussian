"""Do the Gaussians that FORM THE ORGAN at the edit frame get re-used to form the BED later?
If yes, a static layer decomposition of this model is impossible: removing the organ also removes
the very Gaussians that would show what is beneath."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import ArgumentParser
import torch, numpy as np, cv2
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render
from tools.auto_reveal import eval_deform, project, visible_at

p = ArgumentParser(); m = ModelParams(p, sentinel=True); pl = PipelineParams(p); hp = ModelHiddenParams(p)
p.add_argument("--iteration", type=int, default=6000); p.add_argument("--configs", type=str, default="")
p.add_argument("--frame", type=int, required=True); p.add_argument("--ref", type=int, required=True)
p.add_argument("--obj_mask", type=str, required=True)
a = get_combined_args(p)
for k in ("rect","px","py","mask"):
    if not hasattr(a,k): setattr(a,k,None)
if a.configs:
    from utils.config_loader import load_config
    from utils.params_utils import merge_hparams
    a = merge_hparams(a, load_config(a.configs))
ds, pipe, hy = m.extract(a), pl.extract(a), hp.extract(a)
g = GaussianModel(ds.sh_degree, hy); sc = Scene(ds, g, load_iteration=a.iteration, shuffle=False)
bg = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
vc = sc.getVideoCameras(); cam = vc[a.frame]; ref = vc[a.ref]
H, W = cam.image_height, cam.image_width

msk = cv2.imread(a.obj_mask, 0)
msk = (cv2.resize(msk,(W,H),interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
mt = torch.from_numpy(msk).cuda()

_, full_e = eval_deform(g, cam); sx, sy, vz = project(full_e, cam)
in_m = mt[sy.long().clamp(0,H-1), sx.long().clamp(0,W-1)] > 0
z = vz[in_m].cpu().numpy(); z = z[(z>1)&(z<np.percentile(z,99.5))]
from tools.auto_reveal import otsu_split
thr = otsu_split(z) - 8.0
obj = in_m & (vz < thr)

vis_ref = visible_at(g, ref, pipe, bg, 0.05)
_, full_r = eval_deform(g, ref); rx, ry, _ = project(full_r, ref)
in_m_ref = mt[ry.long().clamp(0,H-1), rx.long().clamp(0,W-1)] > 0
bed = vis_ref & in_m_ref                    # forms the visible surface inside the footprint at ref

overlap = (obj & bed).sum().item()
print(f"object @frame{a.frame}: {int(obj.sum())} | visible bed @frame{a.ref} inside footprint: {int(bed.sum())}")
print(f"OVERLAP (same Gaussians doing both jobs): {overlap} "
      f"= {100*overlap/max(int(bed.sum()),1):.1f}% of the bed set")
