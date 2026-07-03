#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = True
        self.render_process=False
        self.extra_mark = None
        self.camera_extent = None
        self.mode = 'binocular'
        self.no_fine = False
        self.init_pts = 200_000
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class ModelHiddenParams(ParamGroup):
    def __init__(self, parser):
        self.net_width = 64
        self.timebase_pe = 4
        self.defor_depth = 1
        self.posebase_pe = 10
        self.scale_rotation_pe = 2
        self.opacity_pe = 2
        self.timenet_width = 64
        self.timenet_output = 32
        self.bounds = 1.6
        self.plane_tv_weight = 0.0001
        self.time_smoothness_weight = 0.01
        self.l1_time_planes = 0.0001
        self.kplanes_config = {
                             'grid_dimensions': 2,
                             'input_coordinate_dim': 4,
                             'output_coordinate_dim': 32,
                             'resolution': [64, 64, 64, 25]
                            }
        self.multires = [1, 2, 4, 8]
        self.no_dx=False
        self.no_grid=False
        self.no_dx=False
        self.no_ds=False
        self.no_dr=False
        self.no_do=False
        # GC-EndoGaussian: control-node graph deformation (off by default = vanilla EndoGaussian)
        self.use_node_graph=False
        self.num_nodes=1024
        self.node_knn=8            # node<->node graph degree
        self.gauss_knn_K=4         # each Gaussian binds to its K nearest nodes
        self.gnn_layers=2          # message-passing layers (0 = SC-GS-style per-node MLP ablation)
        self.gnn_type='edgeconv'   # 'edgeconv' (mean aggregation) | 'gat' (attention aggregation)
        self.gnn_width=64
        self.node_pe=4             # positional-encoding frequencies for node xyz / edges / time
        self.control_route=False   # route the edit as a GNN input (propagate control) vs post-hoc bypass
        self.lambda_control=0.0    # weight of the control-consistency loss (only if control_route)
        self.control_train_K=16    # #handle nodes sampled per iter for the control-consistency loss
        self.bind_sigma_scale=1.0  # softmax temperature scale for distance-based binding weights
        self.lambda_arap=0.01
        self.lambda_isometric=0.0     # as-isometric-as-possible edge-length prior (tissue resists stretch)
        self.lambda_node_temporal=0.001
        self.node_hybrid=False        # graph low-freq motion + small per-Gaussian high-freq residual
        self.node_translation_only=False  # graph affects position only; rotation from full MLP (match mode)
        self.cut_aware=False          # break high-stretch node edges so deformation respects tissue cuts
        self.cut_beta=5.0             # cut gate sharpness
        self.cut_thresh=1.3           # edge stretch ratio beyond which it starts to "cut"
        self.node_reg_anneal=False    # linearly relax graph regularizers over the fine stage
        self.node_refresh_interval=1000
        self.lambda_flow=0.0          # optical-flow consistency loss weight (0 = off); offline cv2 Farneback
        # Occlusion-holdout stress test: blank a spatial box out of the loss mask for a contiguous
        # block of training frames (simulates a tool occluding tissue), then evaluate recovery on
        # that box at those frames. Fractions of image / of the train-frame index range.
        self.occ_holdout=False
        self.occ_x0=0.35
        self.occ_y0=0.30
        self.occ_x1=0.65
        self.occ_y1=0.70
        self.occ_block_lo=0.33
        self.occ_block_hi=0.66
        super().__init__(parser, "ModelHiddenParams")
        
class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.dataloader=False
        self.iterations = 30_000
        self.coarse_iterations = 3000
        self.train_frame_stride = 1   # >1 = sparse-view robustness: train on every Nth frame only
        self.grad_clip = 10.0         # deformation-field grad-norm clip (0 = off; for the stability study)
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 20_000
        self.deformation_lr_init = 0.00016
        self.deformation_lr_final = 0.000016
        self.deformation_lr_delay_mult = 0.01
        self.grid_lr_init = 0.0016
        self.grid_lr_final = 0.00016
        # GC-EndoGaussian: learning rate for the control-node GNN (between deformation & grid lr)
        self.node_lr_init = 0.0008
        self.node_lr_final = 0.00008

        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0
        self.lambda_lpips = 0
        self.weight_constraint_init= 1
        self.weight_constraint_after = 0.2
        self.weight_decay_iteration = 5000
        self.opacity_reset_interval = 3000
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold_coarse = 0.0002
        self.densify_grad_threshold_fine_init = 0.0002
        self.densify_grad_threshold_after = 0.0002
        self.pruning_from_iter = 500
        self.pruning_interval = 100
        self.opacity_threshold_coarse = 0.005
        self.opacity_threshold_fine_init = 0.005
        self.opacity_threshold_fine_after = 0.005
        
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
