# Variant B: SC-GS-style deformation (nodes ARE the motion, full SE(3), no dense residual) +
# tool masks + ARAP/temporal coherence. Strictest: no residual channel through which occluded
# Gaussians can be repurposed; they must move rigidly with their (static) neighbourhood.
# STIR 06/seq23, 6000-iteration variant -- same MATCH recipe as arguments/endonerf/
# cutting_graph_match.py (the config behind the paper's successful peel figure). The 3k run
# (arguments/stir/seq23.py) proved the reveal is real (64.8% of hidden Gaussians observed at
# another time; ~20% pixel change on removal) but rendered frame ~10 soft; the extra budget is
# for figure sharpness, everything else unchanged.
ModelParams = dict(extra_mark='endonerf', camera_extent=10)

OptimizationParams = dict(
    coarse_iterations = 1000,
    deformation_lr_init = 0.00016, deformation_lr_final = 0.0000016, deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.0016, grid_lr_final = 0.000016,
    node_lr_init = 0.0008, node_lr_final = 0.00008,
    iterations = 6000, percent_dense = 0.01,
    opacity_reset_interval = 7000, position_lr_max_steps = 7000, prune_interval = 3000
)

ModelHiddenParams = dict(
    kplanes_config = {'grid_dimensions': 2, 'input_coordinate_dim': 4,
                      'output_coordinate_dim': 64, 'resolution': [64, 64, 64, 100]},
    multires = [1, 2, 4, 8], defor_depth = 0, net_width = 32,
    plane_tv_weight = 0, time_smoothness_weight = 0, l1_time_planes = 0, weight_decay_iteration = 0,
    use_node_graph = True,
    node_hybrid = False,
    node_translation_only = False,
    num_nodes = 2048, node_knn = 8, gauss_knn_K = 4, gnn_layers = 2, gnn_width = 64, node_pe = 4,
    lambda_arap = 0.01, lambda_isometric = 0.0, lambda_node_temporal = 0.001,
    node_reg_anneal = False,
    node_refresh_interval = 999999,
)
