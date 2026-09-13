# C3VD desc_t4_a (colonoscopy phantom, 148 frames) -- GC-EndoGaussian MATCH recipe.
#
# The first scene here with REAL per-frame camera poses: the colonoscope travels 24.8 mm down the
# lumen at ~10.8 mm working distance (baseline/depth ~ 2.3), so surfaces are observed from many
# angles and genuinely occluded volume can be reconstructed. Every earlier scene wrote a single
# repeated pose into poses_bounds.npy, which can only ever yield a surface with thickness.
#
# binocular mode: depth is ground truth (from 2D-3D registration), so the inverse-depth L1 term
# applies -- unlike Endoscapes, where depth was a monocular estimate and needed the Pearson branch.
# camera_extent=11 matches the median scene depth in mm.
ModelParams = dict(extra_mark='endonerf', camera_extent=11)

OptimizationParams = dict(
    coarse_iterations = 1000,
    deformation_lr_init = 0.00016, deformation_lr_final = 0.0000016, deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.0016, grid_lr_final = 0.000016,
    node_lr_init = 1e-5, node_lr_final = 1.0000000000000002e-06,
    iterations = 3000, percent_dense = 0.01,
    opacity_reset_interval = 4000, position_lr_max_steps = 4000, prune_interval = 3000
)

ModelHiddenParams = dict(
    kplanes_config = {'grid_dimensions': 2, 'input_coordinate_dim': 4,
                      'output_coordinate_dim': 64, 'resolution': [64, 64, 64, 100]},
    multires = [1, 2, 4, 8], defor_depth = 0, net_width = 32,
    plane_tv_weight = 0, time_smoothness_weight = 0, l1_time_planes = 0, weight_decay_iteration = 0,
    use_node_graph = True, node_hybrid = True, node_translation_only = True,
    num_nodes = 2048, node_knn = 8, gauss_knn_K = 4, gnn_layers = 2, gnn_width = 64, node_pe = 4,
    lambda_arap = 0.0, lambda_isometric = 0.0, lambda_node_temporal = 0.0,
    node_reg_anneal = False, node_refresh_interval = 999999,
)
