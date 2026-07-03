# GC-EndoGaussian SC-GS-STYLE LEARNED BASELINE (reimplemented in our pipeline, not the official code).
# A faithful proxy for SC-GS (Huang et al., CVPR 2024) using our node-graph infrastructure:
#   - gnn_layers=0        -> independent per-node MLP control points (no message-passing coupling): SC-GS core
#   - node_translation_only OFF -> full SE(3) per control point (rotation via quaternion-LBS blend)
#   - node_hybrid OFF     -> the control points ARE the deformation (no per-Gaussian residual field)
#   - lambda_arap / lambda_node_temporal > 0 -> SC-GS's ARAP + temporal coherence priors (fired in fine stage)
#   - node_refresh_interval=1000 -> adaptive control points (periodic re-seed), like SC-GS's adjustment
# num_nodes=2048 matches our *match* budget so the comparison isolates the GNN + integration recipe,
# not the control-point count. Trained/evaluated identically to the match runs (control-from-tracks).
ModelParams = dict(
    extra_mark = 'endonerf',
    camera_extent = 10
)

OptimizationParams = dict(
    coarse_iterations = 1000,
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.0016,
    grid_lr_final = 0.000016,
    node_lr_init = 0.0008,
    node_lr_final = 0.00008,
    iterations = 3000,
    percent_dense = 0.01,
    opacity_reset_interval = 4000,
    position_lr_max_steps = 4000,
    prune_interval = 3000
)

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 64,
     'resolution': [64, 64, 64, 100]
    },
    multires = [1, 2, 4, 8],
    defor_depth = 0,
    net_width = 32,
    plane_tv_weight = 0,
    time_smoothness_weight = 0,
    l1_time_planes = 0,
    weight_decay_iteration = 0,
    # ---- SC-GS-style control-node graph ----
    use_node_graph = True,
    num_nodes = 2048,
    node_knn = 8,
    gauss_knn_K = 4,
    gnn_layers = 0,             # independent control points, no message passing
    gnn_width = 64,
    node_pe = 4,
    # node_hybrid / node_translation_only left at their False defaults -> full-SE(3), points-are-deformation
    lambda_arap = 0.01,        # ARAP rigidity prior (SC-GS signature)
    lambda_node_temporal = 0.001,
    node_reg_anneal = False,   # constant coherence weight (SC-GS does not anneal)
    node_refresh_interval = 1000,  # adaptive control points
)
