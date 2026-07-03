# SC-GS-style baseline + the per-Gaussian residual (node_hybrid=True) — the "residual-matched" ablation.
# Purpose: rule out that our reconstruction/tracking win over the SC-GS-style baseline is merely because
# our method has a per-Gaussian residual and the SC-GS-style one does not. This config gives the SC-GS-style
# model (gnn_layers=0, full SE(3), ARAP, adaptive nodes) the SAME residual our match recipe uses, isolating
# the residual's contribution. If this still trails our match, the win is the full recipe, not the residual.
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
    # ---- SC-GS-style control-node graph + residual (the only delta vs pulling_graph_scgs.py) ----
    use_node_graph = True,
    num_nodes = 2048,
    node_knn = 8,
    gauss_knn_K = 4,
    gnn_layers = 0,             # independent control points (SC-GS core)
    gnn_width = 64,
    node_pe = 4,
    node_hybrid = True,         # <-- the residual, matched to our match recipe
    # node_translation_only left False -> full SE(3) (SC-GS-style)
    lambda_arap = 0.01,
    lambda_node_temporal = 0.001,
    node_reg_anneal = False,
    node_refresh_interval = 1000,
)
