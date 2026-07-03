# GAT variant of our MATCH recipe: identical to pulling_graph_match_3k.py except the message-passing
# aggregation is GAT-style attention (gnn_type='gat') instead of EdgeConv mean aggregation.
# Ablation: does graph attention change reconstruction/tracking/controllability vs mean aggregation?
ModelParams = dict(extra_mark='endonerf', camera_extent=10)

OptimizationParams = dict(
    coarse_iterations = 1000,
    deformation_lr_init = 0.00016, deformation_lr_final = 0.0000016, deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.0016, grid_lr_final = 0.000016,
    node_lr_init = 0.0008, node_lr_final = 0.00008,
    iterations = 3000, percent_dense = 0.01,
    opacity_reset_interval = 4000, position_lr_max_steps = 4000, prune_interval = 3000
)

ModelHiddenParams = dict(
    kplanes_config = {'grid_dimensions': 2, 'input_coordinate_dim': 4,
                      'output_coordinate_dim': 64, 'resolution': [64, 64, 64, 100]},
    multires = [1, 2, 4, 8], defor_depth = 0, net_width = 32,
    plane_tv_weight = 0, time_smoothness_weight = 0, l1_time_planes = 0, weight_decay_iteration = 0,
    use_node_graph = True,
    node_hybrid = True,
    node_translation_only = True,
    num_nodes = 2048, node_knn = 8, gauss_knn_K = 4, gnn_layers = 2, gnn_width = 64, node_pe = 4,
    gnn_type = 'gat',          # <-- the only change vs the match recipe
    lambda_arap = 0.0, lambda_isometric = 0.0, lambda_node_temporal = 0.0,
    node_reg_anneal = False,
    node_refresh_interval = 999999,
)
