# GC-EndoGaussian ABLATION: control nodes + LBS but NO message passing (gnn_layers = 0).
# This degrades the GNN to a per-node-independent MLP (SC-GS-style control). It is the
# make-or-break baseline: if it matches pulling_graph.py on the occlusion/geometry tests,
# the GNN adds nothing. Identical to pulling_graph.py except gnn_layers = 0.
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
    l1_time_planes =  0,
    weight_decay_iteration=0,
    # ---- control-node graph (no message passing) ----
    use_node_graph = True,
    num_nodes = 1024,
    node_knn = 8,
    gauss_knn_K = 4,
    gnn_layers = 0,        # <-- ablation: per-node independent MLP, no graph coupling
    gnn_width = 64,
    node_pe = 4,
    lambda_arap = 0.01,
    lambda_node_temporal = 0.001,
    node_refresh_interval = 1000,
)
