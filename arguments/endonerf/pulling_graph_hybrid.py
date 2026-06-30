# GC-EndoGaussian HYBRID: control-node GNN supplies coherent low-frequency motion, a small
# per-Gaussian MLP residual recovers high-frequency detail. Uses the as-isometric prior
# (tissue resists stretch, bends freely) instead of rigid ARAP, annealed over the fine stage.
# Goal of this variant: stop losing to vanilla on the standard benchmark (recover the LPIPS
# regression of the pure-replace graph) while keeping the graph's coherence for the
# occlusion/geometry tests. Same num_nodes / gnn_layers / iters as pulling_graph.py for a
# clean A/B against the pure-replace result.
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
    # ---- control-node graph (hybrid) ----
    use_node_graph = True,
    node_hybrid = True,        # <-- graph + per-Gaussian residual
    num_nodes = 1024,
    node_knn = 8,
    gauss_knn_K = 4,
    gnn_layers = 2,
    gnn_width = 64,
    node_pe = 4,
    lambda_arap = 0.0,         # <-- switched off in favour of the isometric prior
    lambda_isometric = 0.01,   # <-- as-isometric-as-possible edge-length preservation
    lambda_node_temporal = 0.001,
    node_reg_anneal = True,    # <-- relax coherence priors toward the end of training
    node_refresh_interval = 1000,
)
