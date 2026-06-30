import torch
import torch.nn as nn
import torch.nn.init as init
from scene.hexplane import HexPlaneField
from scene.node_deformation import NodeGraphDeformation

class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.no_grid = args.no_grid

        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        self.pos_deform, self.scales_deform, self.rotations_deform, self.opacity_deform = self.create_net()
        self.args = args
        # GC-EndoGaussian: optional graph control layer owning position+rotation deformation.
        self.use_node_graph = getattr(args, "use_node_graph", False)
        self.node_deform = NodeGraphDeformation(args) if self.use_node_graph else None
        
    def create_net(self):
        mlp_out_dim = 0
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + self.grid.feat_dim ,self.W)]
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        
        return  \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
    
    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_emb):
        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:
            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            h = grid_feature
        h = self.feature_out(h)
        return h

    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None, time_emb=None, binding_idx=None, binding_w=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, time_emb, binding_idx, binding_w)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx

    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, time_emb, binding_idx=None, binding_w=None):
        # GC-EndoGaussian: when the graph control layer is active and bindings are supplied,
        # the GNN owns position (dx) and rotation (dr); the HexPlane/MLP only handles the
        # cheaper scale/opacity residuals. Falls back to the original per-Gaussian MLP otherwise.
        use_ng = self.node_deform is not None and binding_idx is not None
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_emb).float()

        if use_ng:
            pts, rot_graph = self.node_deform(rays_pts_emb[:, :3], rotations_emb[:, :4],
                                              time_emb[:, :1], binding_idx, binding_w)
            translation_only = getattr(self.args, "node_translation_only", False)
            # Rotation: from the graph (LBS-blended quaternion), OR — in translation-only mode —
            # from the full per-Gaussian MLP exactly like vanilla (avoids the lossy quaternion blend;
            # editing is done by dragging = translation, so the graph rotation isn't needed).
            if translation_only:
                rotations = rotations_emb[:, :4] if self.args.no_dr else rotations_emb[:, :4] + self.rotations_deform(hidden)
            else:
                rotations = rot_graph
            # Hybrid: small per-Gaussian residual recovers high-frequency detail on top of the graph.
            if getattr(self.args, "node_hybrid", False):
                if not self.args.no_dx:
                    pts = pts + self.pos_deform(hidden)
                if not translation_only and not self.args.no_dr:
                    rotations = rotations + self.rotations_deform(hidden)
        else:
            if self.args.no_dx:
                pts = rays_pts_emb[:, :3]
            else:
                dx = self.pos_deform(hidden)
                pts = rays_pts_emb[:, :3] + dx

            if self.args.no_dr:
                rotations = rotations_emb[:,:4]
            else:
                dr = self.rotations_deform(hidden)
                rotations = rotations_emb[:,:4] + dr

        if self.args.no_ds:
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)
            scales = scales_emb[:,:3] + ds

        if self.args.no_do:
            opacity = opacity_emb[:,:1]
        else:
            do = self.opacity_deform(hidden)
            opacity = opacity_emb[:,:1] + do

        return pts, scales, rotations, opacity

    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name and "node_deform" not in name:
                parameter_list.append(param)
        return parameter_list

    def get_node_parameters(self):
        if self.node_deform is None:
            return []
        return list(self.node_deform.parameters())

    def get_grid_parameters(self):
        return list(self.grid.parameters())

class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
            nn.Linear(times_ch, timenet_width), nn.ReLU(),
            nn.Linear(timenet_width, timenet_output))
        
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(4+3)+((4+3)*scale_rotation_pe)*2, input_ch_time=timenet_output, args=args)
        
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        # restore the identity init of the SE(3) head clobbered by the global xavier apply
        if self.deformation_net.node_deform is not None:
            self.deformation_net.node_deform.init_identity_head()
    
    def forward(self, point, scales=None, rotations=None, opacity=None, times_sel=None, binding_idx=None, binding_w=None):
        if times_sel is not None:
            return self.forward_dynamic(point, scales, rotations, opacity, times_sel, binding_idx, binding_w)
        else:
            return self.forward_static(point)

    def forward_static(self, points):
        points = self.deformation_net(points)
        return points

    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, times_sel=None, binding_idx=None, binding_w=None):
        # times_emb = poc_fre(times_sel, self.time_poc)
        means3D, scales, rotations, opacity = self.deformation_net( point,
                                                scales,
                                                rotations,
                                                opacity,
                                                times_sel,
                                                binding_idx,
                                                binding_w)
        return means3D, scales, rotations, opacity

    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())

    def get_node_parameters(self):
        return self.deformation_net.get_node_parameters()

    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
