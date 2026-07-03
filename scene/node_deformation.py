"""Graph-Controlled deformation for EndoGaussian (GC-EndoGaussian).

A sparse set of control "hypernodes" is seeded in high-motion tissue regions; a small
Graph Neural Network passes messages over the node KNN graph and emits a *per-node SE(3)*
(rotation + translation) at each timestamp. Each Gaussian is softly bound to its K nearest
nodes and recovers *both* its movement (dx) and rotation (dr) by a linear-blend-skinning
(LBS) blend of its bound nodes' transforms.

Design notes
------------
* The GNN runs on the SPARSE node graph (M ~ 1024), never on the dense Gaussian set, so the
  only per-Gaussian work is a gather + weighted blend (LBS). Inference cost is equal-or-lower
  than the per-Gaussian HexPlane+MLP it replaces.
* Node features are a pure function of (encoded node position, time) — there are no per-node
  free parameters — so re-seeding the nodes (when the Gaussian cloud densifies) is trivial:
  just move the `node_xyz` buffer, rebuild the graph, recompute bindings. The learnable
  parameters (message + update MLPs and the SE(3) head) are fixed-shape regardless of M.
* `gnn_layers == 0` degrades the module to a per-node-independent MLP + LBS — i.e. the
  SC-GS-style baseline. This is the make-or-break ablation switch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
#  SE(3) / quaternion helpers (pure torch, no external deps; quat convention is (w,x,y,z)
#  to match 3DGS rotation activation and utils.general_utils.build_rotation).
# --------------------------------------------------------------------------------------
def rotation_6d_to_matrix(d6):
    """Zhou et al. continuous 6D rotation -> (...,3,3). Columns are the orthonormal basis."""
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_quaternion(R):
    """(...,3,3) rotation matrices -> (...,4) quaternions in (w,x,y,z)."""
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    trace = m00 + m11 + m22
    w = torch.sqrt(torch.clamp(1.0 + trace, min=1e-8)) * 0.5
    # guard against div-by-zero; w is >= 0.5*sqrt(1e-8) so safe but keep eps anyway
    x = (R[..., 2, 1] - R[..., 1, 2]) / (4.0 * w)
    y = (R[..., 0, 2] - R[..., 2, 0]) / (4.0 * w)
    z = (R[..., 1, 0] - R[..., 0, 1]) / (4.0 * w)
    q = torch.stack([w, x, y, z], dim=-1)
    return F.normalize(q, dim=-1)


def quaternion_multiply(a, b):
    """Hamilton product of two (w,x,y,z) quaternions, broadcasting over leading dims."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([w, x, y, z], dim=-1)


def positional_encoding(x, num_freqs):
    """[x, sin(2^i x), cos(2^i x)] along the last dim -> (..., C*(1+2*num_freqs))."""
    if num_freqs == 0:
        return x
    freqs = (2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype))
    xb = x.unsqueeze(-1) * freqs                      # (..., C, F)
    enc = torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)  # (..., C, 2F)
    enc = enc.flatten(start_dim=-2)                   # (..., C*2F)
    return torch.cat([x, enc], dim=-1)


# --------------------------------------------------------------------------------------
#  Sparse-graph utilities
# --------------------------------------------------------------------------------------
@torch.no_grad()
def chunked_knn(query, ref, k, chunk=65536):
    """k nearest `ref` for each `query` (squared-euclidean). Returns (idx, dist2) of shape (Q,k).

    Chunked over the query dim so the (Q x R) distance matrix never materialises in full.
    """
    R = ref.shape[0]
    k = min(k, R)
    if query.shape[0] == 0 or R == 0 or k == 0:   # degenerate: return correctly-shaped empties
        return (torch.zeros(query.shape[0], k, dtype=torch.long, device=query.device),
                torch.zeros(query.shape[0], k, device=query.device))
    idxs, dists = [], []
    for s in range(0, query.shape[0], chunk):
        q = query[s:s + chunk]
        d2 = torch.cdist(q, ref) ** 2          # (chunk, R)
        dd, ii = torch.topk(d2, k, dim=1, largest=False)
        idxs.append(ii)
        dists.append(dd)
    return torch.cat(idxs, 0), torch.cat(dists, 0)


@torch.no_grad()
def weighted_fps(xyz, weight, num_samples):
    """Weighted farthest-point sampling.

    Picks `num_samples` indices maximising (min distance to already-selected) * weight, so it
    keeps spatial coverage (FPS) while concentrating nodes in high-`weight` (high-motion)
    regions. With uniform weight it reduces to standard FPS. Returns a LongTensor of indices.
    """
    N = xyz.shape[0]
    num_samples = min(num_samples, N)
    sel = torch.zeros(num_samples, dtype=torch.long, device=xyz.device)
    dist = torch.full((N,), float("inf"), device=xyz.device)
    w = weight.clamp_min(1e-8)
    w = w / w.mean()
    far = int(torch.argmax(w).item())            # start from the highest-motion point
    for i in range(num_samples):
        sel[i] = far
        d = ((xyz - xyz[far]) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        score = dist * w
        far = int(torch.argmax(score).item())
    return sel


# --------------------------------------------------------------------------------------
#  The module
# --------------------------------------------------------------------------------------
class NodeGraphDeformation(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_nodes = getattr(args, "num_nodes", 1024)
        self.node_knn = getattr(args, "node_knn", 8)
        self.gauss_knn_K = getattr(args, "gauss_knn_K", 4)
        self.gnn_layers = getattr(args, "gnn_layers", 2)
        self.gnn_type = getattr(args, "gnn_type", "edgeconv")   # 'edgeconv' (mean agg) | 'gat' (attention agg)
        self.W = getattr(args, "gnn_width", 64)
        self.node_pe = getattr(args, "node_pe", 4)
        self.bind_sigma_scale = getattr(args, "bind_sigma_scale", 1.0)
        # cut-aware: break node-graph edges where tissue tears (deformed edge stretched past rest),
        # so motion on the two sides decouples instead of the HexPlane smearing across the cut.
        self.cut_aware = getattr(args, "cut_aware", False)
        self.cut_beta = getattr(args, "cut_beta", 5.0)
        self.cut_thresh = getattr(args, "cut_thresh", 1.3)

        pos_dim = 3 * (1 + 2 * self.node_pe)         # encoded node xyz
        time_dim = 1 + 2 * self.node_pe              # encoded scalar time
        edge_dim = 3 * (1 + 2 * self.node_pe)        # encoded relative edge vector
        self.edge_dim = edge_dim

        # control_route: feed the per-node control signal (edit_translation) as an INPUT to the GNN, so
        # message passing propagates sparse control through the graph (vs the default post-hoc translation
        # that bypasses the GNN). Adds 3 input dims for the control vector.
        self.control_route = getattr(args, "control_route", False)
        in_dim = pos_dim + time_dim + (3 if self.control_route else 0)
        self.input_mlp = nn.Sequential(
            nn.Linear(in_dim, self.W), nn.ReLU(),
            nn.Linear(self.W, self.W))

        self.msg_mlps = nn.ModuleList()
        self.upd_mlps = nn.ModuleList()
        self.att_mlps = nn.ModuleList()   # GAT-style per-edge attention scorer (only used when gnn_type=='gat')
        for _ in range(self.gnn_layers):
            self.msg_mlps.append(nn.Sequential(
                nn.Linear(2 * self.W + edge_dim, self.W), nn.ReLU(),
                nn.Linear(self.W, self.W)))
            self.upd_mlps.append(nn.Sequential(
                nn.Linear(2 * self.W, self.W), nn.ReLU(),
                nn.Linear(self.W, self.W)))
            # attention over the same [h_self, h_nbr, edge] context -> one logit per edge (GAT).
            self.att_mlps.append(nn.Linear(2 * self.W + edge_dim, 1))

        # SE(3) head: 3 translation + 6D rotation. Initialised to the identity transform.
        self.se3_head = nn.Linear(self.W, 9)
        self.init_identity_head()

        # Buffers (fixed shape so they (de)serialise cleanly; filled by seed_nodes()).
        self._register_state_buffers(args)

    def init_identity_head(self):
        """Zero weights + bias encoding identity SE(3) so the field starts as no deformation.
        Re-callable after a global weight init (deform_network applies xavier to all Linears)."""
        nn.init.zeros_(self.se3_head.weight)
        with torch.no_grad():
            self.se3_head.bias.zero_()
            self.se3_head.bias[3:9] = torch.tensor([1., 0., 0., 0., 1., 0.])  # -> identity R

    def _register_state_buffers(self, args):
        self.register_buffer("node_xyz", torch.zeros(self.num_nodes, 3))
        self.register_buffer("node_neighbors", torch.zeros(self.num_nodes, self.node_knn, dtype=torch.long))
        self.register_buffer("node_edge_enc", torch.zeros(self.num_nodes, self.node_knn, self.edge_dim))
        self.register_buffer("seeded", torch.zeros(1, dtype=torch.bool))
        # User edit handle: a per-node translation added on top of the learned motion. Zero during
        # training; set it (e.g. for a region of nodes) to drag tissue — the bound Gaussians follow
        # via LBS. This is the controllable-deformation capability the per-Gaussian baseline lacks.
        self.register_buffer("edit_translation", torch.zeros(self.num_nodes, 3))

    # ---- graph construction -----------------------------------------------------------
    @torch.no_grad()
    def seed_nodes(self, xyz, motion_weight):
        """(Re)place the nodes by motion-weighted FPS over `xyz`, then rebuild the node graph."""
        M = min(self.num_nodes, xyz.shape[0])
        sel = weighted_fps(xyz, motion_weight, M)
        node_xyz = xyz[sel].contiguous()
        nbr_idx, _ = chunked_knn(node_xyz, node_xyz, self.node_knn + 1)
        nbr_idx = nbr_idx[:, 1:]                       # drop self (column 0)
        rel = node_xyz[nbr_idx] - node_xyz.unsqueeze(1)   # (M,k,3)
        edge_enc = positional_encoding(rel, self.node_pe)
        # replace buffers (M may equal num_nodes; shapes stay constant across re-seeds)
        self.register_buffer("node_xyz", node_xyz)
        self.register_buffer("node_neighbors", nbr_idx.contiguous())
        self.register_buffer("node_edge_enc", edge_enc.contiguous())
        self.seeded = torch.ones(1, dtype=torch.bool, device=xyz.device)

    @torch.no_grad()
    def compute_binding(self, xyz):
        """Soft-bind each Gaussian to its K nearest nodes. Returns (idx (N,K) long, w (N,K))."""
        idx, dist2 = chunked_knn(xyz, self.node_xyz, self.gauss_knn_K)
        # softmax over -d^2 / sigma^2; sigma from the per-Gaussian nearest-node distance.
        sigma2 = dist2[:, :1].clamp_min(1e-8) * (self.bind_sigma_scale ** 2)
        w = torch.softmax(-dist2 / sigma2, dim=1)
        return idx, w

    # ---- forward ----------------------------------------------------------------------
    def _run_gnn(self, t, edge_gate=None, edit_input=None):
        """GNN at scalar time `t` with an optional per-edge gate (M,k); gate~0 cuts an edge's message.
        control_route: `edit_input` (M,3) sparse control fed as a node input (defaults to edit_translation)."""
        node_xyz = self.node_xyz
        M = node_xyz.shape[0]
        t_col = torch.full((M, 1), float(t), device=node_xyz.device, dtype=node_xyz.dtype)
        pos_enc = positional_encoding(node_xyz, self.node_pe)
        time_enc = positional_encoding(t_col, self.node_pe)
        feats = [pos_enc, time_enc]
        if self.control_route:
            if edit_input is not None:
                e = edit_input
            elif self.edit_translation.shape[0] == M:
                e = self.edit_translation                            # (M,3) control signal
            else:
                e = torch.zeros(M, 3, device=node_xyz.device, dtype=node_xyz.dtype)
            feats.append(e)
        h = self.input_mlp(torch.cat(feats, dim=-1))                 # (M,W)
        nbr = self.node_neighbors                                    # (M,k)
        for layer in range(self.gnn_layers):
            h_nbr = h[nbr]                                           # (M,k,W)
            h_self = h.unsqueeze(1).expand(-1, nbr.shape[1], -1)     # (M,k,W)
            ctx = torch.cat([h_self, h_nbr, self.node_edge_enc], dim=-1)
            msg = self.msg_mlps[layer](ctx)
            if self.gnn_type == "gat":
                # GAT-style: attention weights over neighbours (softmax of LeakyReLU logits),
                # optionally masked by the cut gate, then attention-weighted sum of messages.
                logit = F.leaky_relu(self.att_mlps[layer](ctx), 0.2)  # (M,k,1)
                if edge_gate is not None:
                    logit = logit + torch.log(edge_gate.unsqueeze(-1).clamp_min(1e-6))
                att = torch.softmax(logit, dim=1)                    # (M,k,1) over neighbours
                agg = (msg * att).sum(dim=1)                         # (M,W)
            elif edge_gate is None:
                agg = msg.mean(dim=1)                               # (M,W) mean aggregation
            else:
                g = edge_gate.unsqueeze(-1)                          # (M,k,1) gated aggregation
                agg = (msg * g).sum(dim=1) / (g.sum(dim=1) + 1e-6)
            h = h + self.upd_mlps[layer](torch.cat([h, agg], dim=-1))
        se3 = torch.nan_to_num(self.se3_head(h))                     # (M,9), guard non-finite
        trans = se3[:, 0:3]
        R = rotation_6d_to_matrix(se3[:, 3:9])                       # (M,3,3)
        if self.control_route:
            # control is routed THROUGH the GNN (fed as input above); the output already reflects the
            # propagated control, so use it directly — no post-hoc add, no bypass.
            return R, trans
        if self.edit_translation.shape == trans.shape:               # user edit handle (0 in training)
            trans = trans + self.edit_translation
        if getattr(self, "control_only", False):
            # eval-only (control-from-tracks): predict purely from the control handle, dropping the
            # learned per-node motion, so the metric isolates controllability (not reconstruction).
            M = trans.shape[0]
            R = torch.eye(3, device=trans.device, dtype=trans.dtype).expand(M, 3, 3)
            trans = (self.edit_translation if self.edit_translation.shape == trans.shape
                     else torch.zeros_like(trans))
        return R, trans

    def node_transforms(self, t):
        """Per-node SE(3) at time `t`. Cut-aware = 2 passes: an ungated pass measures per-edge
        stretch; edges stretched past `cut_thresh` are suppressed; a gated pass then lets the two
        sides of a cut deform independently (a discontinuity the continuous HexPlane cannot model)."""
        if not self.cut_aware:
            return self._run_gnn(t)
        with torch.no_grad():
            _, trans0 = self._run_gnn(t)                            # pass 1: ungated, measure strain
            nbr = self.node_neighbors
            p = self.node_xyz + trans0
            rest = (self.node_xyz.unsqueeze(1) - self.node_xyz[nbr]).norm(dim=-1)   # (M,k)
            cur = (p.unsqueeze(1) - p[nbr]).norm(dim=-1)
            stretch = cur / (rest + 1e-6)
            gate = torch.exp(-self.cut_beta * torch.relu(stretch - self.cut_thresh))  # (M,k); 0=cut
        return self._run_gnn(t, edge_gate=gate)                    # pass 2: gated

    def forward(self, point, rotations_emb, time_emb, binding_idx, binding_w):
        """LBS-blend per-node SE(3) onto each Gaussian.

        Returns deformed (positions (N,3), rotations-quaternion (N,4)).
        """
        t = time_emb.reshape(-1)[0] if time_emb.numel() > 0 else 0.0
        R, trans = self.node_transforms(t)                          # (M,3,3), (M,3)

        Rk = R[binding_idx]                                          # (N,K,3,3)
        tk = trans[binding_idx]                                      # (N,K,3)
        nk = self.node_xyz[binding_idx]                             # (N,K,3)
        xrel = point.unsqueeze(1) - nk                              # (N,K,3)
        warped = torch.einsum("nkij,nkj->nki", Rk, xrel) + nk + tk  # (N,K,3)
        w = binding_w.unsqueeze(-1)                                  # (N,K,1)
        pts = (w * warped).sum(dim=1)                               # (N,3)

        # rotation: weighted quaternion blend (sign-aligned to nearest node), then compose
        qk = matrix_to_quaternion(R)[binding_idx]                   # (N,K,4)
        ref = qk[:, 0:1, :]                                         # nearest node as sign ref
        sign = torch.sign((qk * ref).sum(-1, keepdim=True))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        q_node = F.normalize((binding_w.unsqueeze(-1) * qk * sign).sum(dim=1), dim=-1)
        q_can = F.normalize(rotations_emb[:, :4], dim=-1)
        rotations = quaternion_multiply(q_node, q_can)              # (N,4)
        # numerical safety: fall back to canonical where a degenerate node gave a non-finite value,
        # so NaNs can never reach the rasterizer (which faults with an illegal memory access).
        bad_p = ~torch.isfinite(pts).all(dim=1, keepdim=True)
        pts = torch.where(bad_p, point, pts)
        bad_r = ~torch.isfinite(rotations).all(dim=1, keepdim=True)
        rotations = torch.where(bad_r, q_can, rotations)
        return pts, rotations
