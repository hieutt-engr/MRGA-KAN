from networks.efficient_kan import KAN
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


# ============================================================
# Multi-relation GAT block
# ============================================================

# ============================================================
# Multi-relation GAT block
# ============================================================

class MultiRelationGATBlock(nn.Module):
    """
    One block for multi-relational graph attention.
    Each relation has its own GATv2Conv, then fused by learnable gates.
    """

    def __init__(
        self,
        hidden_dim,
        edge_attr_dim,
        num_relations=4,
        heads=4,
        rel_emb_dim=8,
        dropout=0.2,
        ffn_ratio=2.0,
    ):
        super(MultiRelationGATBlock, self).__init__()

        self.hidden_dim = hidden_dim
        self.edge_attr_dim = edge_attr_dim
        self.num_relations = num_relations
        self.heads = heads
        self.rel_emb_dim = rel_emb_dim
        self.dropout = dropout

        self.rel_emb = nn.Embedding(num_relations, rel_emb_dim)

        self.rel_convs = nn.ModuleList([
            GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=heads,
                concat=False,
                edge_dim=edge_attr_dim + rel_emb_dim,
                dropout=dropout,
                add_self_loops=False,
                bias=True,
            )
            for _ in range(num_relations)
        ])

        self.rel_gate = nn.Parameter(torch.zeros(num_relations))

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

        ffn_hidden = int(hidden_dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_dim),
        )

    def forward(self, h, edge_index, edge_attr, edge_type):
        device = h.device
        relation_outputs = []
        active_relations = []

        for r in range(self.num_relations):
            mask = (edge_type == r)
            if mask.sum().item() == 0:
                continue

            ei_r = edge_index[:, mask]
            ea_r = edge_attr[mask]

            rel_vec = self.rel_emb.weight[r].unsqueeze(0).expand(ea_r.size(0), -1)
            ea_r_full = torch.cat([ea_r, rel_vec], dim=-1)

            h_r = self.rel_convs[r](h, ei_r, ea_r_full)
            relation_outputs.append(h_r)
            active_relations.append(r)

        if len(relation_outputs) == 0:
            h_msg = h
        else:
            active_relations_t = torch.tensor(active_relations, device=device, dtype=torch.long)
            gate_scores = self.rel_gate[active_relations_t]
            gate_weights = torch.softmax(gate_scores, dim=0)

            h_msg = torch.zeros_like(h)
            for w, h_r in zip(gate_weights, relation_outputs):
                h_msg = h_msg + w * h_r

        h = self.norm1(h + self.drop(h_msg))
        h_ffn = self.ffn(h)
        h = self.norm2(h + self.drop(h_ffn))
        return h


# ============================================================
# GraphAttentionKAN
# ============================================================

class GraphAttentionKAN(nn.Module):
    """
    Multi-relation Graph Attention Encoder + KAN Head
    for graph-level classification.
    """

    def __init__(
        self,
        node_feat_dim,
        edge_attr_dim,
        num_classes,
        num_ids,
        hidden_dim=128,
        num_layers=3,
        heads=4,
        id_emb_dim=32,
        rel_emb_dim=8,
        num_relations=4,
        dropout=0.2,
        ffn_ratio=2.0,
        kan_hidden=128,
        kan_grid_size=5,
        kan_spline_order=3,
        kan_scale_noise=0.1,
        kan_scale_base=1.0,
        kan_scale_spline=1.0,
    ):
        super(GraphAttentionKAN, self).__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_attr_dim = edge_attr_dim
        self.num_classes = num_classes
        self.num_ids = num_ids
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.dropout = dropout

        self.id_embedding = nn.Embedding(num_ids + 1, id_emb_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim + id_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList([
            MultiRelationGATBlock(
                hidden_dim=hidden_dim,
                edge_attr_dim=edge_attr_dim,
                num_relations=num_relations,
                heads=heads,
                rel_emb_dim=rel_emb_dim,
                dropout=dropout,
                ffn_ratio=ffn_ratio,
            )
            for _ in range(num_layers)
        ])

        self.readout_norm = nn.LayerNorm(hidden_dim * 2)

        self.head = KAN(
            layers_hidden=[hidden_dim * 2, kan_hidden, num_classes],
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
            scale_noise=kan_scale_noise,
            scale_base=kan_scale_base,
            scale_spline=kan_scale_spline,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=(-1, 1),
        )

    def encode_nodes(self, data):
        x = data.x
        if hasattr(data, "id_token"):
            id_token = data.id_token
        elif hasattr(data, "id_index"):
            id_token = data.id_index
        else:
            raise AttributeError("Batch data must contain 'id_token' or 'id_index'.")

        id_emb = self.id_embedding(id_token)
        h = torch.cat([x, id_emb], dim=-1)
        h = self.input_proj(h)

        for block in self.blocks:
            h = block(
                h=h,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                edge_type=data.edge_type,
            )

        return h

    def readout(self, h, batch):
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        g = torch.cat([h_mean, h_max], dim=-1)
        g = self.readout_norm(g)
        return g

    def forward(self, data, update_grid=False, return_graph_embedding=False):
        if hasattr(data, "batch"):
            batch = data.batch
        else:
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

        h = self.encode_nodes(data)
        g = self.readout(h, batch)
        logits = self.head(g, update_grid=update_grid)

        if return_graph_embedding:
            return logits, g
        return logits

    def kan_regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return self.head.regularization_loss(
            regularize_activation=regularize_activation,
            regularize_entropy=regularize_entropy,
        )

    def compute_loss(
        self,
        logits,
        y,
        class_weights=None,
        label_smoothing=0.0,
        kan_reg_lambda=1e-4,
        reg_activation=1.0,
        reg_entropy=1.0,
    ):
        ce = F.cross_entropy(
            logits,
            y,
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

        kan_reg = self.kan_regularization_loss(
            regularize_activation=reg_activation,
            regularize_entropy=reg_entropy,
        )

        loss = ce + kan_reg_lambda * kan_reg

        stats = {
            "loss": float(loss.detach().cpu()),
            "ce": float(ce.detach().cpu()),
            "kan_reg": float(kan_reg.detach().cpu()),
        }
        return loss, stats