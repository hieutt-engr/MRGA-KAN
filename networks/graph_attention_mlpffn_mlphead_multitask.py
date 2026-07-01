import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


# ============================================================
# Standard MLP FeedForward
# ============================================================
class MLPFeedForward(nn.Module):
    """
    Standard FFN:
        Linear -> SiLU -> Dropout -> Linear
    """
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, update_grid=False):
        return self.net(x)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        device = next(self.parameters()).device
        return torch.tensor(0.0, device=device)


# ============================================================
# Multi-relation GAT block (MLP FFN)
# ============================================================
class MultiRelationGATBlock(nn.Module):
    """
    One block for multi-relational graph attention.
    Each relation has its own GATv2Conv, then fused by learnable gates.
    FFN uses standard MLP.
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
        **kwargs,
    ):
        super().__init__()

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
        self.ffn = MLPFeedForward(
            in_dim=hidden_dim,
            hidden_dim=ffn_hidden,
            out_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, h, edge_index, edge_attr, edge_type, update_grid=False):
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
        h_ffn = self.ffn(h, update_grid=update_grid)
        h = self.norm2(h + self.drop(h_ffn))
        return h

    def kan_regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return self.ffn.regularization_loss(
            regularize_activation=regularize_activation,
            regularize_entropy=regularize_entropy,
        )


# ============================================================
# GraphAttentionMLPHead (multitask, early node head)
# ============================================================
class GraphAttentionKAN(nn.Module):
    """
    Multi-relation Graph Attention Encoder + MLP graph head
    with optional node-classification head for multitask learning.

    Variant:
    - no KAN FFN in backbone
    - no KAN graph head
    - keep MLP node head
    """

    def __init__(
        self,
        node_feat_dim,
        edge_attr_dim,
        num_classes,
        num_ids,
        num_node_classes=None,
        hidden_dim=128,
        num_layers=3,
        heads=4,
        id_emb_dim=32,
        rel_emb_dim=8,
        num_relations=4,
        dropout=0.2,
        ffn_ratio=2.0,
        node_head_from_layer=None,
        # ignored KAN params kept for interface compatibility
        block_kan_grid_size=5,
        block_kan_spline_order=3,
        block_kan_scale_noise=0.1,
        block_kan_scale_base=1.0,
        block_kan_scale_spline=1.0,
        kan_hidden=128,
        kan_grid_size=5,
        kan_spline_order=3,
        kan_scale_noise=0.1,
        kan_scale_base=1.0,
        kan_scale_spline=1.0,
    ):
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_attr_dim = edge_attr_dim
        self.num_classes = num_classes
        self.num_node_classes = num_node_classes
        self.num_ids = num_ids
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.dropout = dropout

        if node_head_from_layer is None:
            if num_layers <= 1:
                node_head_from_layer = 0
            else:
                node_head_from_layer = num_layers - 2
        if node_head_from_layer < 0:
            node_head_from_layer = num_layers + node_head_from_layer
        if not (0 <= node_head_from_layer < num_layers):
            raise ValueError(
                f"node_head_from_layer must be in [0, {num_layers - 1}], got {node_head_from_layer}"
            )
        self.node_head_from_layer = node_head_from_layer

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

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, kan_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(kan_hidden, num_classes),
        )

        if num_node_classes is not None and num_node_classes > 0:
            self.node_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_node_classes),
            )
        else:
            self.node_head = None

    def encode_nodes(self, data, update_grid=False, return_all_hidden=False):
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

        hidden_states = []
        for block in self.blocks:
            h = block(
                h=h,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                edge_type=data.edge_type,
                update_grid=update_grid,
            )
            hidden_states.append(h)

        if return_all_hidden:
            return h, hidden_states
        return h

    def readout(self, h, batch):
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        g = torch.cat([h_mean, h_max], dim=-1)
        g = self.readout_norm(g)
        return g

    def classify_nodes(self, h_node):
        if self.node_head is None:
            return None
        return self.node_head(h_node)

    def classify_graph(self, g, update_grid=False):
        return self.head(g)

    def forward(
        self,
        data,
        update_grid=False,
        return_graph_embedding=False,
        return_node_logits=False,
    ):
        if hasattr(data, "batch"):
            batch = data.batch
        else:
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

        h_final, hidden_states = self.encode_nodes(
            data,
            update_grid=update_grid,
            return_all_hidden=True,
        )

        g = self.readout(h_final, batch)
        graph_logits = self.classify_graph(g, update_grid=update_grid)

        h_node = hidden_states[self.node_head_from_layer]
        node_logits = self.classify_nodes(h_node) if (return_node_logits or self.node_head is not None) else None

        if return_graph_embedding and return_node_logits:
            return {
                "graph_logits": graph_logits,
                "node_logits": node_logits,
                "graph_embedding": g,
                "node_embedding": h_node,
                "graph_backbone_embedding": h_final,
            }
        if return_graph_embedding:
            return graph_logits, g
        if return_node_logits:
            return {
                "graph_logits": graph_logits,
                "node_logits": node_logits,
            }
        return graph_logits

    def kan_regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        device = next(self.parameters()).device
        return torch.tensor(0.0, device=device)

    def compute_loss(
        self,
        graph_logits,
        y,
        class_weights=None,
        label_smoothing=0.0,
        kan_reg_lambda=1e-4,
        reg_activation=1.0,
        reg_entropy=1.0,
        node_logits=None,
        node_y=None,
        node_mask=None,
        node_loss_weight=1.0,
        node_class_weights=None,
    ):
        graph_ce = F.cross_entropy(
            graph_logits,
            y,
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

        node_ce = torch.tensor(0.0, device=graph_logits.device)
        if node_logits is not None and node_y is not None:
            if node_mask is None:
                valid = torch.ones_like(node_y, dtype=torch.bool)
            else:
                valid = node_mask.bool()
            if valid.any():
                node_ce = F.cross_entropy(
                    node_logits[valid],
                    node_y[valid],
                    weight=node_class_weights,
                )

        kan_reg = self.kan_regularization_loss(
            regularize_activation=reg_activation,
            regularize_entropy=reg_entropy,
        )

        loss = graph_ce + node_loss_weight * node_ce + kan_reg_lambda * kan_reg

        stats = {
            "loss": float(loss.detach().cpu()),
            "graph_ce": float(graph_ce.detach().cpu()),
            "node_ce": float(node_ce.detach().cpu()),
            "kan_reg": float(kan_reg.detach().cpu()),
        }
        return loss, stats
