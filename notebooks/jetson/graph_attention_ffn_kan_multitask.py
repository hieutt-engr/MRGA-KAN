from networks.efficient_kan import KAN
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


# ============================================================
# KAN FeedForward
# ============================================================
class KANFeedForward(nn.Module):
    """
    KAN-based FFN replacing:
        Linear -> SiLU -> Dropout -> Linear
    """
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        dropout=0.2,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        grid_eps=0.02,
        grid_range=(-1, 1),
    ):
        super().__init__()

        self.kan = KAN(
            layers_hidden=[in_dim, hidden_dim, out_dim],
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=torch.nn.SiLU,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, update_grid=False):
        x = self.kan(x, update_grid=update_grid)
        x = self.drop(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return self.kan.regularization_loss(
            regularize_activation=regularize_activation,
            regularize_entropy=regularize_entropy,
        )


# ============================================================
# Multi-relation GAT block
# ============================================================
class MultiRelationGATBlock(nn.Module):
    """
    One block for multi-relational graph attention.
    Each relation has its own GATv2Conv, then fused by learnable gates.
    FFN is replaced by KAN-FFN.
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
        # KAN params for block FFN
        block_kan_grid_size=5,
        block_kan_spline_order=3,
        block_kan_scale_noise=0.1,
        block_kan_scale_base=1.0,
        block_kan_scale_spline=1.0,
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

        self.ffn = KANFeedForward(
            in_dim=hidden_dim,
            hidden_dim=ffn_hidden,
            out_dim=hidden_dim,
            dropout=dropout,
            grid_size=block_kan_grid_size,
            spline_order=block_kan_spline_order,
            scale_noise=block_kan_scale_noise,
            scale_base=block_kan_scale_base,
            scale_spline=block_kan_scale_spline,
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
# GraphAttentionKAN (multitask)
# ============================================================
class GraphAttentionKAN(nn.Module):
    """
    Multi-relation Graph Attention Encoder + KAN graph head
    with optional node-classification head for multitask learning.
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
        node_head_from_block=1,
        heads=4,
        id_emb_dim=32,
        rel_emb_dim=8,
        num_relations=4,
        dropout=0.2,
        ffn_ratio=2.0,
        # Block KAN params
        block_kan_grid_size=5,
        block_kan_spline_order=3,
        block_kan_scale_noise=0.1,
        block_kan_scale_base=1.0,
        block_kan_scale_spline=1.0,
        # Graph head KAN params
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
                block_kan_grid_size=block_kan_grid_size,
                block_kan_spline_order=block_kan_spline_order,
                block_kan_scale_noise=block_kan_scale_noise,
                block_kan_scale_base=block_kan_scale_base,
                block_kan_scale_spline=block_kan_scale_spline,
            )
            for _ in range(num_layers)
        ])

        self.readout_norm = nn.LayerNorm(hidden_dim * 2)

        # graph/window head
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

        # node head for multitask
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

    def encode_nodes(self, data, update_grid=False):
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
                update_grid=update_grid,
            )

        return h

    def readout(self, h, batch):
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        g = torch.cat([h_mean, h_max], dim=-1)
        g = self.readout_norm(g)
        return g

    def classify_nodes(self, h):
        if self.node_head is None:
            return None
        return self.node_head(h)

    def classify_graph(self, g, update_grid=False):
        return self.head(g, update_grid=update_grid)

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

        h = self.encode_nodes(data, update_grid=update_grid)
        g = self.readout(h, batch)
        graph_logits = self.classify_graph(g, update_grid=update_grid)
        node_logits = self.classify_nodes(h) if (return_node_logits or self.node_head is not None) else None

        if return_graph_embedding and return_node_logits:
            return {
                "graph_logits": graph_logits,
                "node_logits": node_logits,
                "graph_embedding": g,
                "node_embedding": h,
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
        head_reg = self.head.regularization_loss(
            regularize_activation=regularize_activation,
            regularize_entropy=regularize_entropy,
        )

        block_reg = 0.0
        for block in self.blocks:
            block_reg = block_reg + block.kan_regularization_loss(
                regularize_activation=regularize_activation,
                regularize_entropy=regularize_entropy,
            )

        return head_reg + block_reg

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
