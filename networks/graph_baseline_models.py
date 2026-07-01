"""Graph baseline models for the IVN dynamic graph pipeline.

This file is designed to be compatible with the graph objects produced by
`build_graphs_node_classification.py` and with the multitask training logic in
`train_gat_ffn_kan_multitask.py`.

Expected PyG Data fields
------------------------
Required:
    data.x          : FloatTensor [num_nodes, node_feat_dim]
    data.edge_index : LongTensor  [2, num_edges]
    data.y          : LongTensor  [num_graphs]

Used when available:
    data.edge_attr  : FloatTensor [num_edges, edge_attr_dim]
    data.edge_type  : LongTensor  [num_edges]
    data.id_token   : LongTensor  [num_nodes]
    data.id_index   : LongTensor  [num_nodes]
    data.batch      : LongTensor  [num_nodes]

Forward output follows the same convention as GraphAttentionKAN:
    model(data, return_node_logits=True) -> {
        "graph_logits": Tensor [num_graphs, num_classes],
        "node_logits" : Tensor [num_nodes, num_node_classes] or None,
    }

Recommended baselines
---------------------
1. GCNBaseline
2. GraphSAGEBaseline
3. GINBaseline
4. GATBaseline
5. GATv2Baseline
6. EdgeGATv2Baseline
7. TransformerConvBaseline
8. RGCNBaseline
9. MultiRelationGATBaseline

The last one is the strongest ablation baseline because it keeps your
multi-relation graph design but replaces the KAN-FFN and KAN head with standard
MLP blocks.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    RGCNConv,
    SAGEConv,
    TransformerConv,
    global_max_pool,
    global_mean_pool,
)


# ============================================================================
# Small shared modules
# ============================================================================


class MLP(nn.Module):
    """Simple MLP used for graph head, node head, and GIN update network."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.2,
        num_layers: int = 2,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    if use_layer_norm:
                        layers.append(nn.LayerNorm(dims[i + 1]))
                    layers.append(nn.SiLU())
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FFNBlock(nn.Module):
    """Transformer-style residual FFN block with LayerNorm."""

    def __init__(self, hidden_dim: int, ffn_ratio: float = 2.0, dropout: float = 0.2) -> None:
        super().__init__()
        ffn_hidden = int(hidden_dim * ffn_ratio)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.drop(self.ffn(self.norm(h)))


# ============================================================================
# Homogeneous graph baselines
# ============================================================================


class HomogeneousConvBlock(nn.Module):
    """One residual message-passing block for non-relational baselines.

    These models treat all edges as one homogeneous edge set. This is useful for
    testing whether the proposed relation-aware design is really helpful.
    """

    def __init__(
        self,
        conv_type: str,
        hidden_dim: int,
        edge_attr_dim: int,
        num_relations: int = 4,
        heads: int = 4,
        dropout: float = 0.2,
        ffn_ratio: float = 2.0,
        use_edge_attr: bool = False,
    ) -> None:
        super().__init__()
        self.conv_type = conv_type.lower()
        self.use_edge_attr = use_edge_attr
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = FFNBlock(hidden_dim, ffn_ratio=ffn_ratio, dropout=dropout)

        if self.conv_type == "gcn":
            self.conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)
        elif self.conv_type == "sage":
            self.conv = SAGEConv(hidden_dim, hidden_dim)
        elif self.conv_type == "gin":
            gin_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.conv = GINConv(gin_mlp, train_eps=True)
        elif self.conv_type == "gat":
            edge_dim = edge_attr_dim if use_edge_attr else None
            self.conv = GATConv(
                hidden_dim,
                hidden_dim,
                heads=heads,
                concat=False,
                dropout=dropout,
                edge_dim=edge_dim,
                add_self_loops=True,
            )
        elif self.conv_type == "gatv2":
            edge_dim = edge_attr_dim if use_edge_attr else None
            self.conv = GATv2Conv(
                hidden_dim,
                hidden_dim,
                heads=heads,
                concat=False,
                dropout=dropout,
                edge_dim=edge_dim,
                add_self_loops=True,
            )
        elif self.conv_type == "transformer":
            edge_dim = edge_attr_dim if use_edge_attr else None
            self.conv = TransformerConv(
                hidden_dim,
                hidden_dim,
                heads=heads,
                concat=False,
                beta=True,
                dropout=dropout,
                edge_dim=edge_dim,
            )
        elif self.conv_type == "rgcn":
            self.conv = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.conv_type == "rgcn":
            if edge_type is None:
                raise ValueError("RGCNBaseline requires data.edge_type")
            h_msg = self.conv(h, edge_index, edge_type)
        elif self.use_edge_attr and self.conv_type in {"gat", "gatv2", "transformer"}:
            if edge_attr is None:
                raise ValueError(f"{self.conv_type} with use_edge_attr=True requires data.edge_attr")
            h_msg = self.conv(h, edge_index, edge_attr=edge_attr)
        else:
            h_msg = self.conv(h, edge_index)

        h = self.norm1(h + self.drop(h_msg))
        h = self.ffn(h)
        return h


class HomogeneousGraphBaseline(nn.Module):
    """Generic baseline for GCN, GraphSAGE, GIN, GAT, GATv2, R-GCN, TransformerConv.

    The constructor intentionally accepts most arguments used by GraphAttentionKAN
    so that this class can replace it with minimal changes in your training script.
    KAN-specific arguments are accepted and ignored.
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_attr_dim: int,
        num_classes: int,
        num_ids: int,
        num_node_classes: Optional[int] = None,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        id_emb_dim: int = 32,
        rel_emb_dim: int = 8,  # accepted for API compatibility
        num_relations: int = 4,
        dropout: float = 0.2,
        ffn_ratio: float = 2.0,
        node_head_from_layer: Optional[int] = None,
        conv_type: str = "gcn",
        use_edge_attr: bool = False,
        graph_head_hidden: Optional[int] = None,
        # ignored KAN-compatible arguments
        block_kan_grid_size: int = 5,
        block_kan_spline_order: int = 3,
        block_kan_scale_noise: float = 0.1,
        block_kan_scale_base: float = 1.0,
        block_kan_scale_spline: float = 1.0,
        kan_hidden: int = 128,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
        kan_scale_noise: float = 0.1,
        kan_scale_base: float = 1.0,
        kan_scale_spline: float = 1.0,
        **_: object,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.node_feat_dim = node_feat_dim
        self.edge_attr_dim = edge_attr_dim
        self.num_classes = num_classes
        self.num_node_classes = num_node_classes
        self.num_ids = num_ids
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.conv_type = conv_type.lower()
        self.use_edge_attr = use_edge_attr

        if node_head_from_layer is None:
            node_head_from_layer = max(0, num_layers - 2)
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

        self.blocks = nn.ModuleList(
            [
                HomogeneousConvBlock(
                    conv_type=self.conv_type,
                    hidden_dim=hidden_dim,
                    edge_attr_dim=edge_attr_dim,
                    num_relations=num_relations,
                    heads=heads,
                    dropout=dropout,
                    ffn_ratio=ffn_ratio,
                    use_edge_attr=use_edge_attr,
                )
                for _ in range(num_layers)
            ]
        )

        self.readout_norm = nn.LayerNorm(hidden_dim * 2)
        head_hidden = graph_head_hidden if graph_head_hidden is not None else kan_hidden
        self.head = MLP(
            in_dim=hidden_dim * 2,
            hidden_dim=head_hidden,
            out_dim=num_classes,
            dropout=dropout,
            num_layers=2,
        )

        if num_node_classes is not None and num_node_classes > 0:
            self.node_head = MLP(
                in_dim=hidden_dim,
                hidden_dim=hidden_dim,
                out_dim=num_node_classes,
                dropout=dropout,
                num_layers=2,
            )
        else:
            self.node_head = None

    def _get_id_token(self, data) -> torch.Tensor:
        if hasattr(data, "id_token"):
            return data.id_token.long()
        if hasattr(data, "id_index"):
            return data.id_index.long()
        raise AttributeError("Batch data must contain 'id_token' or 'id_index'.")

    def _get_batch(self, data) -> torch.Tensor:
        if hasattr(data, "batch"):
            return data.batch
        return torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

    def encode_nodes(self, data, return_all_hidden: bool = False) -> Tuple[torch.Tensor, Optional[list]]:
        id_emb = self.id_embedding(self._get_id_token(data))
        h = torch.cat([data.x.float(), id_emb], dim=-1)
        h = self.input_proj(h)

        edge_attr = data.edge_attr.float() if hasattr(data, "edge_attr") else None
        edge_type = data.edge_type.long() if hasattr(data, "edge_type") else None

        hidden_states = []
        for block in self.blocks:
            h = block(
                h=h,
                edge_index=data.edge_index.long(),
                edge_attr=edge_attr,
                edge_type=edge_type,
            )
            hidden_states.append(h)

        if return_all_hidden:
            return h, hidden_states
        return h, None

    def readout(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        g = torch.cat([h_mean, h_max], dim=-1)
        return self.readout_norm(g)

    def classify_graph(self, g: torch.Tensor) -> torch.Tensor:
        return self.head(g)

    def classify_nodes(self, h_node: torch.Tensor) -> Optional[torch.Tensor]:
        if self.node_head is None:
            return None
        return self.node_head(h_node)

    def forward(
        self,
        data,
        update_grid: bool = False,  # accepted for compatibility; unused
        return_graph_embedding: bool = False,
        return_node_logits: bool = False,
    ):
        batch = self._get_batch(data)
        h_final, hidden_states = self.encode_nodes(data, return_all_hidden=True)
        assert hidden_states is not None

        g = self.readout(h_final, batch)
        graph_logits = self.classify_graph(g)

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
            return {"graph_logits": graph_logits, "node_logits": node_logits}
        return graph_logits

    def kan_regularization_loss(self, *args, **kwargs) -> torch.Tensor:
        """Return zero so existing training code can keep calling this safely."""
        device = next(self.parameters()).device
        return torch.tensor(0.0, device=device)


# Concrete homogeneous baselines ------------------------------------------------


class GCNBaseline(HomogeneousGraphBaseline):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, conv_type="gcn", use_edge_attr=False, **kwargs)


class GraphSAGEBaseline(HomogeneousGraphBaseline):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, conv_type="sage", use_edge_attr=False, **kwargs)


class GINBaseline(HomogeneousGraphBaseline):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, conv_type="gin", use_edge_attr=False, **kwargs)


class GATBaseline(HomogeneousGraphBaseline):
    """Classic GAT baseline; ignores edge_attr and edge_type."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, conv_type="gat", use_edge_attr=False, **kwargs)


class GATv2Baseline(HomogeneousGraphBaseline):
    """Classic GATv2 baseline; ignores edge_attr and edge_type."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, conv_type="gatv2", use_edge_attr=False, **kwargs)


class EdgeGATv2Baseline(HomogeneousGraphBaseline):
    """GATv2 baseline using edge_attr but still ignoring edge_type."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, conv_type="gatv2", use_edge_attr=True, **kwargs)


class TransformerConvBaseline(HomogeneousGraphBaseline):
    """Graph Transformer-style baseline using PyG TransformerConv and edge_attr."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, conv_type="transformer", use_edge_attr=True, **kwargs)


class RGCNBaseline(HomogeneousGraphBaseline):
    """Relational GCN baseline using edge_type but ignoring edge_attr."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, conv_type="rgcn", use_edge_attr=False, **kwargs)


# ============================================================================
# Multi-relation GAT baseline without KAN
# ============================================================================


class MultiRelationGATLinearBlock(nn.Module):
    """Relation-specific GATv2 block with standard MLP FFN.

    This is the cleanest ablation against your proposed model:
    - Same relation-aware message passing idea.
    - Same relation embedding idea.
    - Same mean+max graph readout idea.
    - No KAN-FFN.
    - No KAN graph head.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_attr_dim: int,
        num_relations: int = 4,
        heads: int = 4,
        rel_emb_dim: int = 8,
        dropout: float = 0.2,
        ffn_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_attr_dim = edge_attr_dim
        self.num_relations = num_relations
        self.rel_emb_dim = rel_emb_dim

        self.rel_emb = nn.Embedding(num_relations, rel_emb_dim)
        self.rel_convs = nn.ModuleList(
            [
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
            ]
        )
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

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        device = h.device
        relation_outputs = []
        active_relations = []

        for r in range(self.num_relations):
            mask = edge_type == r
            if int(mask.sum().item()) == 0:
                continue

            ei_r = edge_index[:, mask]
            ea_r = edge_attr[mask]
            rel_vec = self.rel_emb.weight[r].unsqueeze(0).expand(ea_r.size(0), -1)
            ea_r_full = torch.cat([ea_r, rel_vec], dim=-1)
            h_r = self.rel_convs[r](h, ei_r, ea_r_full)
            relation_outputs.append(h_r)
            active_relations.append(r)

        if len(relation_outputs) == 0:
            h_msg = torch.zeros_like(h)
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


class MultiRelationGATBaseline(nn.Module):
    """Multi-relation GATv2 + standard MLP baseline.

    This model is intentionally close to the proposed model architecture but
    removes KAN components. It is therefore suitable for the main ablation table.
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_attr_dim: int,
        num_classes: int,
        num_ids: int,
        num_node_classes: Optional[int] = None,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        id_emb_dim: int = 32,
        rel_emb_dim: int = 8,
        num_relations: int = 4,
        dropout: float = 0.2,
        ffn_ratio: float = 2.0,
        node_head_from_layer: Optional[int] = None,
        graph_head_hidden: Optional[int] = None,
        # ignored KAN-compatible arguments
        block_kan_grid_size: int = 5,
        block_kan_spline_order: int = 3,
        block_kan_scale_noise: float = 0.1,
        block_kan_scale_base: float = 1.0,
        block_kan_scale_spline: float = 1.0,
        kan_hidden: int = 128,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
        kan_scale_noise: float = 0.1,
        kan_scale_base: float = 1.0,
        kan_scale_spline: float = 1.0,
        **_: object,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.node_feat_dim = node_feat_dim
        self.edge_attr_dim = edge_attr_dim
        self.num_classes = num_classes
        self.num_node_classes = num_node_classes
        self.num_ids = num_ids
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_relations = num_relations

        if node_head_from_layer is None:
            node_head_from_layer = max(0, num_layers - 2)
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

        self.blocks = nn.ModuleList(
            [
                MultiRelationGATLinearBlock(
                    hidden_dim=hidden_dim,
                    edge_attr_dim=edge_attr_dim,
                    num_relations=num_relations,
                    heads=heads,
                    rel_emb_dim=rel_emb_dim,
                    dropout=dropout,
                    ffn_ratio=ffn_ratio,
                )
                for _ in range(num_layers)
            ]
        )

        self.readout_norm = nn.LayerNorm(hidden_dim * 2)
        head_hidden = graph_head_hidden if graph_head_hidden is not None else kan_hidden
        self.head = MLP(hidden_dim * 2, head_hidden, num_classes, dropout=dropout, num_layers=2)

        if num_node_classes is not None and num_node_classes > 0:
            self.node_head = MLP(hidden_dim, hidden_dim, num_node_classes, dropout=dropout, num_layers=2)
        else:
            self.node_head = None

    def _get_id_token(self, data) -> torch.Tensor:
        if hasattr(data, "id_token"):
            return data.id_token.long()
        if hasattr(data, "id_index"):
            return data.id_index.long()
        raise AttributeError("Batch data must contain 'id_token' or 'id_index'.")

    def _get_batch(self, data) -> torch.Tensor:
        if hasattr(data, "batch"):
            return data.batch
        return torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

    def encode_nodes(self, data, return_all_hidden: bool = False):
        if not hasattr(data, "edge_attr"):
            raise AttributeError("MultiRelationGATBaseline requires data.edge_attr")
        if not hasattr(data, "edge_type"):
            raise AttributeError("MultiRelationGATBaseline requires data.edge_type")

        id_emb = self.id_embedding(self._get_id_token(data))
        h = torch.cat([data.x.float(), id_emb], dim=-1)
        h = self.input_proj(h)

        hidden_states = []
        for block in self.blocks:
            h = block(
                h=h,
                edge_index=data.edge_index.long(),
                edge_attr=data.edge_attr.float(),
                edge_type=data.edge_type.long(),
            )
            hidden_states.append(h)

        if return_all_hidden:
            return h, hidden_states
        return h, None

    def readout(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        g = torch.cat([h_mean, h_max], dim=-1)
        return self.readout_norm(g)

    def forward(
        self,
        data,
        update_grid: bool = False,  # accepted for compatibility; unused
        return_graph_embedding: bool = False,
        return_node_logits: bool = False,
    ):
        batch = self._get_batch(data)
        h_final, hidden_states = self.encode_nodes(data, return_all_hidden=True)
        assert hidden_states is not None

        g = self.readout(h_final, batch)
        graph_logits = self.head(g)

        h_node = hidden_states[self.node_head_from_layer]
        node_logits = self.node_head(h_node) if (self.node_head is not None and return_node_logits) else None

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
            return {"graph_logits": graph_logits, "node_logits": node_logits}
        return graph_logits

    def kan_regularization_loss(self, *args, **kwargs) -> torch.Tensor:
        device = next(self.parameters()).device
        return torch.tensor(0.0, device=device)


# ============================================================================
# Factory
# ============================================================================


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "gcn": GCNBaseline,
    "graphsage": GraphSAGEBaseline,
    "sage": GraphSAGEBaseline,
    "gin": GINBaseline,
    "gat": GATBaseline,
    "gatv2": GATv2Baseline,
    "edge_gatv2": EdgeGATv2Baseline,
    "transformer": TransformerConvBaseline,
    "transformerconv": TransformerConvBaseline,
    "rgcn": RGCNBaseline,
    "multi_relation_gat": MultiRelationGATBaseline,
    "mrgat_mlp": MultiRelationGATBaseline,
}


SUPPORTED_BASELINES = tuple(sorted(MODEL_REGISTRY.keys()))


def create_graph_baseline(model_name: str, **kwargs) -> nn.Module:
    """Create a graph baseline by name.

    Example:
        model = create_graph_baseline(
            "gatv2",
            node_feat_dim=16,
            edge_attr_dim=6,
            num_classes=5,
            num_ids=512,
            num_node_classes=5,
            hidden_dim=128,
            num_layers=3,
        )
    """
    key = model_name.lower().strip()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown baseline model '{model_name}'. Supported models: {', '.join(SUPPORTED_BASELINES)}"
        )
    return MODEL_REGISTRY[key](**kwargs)
