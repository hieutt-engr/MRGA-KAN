# networks/graph_attention_encoder.py
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
from networks.graph_attention_kan import MultiRelationGATBlock

class GraphAttentionEncoder(nn.Module):
    def __init__(
        self,
        node_feat_dim,
        edge_attr_dim,
        num_ids,
        hidden_dim=128,
        num_layers=3,
        heads=4,
        id_emb_dim=32,
        rel_emb_dim=8,
        num_relations=4,
        dropout=0.2,
        ffn_ratio=2.0,
    ):
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_attr_dim = edge_attr_dim
        self.num_ids = num_ids
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_relations = num_relations

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
        h_max  = global_max_pool(h, batch)
        g = torch.cat([h_mean, h_max], dim=-1)
        g = self.readout_norm(g)
        return g

    def forward(self, data, return_node_embedding=False):
        if hasattr(data, "batch"):
            batch = data.batch
        else:
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

        h = self.encode_nodes(data)
        g = self.readout(h, batch)

        if return_node_embedding:
            return h, g
        return g