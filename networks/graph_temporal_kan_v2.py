# networks/graph_temporal_kan_v2.py
import torch
import torch.nn as nn
from networks.efficient_kan import KAN

class SequenceAttentionPooling(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, x, mask=None):
        # x: [B, L, D]
        score = self.score(x).squeeze(-1)   # [B, L]

        if mask is not None:
            score = score.masked_fill(~mask, -1e9)

        attn = torch.softmax(score, dim=1)  # [B, L]
        pooled = torch.sum(x * attn.unsqueeze(-1), dim=1)  # [B, D]
        return pooled, attn


class GraphTemporalKANv2(nn.Module):
    def __init__(
        self,
        graph_encoder,           # GraphAttentionEncoder
        num_classes,
        seq_len=4,
        graph_emb_dim=256,
        temporal_hidden_dim=256,
        num_transformer_layers=2,
        num_transformer_heads=4,
        dropout=0.2,
        kan_hidden=128,
        kan_grid_size=5,
        kan_spline_order=3,
        use_cls_token=True,
    ):
        super().__init__()

        self.graph_encoder = graph_encoder
        self.seq_len = seq_len
        self.graph_emb_dim = graph_emb_dim
        self.use_cls_token = use_cls_token

        token_len = seq_len + 1 if use_cls_token else seq_len
        self.pos_embedding = nn.Parameter(torch.zeros(1, token_len, graph_emb_dim))

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, graph_emb_dim))
        else:
            self.cls_token = None

        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=graph_emb_dim,
            nhead=num_transformer_heads,
            dim_feedforward=temporal_hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        self.temporal_norm = nn.LayerNorm(graph_emb_dim)
        self.attn_pool = SequenceAttentionPooling(graph_emb_dim, dropout=dropout)

        # fusion của toàn chuỗi
        self.fusion = nn.Sequential(
            nn.LayerNorm(graph_emb_dim * 4),
            nn.Linear(graph_emb_dim * 4, graph_emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(graph_emb_dim),
        )

        self.head = KAN(
            layers_hidden=[graph_emb_dim, kan_hidden, num_classes],
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
            base_activation=torch.nn.SiLU,
        )

    def forward(self, batched_graph_list, update_grid=False, return_aux=False):
        seq_embs = []

        for batch_t in batched_graph_list:
            g = self.graph_encoder(batch_t)   # [B, D]
            seq_embs.append(g)

        x = torch.stack(seq_embs, dim=1)      # [B, L, D]

        if self.use_cls_token:
            B = x.size(0)
            cls = self.cls_token.expand(B, -1, -1)   # [B,1,D]
            x = torch.cat([cls, x], dim=1)           # [B,L+1,D]

        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.input_dropout(x)

        x = self.temporal_transformer(x)
        x = self.temporal_norm(x)

        if self.use_cls_token:
            cls_token = x[:, 0, :]
            tokens = x[:, 1:, :]
        else:
            cls_token = x.mean(dim=1)
            tokens = x

        attn_pool, attn_weights = self.attn_pool(tokens)
        mean_pool = tokens.mean(dim=1)
        last_token = tokens[:, -1, :]

        fused = torch.cat([cls_token, attn_pool, mean_pool, last_token], dim=-1)
        fused = self.fusion(fused)

        if fused.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                logits = self.head(fused.float(), update_grid=update_grid)
        else:
            logits = self.head(fused.float(), update_grid=update_grid)

        if return_aux:
            return logits, {
                "cls_token": cls_token,
                "attn_pool": attn_pool,
                "mean_pool": mean_pool,
                "last_token": last_token,
                "attn_weights": attn_weights,
                "fused": fused,
            }
        return logits

    def kan_regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return self.head.regularization_loss(
            regularize_activation=regularize_activation,
            regularize_entropy=regularize_entropy,
        )