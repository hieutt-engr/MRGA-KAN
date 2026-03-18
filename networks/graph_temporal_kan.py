import torch
import torch.nn as nn

from networks.graph_attention_kan import GraphAttentionKAN, KAN


class GraphTemporalKAN(nn.Module):
    def __init__(
        self,
        graph_encoder: GraphAttentionKAN,
        num_classes: int,
        seq_len: int = 4,
        graph_emb_dim: int = 256,   # hidden_dim*2 if mean+max readout
        temporal_hidden_dim: int = 256,
        num_transformer_layers: int = 2,
        num_transformer_heads: int = 4,
        dropout: float = 0.2,
        kan_hidden: int = 128,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
    ):
        super().__init__()

        self.graph_encoder = graph_encoder
        self.seq_len = seq_len
        self.graph_emb_dim = graph_emb_dim

        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, graph_emb_dim))
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

        self.head = KAN(
            layers_hidden=[graph_emb_dim, kan_hidden, num_classes],
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
            base_activation=torch.nn.SiLU,
        )

    def forward(self, batched_graph_list, update_grid=False):
        """
        batched_graph_list: list length L
            each element is a PyG Batch for the graphs at timestep t
        """
        seq_embs = []

        for batch_t in batched_graph_list:
            # get graph embedding from graph encoder
            _, g = self.graph_encoder(batch_t, update_grid=False, return_graph_embedding=True)
            seq_embs.append(g)

        # [B, L, D]
        z = torch.stack(seq_embs, dim=1)

        # positional embedding
        z = z + self.pos_embedding[:, :z.size(1), :]
        z = self.input_dropout(z)

        z = self.temporal_transformer(z)
        z = self.temporal_norm(z)

        # use last token
        last_token = z[:, -1, :]

        # KAN head in fp32 is safer
        if last_token.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                logits = self.head(last_token.float(), update_grid=update_grid)
        else:
            logits = self.head(last_token.float(), update_grid=update_grid)

        return logits

    def kan_regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        reg = self.head.regularization_loss(
            regularize_activation=regularize_activation,
            regularize_entropy=regularize_entropy,
        )
        if hasattr(self.graph_encoder, "kan_regularization_loss"):
            reg = reg + self.graph_encoder.kan_regularization_loss(
                regularize_activation=regularize_activation,
                regularize_entropy=regularize_entropy,
            )
        return reg