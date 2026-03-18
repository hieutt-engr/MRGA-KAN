import torch
import torch.nn as nn

from networks.graph_attention_kan import GraphAttentionKAN, KAN


class GraphSequenceLastKAN(nn.Module):
    """
    Baseline for same clean split, no transformer.

    Input:
      batched_graph_list = [Batch_t1, ..., Batch_tL]

    Behavior:
      - ignore all previous graphs
      - only use the last graph in the sequence
      - graph_encoder -> graph embedding
      - KAN head for classification
    """

    def __init__(
        self,
        graph_encoder: GraphAttentionKAN,
        num_classes: int,
        graph_emb_dim: int = 256,
        kan_hidden: int = 128,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
    ):
        super().__init__()

        self.graph_encoder = graph_encoder
        self.graph_emb_dim = graph_emb_dim

        self.head = KAN(
            layers_hidden=[graph_emb_dim, kan_hidden, num_classes],
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
            base_activation=torch.nn.SiLU,
        )

    def forward(self, batched_graph_list, update_grid=False):
        """
        batched_graph_list: list length L
            each element is a PyG Batch for timestep t
        """
        last_batch = batched_graph_list[-1]

        # lấy graph embedding từ graph encoder
        _, g = self.graph_encoder(last_batch, update_grid=False, return_graph_embedding=True)

        # KAN head chạy fp32 để ổn định hơn
        if g.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                logits = self.head(g.float(), update_grid=update_grid)
        else:
            logits = self.head(g.float(), update_grid=update_grid)

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