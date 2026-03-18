import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch


def graph_dict_to_data(graph: Dict[str, Any]) -> Data:
    """
    Convert one saved graph dict into a PyG Data object.
    Important:
    - rename id_index -> id_token to avoid PyG automatic index increment on batching.
    """
    data = Data(
        x=graph["x"].float(),
        edge_index=graph["edge_index"].long(),
        edge_attr=graph["edge_attr"].float(),
        edge_type=graph["edge_type"].long(),
        id_token=graph["id_index"].long(),
        y=graph["y"].view(-1).long(),
    )

    # Optional node-wise fields if they exist
    if "arbitration_id" in graph:
        data.arbitration_id = graph["arbitration_id"].long()
    if "timestamp" in graph:
        data.timestamp = graph["timestamp"].float()
    if "msg_idx_in_file" in graph:
        data.msg_idx_in_file = graph["msg_idx_in_file"].long()

    # Optional graph metadata
    meta = graph.get("meta", {})
    data.attack_count = torch.tensor([int(meta.get("attack_count", 0))], dtype=torch.long)
    data.attack_ratio = torch.tensor([float(meta.get("attack_ratio", 0.0))], dtype=torch.float32)
    data.is_mixed_window = torch.tensor([int(bool(meta.get("is_mixed_window", False)))], dtype=torch.long)
    data.has_attack = torch.tensor([int(bool(meta.get("has_attack", False)))], dtype=torch.long)

    return data


class SequenceGraphDataset(Dataset):
    """
    Dataset for sequence-of-graphs.

    Expected saved sequence item format:
    {
        "sequence_id": str,
        "split": str,
        "graphs": [graph_dict_1, ..., graph_dict_L],
        "y": int,
        "graph_ids": [...],
        "graph_labels": [...],
        "meta": {...}
    }
    """

    def __init__(
        self,
        seq_path: str,
        max_samples: int = 0,
        return_meta: bool = True,
    ):
        super().__init__()
        self.seq_path = seq_path
        self.return_meta = return_meta

        self.samples: List[Dict[str, Any]] = torch.load(seq_path, map_location="cpu", weights_only=False)

        if max_samples > 0:
            self.samples = self.samples[:max_samples]

        if len(self.samples) == 0:
            raise ValueError("SequenceGraphDataset is empty: {}".format(seq_path))

        self.seq_len = len(self.samples[0]["graphs"])
        self.label_mapping = self._infer_label_mapping()

    def _infer_label_mapping(self) -> Dict[int, str]:
        mapping: Dict[int, str] = {}
        for s in self.samples:
            y = int(s["y"])
            label_name = None

            if "graph_labels" in s and len(s["graph_labels"]) > 0:
                label_name = str(s["graph_labels"][-1])
            elif "graphs" in s and len(s["graphs"]) > 0 and "window_label" in s["graphs"][-1]:
                label_name = str(s["graphs"][-1]["window_label"])

            if label_name is not None:
                mapping[y] = label_name

        return dict(sorted(mapping.items(), key=lambda kv: kv[0]))

    def get_class_counts(self, num_classes: Optional[int] = None) -> List[int]:
        if num_classes is None:
            if len(self.label_mapping) > 0:
                num_classes = max(self.label_mapping.keys()) + 1
            else:
                ys = [int(s["y"]) for s in self.samples]
                num_classes = max(ys) + 1

        counts = [0 for _ in range(num_classes)]
        for s in self.samples:
            y = int(s["y"])
            counts[y] += 1
        return counts

    def infer_num_ids(self) -> int:
        max_id = 0
        for s in self.samples:
            for g in s["graphs"]:
                if "id_index" in g:
                    cur = int(g["id_index"].max().item())
                    if cur > max_id:
                        max_id = cur
        return max_id + 1

    def infer_node_feat_dim(self) -> int:
        g0 = self.samples[0]["graphs"][0]
        return int(g0["x"].shape[1])

    def infer_edge_attr_dim(self) -> int:
        g0 = self.samples[0]["graphs"][0]
        return int(g0["edge_attr"].shape[1])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]

        graph_list = [graph_dict_to_data(g) for g in s["graphs"]]
        out = {
            "graphs": graph_list,
            "y": torch.tensor(int(s["y"]), dtype=torch.long),
        }

        if self.return_meta:
            out["sequence_id"] = s.get("sequence_id", str(idx))
            out["graph_ids"] = s.get("graph_ids", [])
            out["graph_labels"] = s.get("graph_labels", [])
            out["meta"] = s.get("meta", {})

        return out


def collate_sequence_graphs(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Batch a list of sequence samples.

    Returns:
    {
        "graph_batches": [Batch_t1, Batch_t2, ..., Batch_tL],
        "y": LongTensor[B],
        "sequence_ids": [...],
        "graph_ids": [...],
        "graph_labels": [...],
        "meta": [...],
    }
    """
    if len(batch) == 0:
        raise ValueError("Empty batch")

    seq_len = len(batch[0]["graphs"])
    for item in batch:
        if len(item["graphs"]) != seq_len:
            raise ValueError("All samples in a batch must have the same seq_len")

    graph_batches: List[Batch] = []
    for t in range(seq_len):
        graphs_t = [item["graphs"][t] for item in batch]
        batch_t = Batch.from_data_list(graphs_t)
        graph_batches.append(batch_t)

    y = torch.stack([item["y"] for item in batch], dim=0).view(-1)

    out = {
        "graph_batches": graph_batches,
        "y": y,
    }

    # Optional metadata
    if "sequence_id" in batch[0]:
        out["sequence_ids"] = [item["sequence_id"] for item in batch]
    if "graph_ids" in batch[0]:
        out["graph_ids"] = [item["graph_ids"] for item in batch]
    if "graph_labels" in batch[0]:
        out["graph_labels"] = [item["graph_labels"] for item in batch]
    if "meta" in batch[0]:
        out["meta"] = [item["meta"] for item in batch]

    return out


def move_sequence_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    graph_batches = []
    for bg in batch["graph_batches"]:
        bg = bg.to(device)
        bg.x = bg.x.float()
        bg.edge_attr = bg.edge_attr.float()
        bg.edge_index = bg.edge_index.long()
        bg.edge_type = bg.edge_type.long()
        if hasattr(bg, "id_token"):
            bg.id_token = bg.id_token.long()
        graph_batches.append(bg)

    y = batch["y"].to(device).long()

    out = dict(batch)
    out["graph_batches"] = graph_batches
    out["y"] = y
    return out