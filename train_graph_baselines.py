#!/usr/bin/env python3
"""
train_graph_baselines.py

Train graph baseline models for the IVN dynamic graph pipeline.

This script is designed to train the baseline models defined in
`graph_baseline_models.py` on graph shards created by
`build_graphs_node_classification.py`.

Main purpose
------------
- Select one baseline model using --model_type.
- Reuse the same graph data format used by the proposed GraphAttentionKAN model.
- Save only the best checkpoint based on the selected validation metric.
- Save lightweight logs, history, confusion matrices, and final reports.

Expected folder structure
-------------------------
<graph_folder>/
    train/graphs_train_shard00000.pt ...
    val/graphs_val_shard00000.pt ...
    test/graphs_test_shard00000.pt ...
    graph_index_train.parquet/csv
    graph_index_val.parquet/csv
    graph_index_test.parquet/csv

Example commands
----------------
# GCN baseline
python train_graph_baselines.py \
  --graph_folder ./data/2017-subaru-forester/graphs_v1 \
  --save_folder ./save/baselines/gcn \
  --model_type gcn \
  --epochs 100 --batch_size 64 --device cuda --gpu_id 0

# GATv2 baseline
python train_graph_baselines.py \
  --graph_folder ./data/2017-subaru-forester/graphs_v1 \
  --save_folder ./save/baselines/gatv2 \
  --model_type gatv2 \
  --epochs 100 --batch_size 64 --device cuda --gpu_id 0

# Strong ablation baseline: multi-relation GAT without KAN
python train_graph_baselines.py \
  --graph_folder ./data/2017-subaru-forester/graphs_v1 \
  --save_folder ./save/baselines/mrgat_mlp \
  --model_type multi_relation_gat \
  --epochs 100 --batch_size 64 --device cuda --gpu_id 0

# Optional multitask node supervision
python train_graph_baselines.py \
  --graph_folder ./data/2017-subaru-forester/graphs_v1 \
  --save_folder ./save/baselines/mrgat_mlp_multitask \
  --model_type multi_relation_gat \
  --enable_node_task --node_loss_weight 0.5 \
  --selection_metric joint \
  --epochs 100 --batch_size 64 --device cuda --gpu_id 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Import baseline model factory.

from networks.graph_baseline_models import create_graph_baseline, SUPPORTED_BASELINES


# =============================================================================
# CLI
# =============================================================================


def parse_option() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train graph baseline models for IVN IDS")

    # Paths
    parser.add_argument("--graph_folder", type=str, required=True,
                        help="Folder containing graph shards and graph_index_*.parquet/csv")
    parser.add_argument("--save_folder", type=str, required=True,
                        help="Folder to save best checkpoint, logs, and reports")
    parser.add_argument("--model_name", type=str, default="",
                        help="Optional checkpoint name prefix. Default: baseline_<model_type>")

    # Baseline model selection
    parser.add_argument("--model_type", type=str, required=True, choices=SUPPORTED_BASELINES,
                        help=f"Baseline model to train. Supported: {', '.join(SUPPORTED_BASELINES)}")

    # Data loading
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--max_shards_train", type=int, default=0, help="0 means use all train shards")
    parser.add_argument("--max_shards_val", type=int, default=0, help="0 means use all val shards")
    parser.add_argument("--max_shards_test", type=int, default=0, help="0 means use all test shards")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda", help="cpu, cuda, or cuda:N")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")

    # Scheduler
    parser.add_argument("--scheduler", type=str, default="cosine_restart",
                        choices=["none", "cosine", "cosine_restart"])
    parser.add_argument("--t0", type=int, default=10, help="T_0 for CosineAnnealingWarmRestarts")
    parser.add_argument("--tmult", type=int, default=2, help="T_mult for CosineAnnealingWarmRestarts")
    parser.add_argument("--eta_min", type=float, default=1e-5)

    # Imbalance handling
    parser.add_argument("--use_weighted_sampler", action="store_true",
                        help="Use WeightedRandomSampler based on graph labels")
    parser.add_argument("--use_class_weights", action="store_true",
                        help="Use inverse-frequency class weights for graph loss")
    parser.add_argument("--use_node_class_weights", action="store_true",
                        help="Use inverse-frequency class weights for node loss")

    # Multi-task node classification
    parser.add_argument("--enable_node_task", action="store_true",
                        help="Enable node classification loss and node metrics")
    parser.add_argument("--node_loss_weight", type=float, default=1.0,
                        help="Weight for node classification loss")
    parser.add_argument("--node_target", type=str, default="node_y",
                        choices=["node_y", "node_is_attack"],
                        help="Node target used when --enable_node_task is set")
    parser.add_argument("--selection_metric", type=str, default="graph_macro_f1",
                        choices=["graph_macro_f1", "node_macro_f1", "joint"],
                        help="Validation metric used to save the best checkpoint")
    parser.add_argument("--node_head_from_layer", type=int, default=-2,
                        help="Node head layer index. -1=last block, -2=one block before last")

    # Loss
    parser.add_argument("--loss_name", type=str, default="ce", choices=["ce", "focal"],
                        help="Graph loss. Focal is implemented locally in this script")
    parser.add_argument("--node_loss_name", type=str, default="same_as_graph",
                        choices=["same_as_graph", "ce", "focal"],
                        help="Node loss. Use same_as_graph to mirror graph loss")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--node_label_smoothing", type=float, default=0.0)
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    # Model size
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--id_emb_dim", type=int, default=32)
    parser.add_argument("--rel_emb_dim", type=int, default=8)
    parser.add_argument("--num_relations", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--ffn_ratio", type=float, default=2.0)
    parser.add_argument("--graph_head_hidden", type=int, default=128)

    opt = parser.parse_args()

    if opt.model_name == "":
        opt.model_name = f"baseline_{opt.model_type}"

    if opt.selection_metric in {"node_macro_f1", "joint"} and not opt.enable_node_task:
        raise ValueError(
            f"selection_metric={opt.selection_metric} requires --enable_node_task. "
            "Use --selection_metric graph_macro_f1 for graph-only training."
        )

    os.makedirs(opt.save_folder, exist_ok=True)
    return opt


# =============================================================================
# Logging / reproducibility
# =============================================================================


def set_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("train_graph_baselines")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(opt: argparse.Namespace, logger: logging.Logger) -> str:
    requested = str(opt.device).lower().strip()
    if requested == "cpu" or not torch.cuda.is_available():
        logger.info("Using CPU")
        return "cpu"

    if requested == "cuda":
        device = f"cuda:{opt.gpu_id}"
    else:
        device = str(opt.device)

    gpu_index = int(device.split(":")[1]) if ":" in device else 0
    logger.info(f"Using GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")
    return device


# =============================================================================
# Table / graph loading helpers
# =============================================================================


def load_table(base: Path) -> pd.DataFrame:
    parquet_path = base.with_suffix(".parquet")
    csv_path = base.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Cannot find {parquet_path} or {csv_path}")


def infer_label_mapping(graph_folder: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for split in ["train", "val", "test"]:
        try:
            df = load_table(graph_folder / f"graph_index_{split}")
            if "y" in df.columns and "window_label" in df.columns:
                pairs = df[["y", "window_label"]].drop_duplicates()
                for _, row in pairs.iterrows():
                    mapping[int(row["y"])] = str(row["window_label"])
        except Exception:
            continue

    if not mapping:
        raise RuntimeError("Could not infer label mapping from graph_index files.")
    return dict(sorted(mapping.items(), key=lambda kv: kv[0]))


def get_class_names(label_mapping: Dict[int, str], num_classes: int) -> List[str]:
    return [str(label_mapping.get(i, f"class_{i}")) for i in range(num_classes)]


def graph_dict_to_data(graph: dict) -> Data:
    data = Data(
        x=graph["x"].float(),
        edge_index=graph["edge_index"].long(),
        edge_attr=graph["edge_attr"].float(),
        edge_type=graph["edge_type"].long(),
        y=graph["y"].view(-1).long(),
    )

    # The baseline models accept id_token. The graph builder stores id_index.
    if "id_index" in graph:
        data.id_token = graph["id_index"].long()
    elif "id_token" in graph:
        data.id_token = graph["id_token"].long()
    else:
        raise KeyError("Graph dict must contain id_index or id_token")

    # Optional node labels for multitask learning.
    if "node_y" in graph:
        data.node_y = graph["node_y"].long()
    if "node_mask" in graph:
        data.node_mask = graph["node_mask"].bool()
    if "node_is_attack" in graph:
        data.node_is_attack = graph["node_is_attack"].long()

    meta = graph.get("meta", {})
    data.attack_count = torch.tensor([int(meta.get("attack_count", 0))], dtype=torch.long)
    data.attack_ratio = torch.tensor([float(meta.get("attack_ratio", 0.0))], dtype=torch.float32)
    data.is_mixed_window = torch.tensor([int(bool(meta.get("is_mixed_window", False)))], dtype=torch.long)
    return data


def load_graph_split(
    graph_folder: Path,
    split_name: str,
    logger: logging.Logger,
    max_shards: int = 0,
) -> List[Data]:
    split_dir = graph_folder / split_name
    shard_paths = sorted(split_dir.glob(f"graphs_{split_name}_shard*.pt"))
    if len(shard_paths) == 0:
        raise FileNotFoundError(f"No shard files found in {split_dir}")

    if max_shards > 0:
        shard_paths = shard_paths[:max_shards]

    logger.info(f"[{split_name}] Loading {len(shard_paths)} shard(s) from {split_dir}")
    dataset: List[Data] = []
    for shard_path in shard_paths:
        shard_graphs = torch.load(shard_path, map_location="cpu", weights_only=False)
        for g in shard_graphs:
            dataset.append(graph_dict_to_data(g))

    logger.info(f"[{split_name}] Loaded graphs: {len(dataset)}")
    return dataset


# =============================================================================
# Losses / metrics
# =============================================================================


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weights."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits,
            target,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        loss = (1.0 - pt).pow(self.gamma) * ce
        return loss.mean()


def get_graph_class_counts(dataset: List[Data], num_classes: int) -> List[int]:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for data in dataset:
        y = int(data.y.view(-1)[0].item())
        counts[y] += 1
    return counts.tolist()


def get_node_class_counts(dataset: List[Data], num_classes: int, node_target: str = "node_y") -> List[int]:
    counts = torch.zeros(num_classes, dtype=torch.long)
    found = False
    for data in dataset:
        if not hasattr(data, node_target):
            continue
        found = True
        node_y = getattr(data, node_target).view(-1).long()
        if hasattr(data, "node_mask"):
            node_y = node_y[data.node_mask.view(-1).bool()]
        for c in node_y.tolist():
            if 0 <= int(c) < num_classes:
                counts[int(c)] += 1
    if not found:
        return [0 for _ in range(num_classes)]
    return counts.tolist()


def get_class_weights_tensor(cls_num_list: List[int], device: str) -> torch.Tensor:
    counts = np.array(cls_num_list, dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_weighted_sampler(dataset: List[Data], num_classes: int) -> WeightedRandomSampler:
    cls_counts = np.array(get_graph_class_counts(dataset, num_classes), dtype=np.float64)
    cls_counts = np.maximum(cls_counts, 1.0)
    cls_weights = 1.0 / cls_counts

    sample_weights = []
    for data in dataset:
        y = int(data.y.view(-1)[0].item())
        sample_weights.append(cls_weights[y])

    sample_weights_t = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(
        weights=sample_weights_t,
        num_samples=len(sample_weights_t),
        replacement=True,
    )


def build_criterion(
    loss_name: str,
    class_weights: Optional[torch.Tensor],
    label_smoothing: float,
    focal_gamma: float,
) -> nn.Module:
    loss_name = loss_name.lower()
    if loss_name == "ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if loss_name == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weights, label_smoothing=label_smoothing)
    raise ValueError(f"Unsupported loss_name: {loss_name}")


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"acc": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "balanced_acc": 0.0}
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


def selection_score(metrics: Dict[str, Dict[str, float]], mode: str) -> float:
    graph_f1 = float(metrics["graph"]["macro_f1"])
    node_f1 = float(metrics["node"]["macro_f1"])
    if mode == "graph_macro_f1":
        return graph_f1
    if mode == "node_macro_f1":
        return node_f1
    if mode == "joint":
        return 0.5 * (graph_f1 + node_f1)
    raise ValueError(f"Unknown selection metric: {mode}")


def parse_model_outputs(output):
    if isinstance(output, dict):
        return output.get("graph_logits", None), output.get("node_logits", None)
    if isinstance(output, (tuple, list)):
        if len(output) >= 2:
            return output[0], output[1]
        if len(output) == 1:
            return output[0], None
    return output, None


def forward_model(model: nn.Module, data: Data, enable_node_task: bool):
    try:
        output = model(data, return_node_logits=enable_node_task)
    except TypeError:
        output = model(data)
    graph_logits, node_logits = parse_model_outputs(output)
    if graph_logits is None:
        raise RuntimeError("Model forward did not return graph_logits.")
    return graph_logits, node_logits


def get_node_targets_and_mask(data: Data, node_target: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not hasattr(data, node_target):
        return None, None
    node_y = getattr(data, node_target).view(-1).long()
    if hasattr(data, "node_mask"):
        node_mask = data.node_mask.view(-1).bool()
    else:
        node_mask = torch.ones_like(node_y, dtype=torch.bool)
    return node_y, node_mask


def compute_loss(
    graph_logits: torch.Tensor,
    graph_y: torch.Tensor,
    node_logits: Optional[torch.Tensor],
    node_y: Optional[torch.Tensor],
    node_mask: Optional[torch.Tensor],
    criterion_graph: nn.Module,
    criterion_node: nn.Module,
    opt: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    graph_loss = criterion_graph(graph_logits.float(), graph_y)
    node_loss = torch.tensor(0.0, device=graph_logits.device)

    if opt.enable_node_task:
        if node_logits is None:
            raise RuntimeError("--enable_node_task is set, but the model returned node_logits=None.")
        if node_y is None or node_mask is None:
            raise RuntimeError(f"--enable_node_task is set, but batch has no {opt.node_target}/node_mask.")
        valid = node_mask.bool()
        if valid.any():
            node_loss = criterion_node(node_logits[valid].float(), node_y[valid])

    total_loss = graph_loss + opt.node_loss_weight * node_loss
    return total_loss, graph_loss, node_loss


# =============================================================================
# Train / evaluate
# =============================================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_graph: nn.Module,
    criterion_node: nn.Module,
    scaler: torch.amp.GradScaler,
    epoch: int,
    opt: argparse.Namespace,
    logger: logging.Logger,
) -> Dict[str, Dict[str, float]]:
    model.train()

    total_loss_sum = 0.0
    graph_loss_sum = 0.0
    node_loss_sum = 0.0
    graph_samples = 0
    node_samples = 0

    graph_true: List[int] = []
    graph_pred: List[int] = []
    node_true: List[int] = []
    node_pred: List[int] = []

    start_time = time.time()

    for batch_idx, data in enumerate(loader):
        data = data.to(opt.device)
        data.x = data.x.float()
        data.edge_index = data.edge_index.long()
        if hasattr(data, "edge_attr"):
            data.edge_attr = data.edge_attr.float()
        if hasattr(data, "edge_type"):
            data.edge_type = data.edge_type.long()
        if hasattr(data, "id_token"):
            data.id_token = data.id_token.long()

        y = data.y.view(-1).long()
        node_y, node_mask = get_node_targets_and_mask(data, opt.node_target)

        optimizer.zero_grad(set_to_none=True)

        if opt.amp and str(opt.device).startswith("cuda"):
            with torch.amp.autocast("cuda", enabled=True):
                graph_logits, node_logits = forward_model(model, data, opt.enable_node_task)
                loss, graph_loss, node_loss = compute_loss(
                    graph_logits, y, node_logits, node_y, node_mask,
                    criterion_graph, criterion_node, opt,
                )
            scaler.scale(loss).backward()
            if opt.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            graph_logits, node_logits = forward_model(model, data, opt.enable_node_task)
            loss, graph_loss, node_loss = compute_loss(
                graph_logits, y, node_logits, node_y, node_mask,
                criterion_graph, criterion_node, opt,
            )
            loss.backward()
            if opt.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()

        bs = int(y.numel())
        graph_samples += bs
        total_loss_sum += float(loss.item()) * bs
        graph_loss_sum += float(graph_loss.item()) * bs

        preds = graph_logits.argmax(dim=1)
        graph_true.extend(y.detach().cpu().numpy().tolist())
        graph_pred.extend(preds.detach().cpu().numpy().tolist())

        valid_node_count = 0
        if opt.enable_node_task and node_logits is not None and node_y is not None and node_mask is not None:
            valid = node_mask.bool()
            if valid.any():
                n_preds = node_logits[valid].argmax(dim=1)
                valid_node_count = int(valid.sum().item())
                node_samples += valid_node_count
                node_loss_sum += float(node_loss.item()) * valid_node_count
                node_true.extend(node_y[valid].detach().cpu().numpy().tolist())
                node_pred.extend(n_preds.detach().cpu().numpy().tolist())

        if (batch_idx + 1) % opt.print_freq == 0:
            gm = compute_metrics(graph_true, graph_pred)
            nm = compute_metrics(node_true, node_pred) if opt.enable_node_task else compute_metrics([], [])
            logger.info(
                f"Train Epoch [{epoch}] Batch [{batch_idx + 1}/{len(loader)}] | "
                f"loss={total_loss_sum / max(graph_samples, 1):.4f} | "
                f"g_loss={graph_loss_sum / max(graph_samples, 1):.4f} | "
                f"n_loss={node_loss_sum / max(node_samples, 1):.4f} | "
                f"G_acc={gm['acc'] * 100:.2f}% | G_mF1={gm['macro_f1']:.4f} | "
                f"N_acc={nm['acc'] * 100:.2f}% | N_mF1={nm['macro_f1']:.4f}"
            )

    graph_metrics = compute_metrics(graph_true, graph_pred)
    node_metrics = compute_metrics(node_true, node_pred) if opt.enable_node_task else compute_metrics([], [])

    return {
        "graph": {**graph_metrics, "loss": graph_loss_sum / max(graph_samples, 1)},
        "node": {**node_metrics, "loss": node_loss_sum / max(node_samples, 1) if node_samples > 0 else 0.0},
        "total_loss": total_loss_sum / max(graph_samples, 1),
        "epoch_time_sec": time.time() - start_time,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion_graph: nn.Module,
    criterion_node: nn.Module,
    opt: argparse.Namespace,
) -> Tuple[Dict[str, Dict[str, float]], List[int], List[int], List[int], List[int]]:
    model.eval()

    total_loss_sum = 0.0
    graph_loss_sum = 0.0
    node_loss_sum = 0.0
    graph_samples = 0
    node_samples = 0

    graph_true: List[int] = []
    graph_pred: List[int] = []
    node_true: List[int] = []
    node_pred: List[int] = []

    for data in loader:
        data = data.to(opt.device)
        data.x = data.x.float()
        data.edge_index = data.edge_index.long()
        if hasattr(data, "edge_attr"):
            data.edge_attr = data.edge_attr.float()
        if hasattr(data, "edge_type"):
            data.edge_type = data.edge_type.long()
        if hasattr(data, "id_token"):
            data.id_token = data.id_token.long()

        y = data.y.view(-1).long()
        node_y, node_mask = get_node_targets_and_mask(data, opt.node_target)

        graph_logits, node_logits = forward_model(model, data, opt.enable_node_task)
        loss, graph_loss, node_loss = compute_loss(
            graph_logits, y, node_logits, node_y, node_mask,
            criterion_graph, criterion_node, opt,
        )

        bs = int(y.numel())
        graph_samples += bs
        total_loss_sum += float(loss.item()) * bs
        graph_loss_sum += float(graph_loss.item()) * bs

        preds = graph_logits.argmax(dim=1)
        graph_true.extend(y.detach().cpu().numpy().tolist())
        graph_pred.extend(preds.detach().cpu().numpy().tolist())

        if opt.enable_node_task and node_logits is not None and node_y is not None and node_mask is not None:
            valid = node_mask.bool()
            if valid.any():
                n_preds = node_logits[valid].argmax(dim=1)
                valid_count = int(valid.sum().item())
                node_samples += valid_count
                node_loss_sum += float(node_loss.item()) * valid_count
                node_true.extend(node_y[valid].detach().cpu().numpy().tolist())
                node_pred.extend(n_preds.detach().cpu().numpy().tolist())

    graph_metrics = compute_metrics(graph_true, graph_pred)
    node_metrics = compute_metrics(node_true, node_pred) if opt.enable_node_task else compute_metrics([], [])

    metrics = {
        "graph": {**graph_metrics, "loss": graph_loss_sum / max(graph_samples, 1)},
        "node": {**node_metrics, "loss": node_loss_sum / max(node_samples, 1) if node_samples > 0 else 0.0},
        "total_loss": total_loss_sum / max(graph_samples, 1),
    }
    return metrics, graph_true, graph_pred, node_true, node_pred


# =============================================================================
# Saving helpers
# =============================================================================


def save_confusion_matrix(cm: np.ndarray, class_names: List[str], path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    df.to_csv(path_base.with_suffix(".csv"))
    with open(path_base.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write(df.to_string())


def confusion_matrix_to_string(
    cm: np.ndarray,
    class_names: List[str],
    index_label: str = r"true\pred",
) -> str:
    """Return a readable confusion matrix string for logger output."""
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    df.index.name = index_label
    return df.to_string()


def log_confusion_matrix(
    logger: logging.Logger,
    cm: np.ndarray,
    class_names: List[str],
    title: str,
) -> None:
    """Log a confusion matrix in table form."""
    logger.info("\n" + title + "\n" + confusion_matrix_to_string(cm, class_names))


def save_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def checkpoint_state(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    best_val_score: float,
    best_epoch_record: dict,
    label_mapping: Dict[int, str],
    node_label_mapping: Dict[int, str],
    opt: argparse.Namespace,
) -> dict:
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_val_score": best_val_score,
        "best_epoch_record": best_epoch_record,
        "label_mapping": label_mapping,
        "node_label_mapping": node_label_mapping,
        "args": vars(opt),
        "model_type": opt.model_type,
    }


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    opt = parse_option()
    logger = set_logger(os.path.join(opt.save_folder, "training_log.txt"))
    set_seed(opt.seed)
    opt.device = resolve_device(opt, logger)

    logger.info("=" * 90)
    logger.info("Train graph baseline model")
    logger.info(f"Options:\n{json.dumps(vars(opt), indent=2)}")
    logger.info("=" * 90)

    graph_folder = Path(opt.graph_folder)
    save_folder = Path(opt.save_folder)
    reports_dir = save_folder / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Per-epoch TEST confusion matrices are logged and saved here.
    # These files are not checkpoints; they are small evaluation artifacts.
    epoch_test_cm_dir = save_folder / "epoch_test_confusion_matrices"
    epoch_test_cm_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Labels and data
    # -------------------------------------------------------------------------
    label_mapping = infer_label_mapping(graph_folder)
    graph_class_names = get_class_names(label_mapping, len(label_mapping))
    num_graph_classes = len(graph_class_names)

    train_dataset = load_graph_split(graph_folder, "train", logger, opt.max_shards_train)
    val_dataset = load_graph_split(graph_folder, "val", logger, opt.max_shards_val)
    test_dataset = load_graph_split(graph_folder, "test", logger, opt.max_shards_test)

    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty.")

    sample = train_dataset[0]
    node_feat_dim = int(sample.x.size(1))
    edge_attr_dim = int(sample.edge_attr.size(1)) if hasattr(sample, "edge_attr") else 0
    num_ids = max(int(d.id_token.max().item()) for d in train_dataset + val_dataset + test_dataset) + 1

    if opt.enable_node_task:
        if opt.node_target == "node_is_attack":
            num_node_classes = 2
            node_label_mapping = {0: "normal", 1: "attack"}
        else:
            num_node_classes = num_graph_classes
            node_label_mapping = label_mapping.copy()
    else:
        num_node_classes = None
        node_label_mapping = {}

    node_class_names = get_class_names(node_label_mapping, num_node_classes or 0) if opt.enable_node_task else []

    logger.info(f"Graph/window label mapping: {label_mapping}")
    logger.info(f"node_feat_dim      = {node_feat_dim}")
    logger.info(f"edge_attr_dim      = {edge_attr_dim}")
    logger.info(f"num_ids            = {num_ids}")
    logger.info(f"num_graph_classes  = {num_graph_classes}")
    logger.info(f"num_node_classes   = {num_node_classes}")

    graph_cls_counts = get_graph_class_counts(train_dataset, num_graph_classes)
    logger.info(f"Train graph class distribution: {dict(zip(graph_class_names, graph_cls_counts))}")

    if opt.enable_node_task:
        node_cls_counts = get_node_class_counts(train_dataset, num_node_classes or 0, opt.node_target)
        logger.info(f"Train node class distribution: {dict(zip(node_class_names, node_cls_counts))}")
    else:
        node_cls_counts = []

    # -------------------------------------------------------------------------
    # DataLoaders
    # -------------------------------------------------------------------------
    train_sampler = None
    if opt.use_weighted_sampler:
        train_sampler = build_weighted_sampler(train_dataset, num_graph_classes)
        logger.info("Using WeightedRandomSampler based on graph labels")

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = create_graph_baseline(
        opt.model_type,
        node_feat_dim=node_feat_dim,
        edge_attr_dim=edge_attr_dim,
        num_classes=num_graph_classes,
        num_ids=num_ids,
        num_node_classes=num_node_classes,
        hidden_dim=opt.hidden_dim,
        num_layers=opt.num_layers,
        heads=opt.heads,
        id_emb_dim=opt.id_emb_dim,
        rel_emb_dim=opt.rel_emb_dim,
        num_relations=opt.num_relations,
        dropout=opt.dropout,
        ffn_ratio=opt.ffn_ratio,
        node_head_from_layer=opt.node_head_from_layer,
        graph_head_hidden=opt.graph_head_hidden,
    ).to(opt.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Initialized model_type={opt.model_type}")
    logger.info(f"Trainable parameters: {n_params:,}")

    # -------------------------------------------------------------------------
    # Optimizer, scheduler, criterion
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

    if opt.scheduler == "cosine_restart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=opt.t0, T_mult=opt.tmult, eta_min=opt.eta_min
        )
    elif opt.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.epochs, eta_min=opt.eta_min
        )
    else:
        scheduler = None

    graph_weights = get_class_weights_tensor(graph_cls_counts, opt.device) if opt.use_class_weights else None
    if graph_weights is not None:
        logger.info(f"Using graph class weights: {graph_weights.detach().cpu().tolist()}")

    criterion_graph = build_criterion(
        loss_name=opt.loss_name,
        class_weights=graph_weights,
        label_smoothing=opt.label_smoothing,
        focal_gamma=opt.focal_gamma,
    )

    if opt.enable_node_task:
        node_weights = get_class_weights_tensor(node_cls_counts, opt.device) if opt.use_node_class_weights else None
        if node_weights is not None:
            logger.info(f"Using node class weights: {node_weights.detach().cpu().tolist()}")
        node_loss_name = opt.loss_name if opt.node_loss_name == "same_as_graph" else opt.node_loss_name
        criterion_node = build_criterion(
            loss_name=node_loss_name,
            class_weights=node_weights,
            label_smoothing=opt.node_label_smoothing,
            focal_gamma=opt.focal_gamma,
        )
    else:
        criterion_node = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda", enabled=(opt.amp and str(opt.device).startswith("cuda")))

    # -------------------------------------------------------------------------
    # Training loop: only save best checkpoint
    # -------------------------------------------------------------------------
    best_val_score = -1.0
    best_epoch = 0
    best_epoch_record: Dict[str, object] = {}
    best_ckpt_path = save_folder / f"{opt.model_name}_best.pth"
    history: List[dict] = []

    for epoch in range(1, opt.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion_graph=criterion_graph,
            criterion_node=criterion_node,
            scaler=scaler,
            epoch=epoch,
            opt=opt,
            logger=logger,
        )

        val_metrics, y_val_true, y_val_pred, node_val_true, node_val_pred = evaluate(
            model, val_loader, criterion_graph, criterion_node, opt
        )
        test_metrics, y_test_true, y_test_pred, node_test_true, node_test_pred = evaluate(
            model, test_loader, criterion_graph, criterion_node, opt
        )

        if scheduler is not None:
            if opt.scheduler == "cosine_restart":
                scheduler.step(epoch - 1 + 1.0)
            else:
                scheduler.step()

        lr_now = float(optimizer.param_groups[0]["lr"])
        val_score = selection_score(val_metrics, opt.selection_metric)
        test_score = selection_score(test_metrics, opt.selection_metric)

        epoch_record = {
            "epoch": epoch,
            "lr": lr_now,
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "val_selection": val_score,
            "test_selection": test_score,
        }
        history.append(epoch_record)

        logger.info("\n" + "=" * 100)
        logger.info(f"Epoch {epoch}/{opt.epochs} | LR={lr_now:.8f}")
        logger.info(
            f"Train | total={train_metrics['total_loss']:.4f} | "
            f"G_loss={train_metrics['graph']['loss']:.4f} | G_acc={train_metrics['graph']['acc']*100:.2f}% | "
            f"G_mF1={train_metrics['graph']['macro_f1']:.4f} | "
            f"N_loss={train_metrics['node']['loss']:.4f} | N_acc={train_metrics['node']['acc']*100:.2f}% | "
            f"N_mF1={train_metrics['node']['macro_f1']:.4f}"
        )
        logger.info(
            f"Val   | total={val_metrics['total_loss']:.4f} | "
            f"G_loss={val_metrics['graph']['loss']:.4f} | G_acc={val_metrics['graph']['acc']*100:.2f}% | "
            f"G_mF1={val_metrics['graph']['macro_f1']:.4f} | "
            f"N_loss={val_metrics['node']['loss']:.4f} | N_acc={val_metrics['node']['acc']*100:.2f}% | "
            f"N_mF1={val_metrics['node']['macro_f1']:.4f} | selection={val_score:.4f}"
        )
        logger.info(
            f"Test  | total={test_metrics['total_loss']:.4f} | "
            f"G_loss={test_metrics['graph']['loss']:.4f} | G_acc={test_metrics['graph']['acc']*100:.2f}% | "
            f"G_mF1={test_metrics['graph']['macro_f1']:.4f} | "
            f"N_loss={test_metrics['node']['loss']:.4f} | N_acc={test_metrics['node']['acc']*100:.2f}% | "
            f"N_mF1={test_metrics['node']['macro_f1']:.4f} | selection={test_score:.4f}"
        )

        # ------------------------------------------------------------------
        # Log and save TEST confusion matrices after every epoch.
        # Graph CM is always produced. Node CM is produced when node task is enabled.
        # ------------------------------------------------------------------
        test_graph_cm_epoch = confusion_matrix(
            y_test_true,
            y_test_pred,
            labels=list(range(num_graph_classes)),
        )
        log_confusion_matrix(
            logger,
            test_graph_cm_epoch,
            graph_class_names,
            title=f"Epoch {epoch} TEST GRAPH confusion matrix",
        )
        save_confusion_matrix(
            test_graph_cm_epoch,
            graph_class_names,
            epoch_test_cm_dir / f"epoch_{epoch:03d}_test_graph_confusion_matrix",
        )

        if opt.enable_node_task and len(node_test_true) > 0:
            test_node_cm_epoch = confusion_matrix(
                node_test_true,
                node_test_pred,
                labels=list(range(num_node_classes or 0)),
            )
            log_confusion_matrix(
                logger,
                test_node_cm_epoch,
                node_class_names,
                title=f"Epoch {epoch} TEST NODE confusion matrix",
            )
            save_confusion_matrix(
                test_node_cm_epoch,
                node_class_names,
                epoch_test_cm_dir / f"epoch_{epoch:03d}_test_node_confusion_matrix",
            )

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_epoch_record = epoch_record

            state = checkpoint_state(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_val_score=best_val_score,
                best_epoch_record=best_epoch_record,
                label_mapping=label_mapping,
                node_label_mapping=node_label_mapping,
                opt=opt,
            )
            torch.save(state, best_ckpt_path)

            val_cm = confusion_matrix(y_val_true, y_val_pred, labels=list(range(num_graph_classes)))
            save_confusion_matrix(val_cm, graph_class_names, save_folder / "best_val_graph_confusion_matrix")

            logger.info(
                f"New best validation {opt.selection_metric}: {best_val_score:.4f} "
                f"at epoch {best_epoch}. Saved: {best_ckpt_path}"
            )

        # History is not a checkpoint. It is useful for plotting and comparison tables.
        save_json(history, save_folder / "history.json")

    # -------------------------------------------------------------------------
    # Final evaluation using best checkpoint
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 100)
    logger.info(f"Final evaluation using best checkpoint from epoch {best_epoch}: {best_ckpt_path}")
    logger.info("=" * 100)

    best_ckpt = torch.load(best_ckpt_path, map_location=opt.device, weights_only=False)
    model.load_state_dict(best_ckpt["model"])

    test_metrics, y_test_true, y_test_pred, node_test_true, node_test_pred = evaluate(
        model, test_loader, criterion_graph, criterion_node, opt
    )

    test_cm = confusion_matrix(y_test_true, y_test_pred, labels=list(range(num_graph_classes)))
    save_confusion_matrix(test_cm, graph_class_names, reports_dir / "final_test_graph_confusion_matrix")

    graph_report = classification_report(
        y_test_true,
        y_test_pred,
        labels=list(range(num_graph_classes)),
        target_names=graph_class_names,
        digits=4,
        zero_division=0,
    )
    with open(reports_dir / "final_test_graph_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(graph_report)

    logger.info(
        f"BEST TEST | Graph acc={test_metrics['graph']['acc']*100:.2f}% | "
        f"Graph macro-F1={test_metrics['graph']['macro_f1']:.4f} | "
        f"Graph weighted-F1={test_metrics['graph']['weighted_f1']:.4f} | "
        f"Graph balanced-acc={test_metrics['graph']['balanced_acc']:.4f}"
    )
    logger.info(f"\nFinal GRAPH classification report:\n{graph_report}")

    if opt.enable_node_task and len(node_test_true) > 0:
        node_cm = confusion_matrix(node_test_true, node_test_pred, labels=list(range(num_node_classes or 0)))
        save_confusion_matrix(node_cm, node_class_names, reports_dir / "final_test_node_confusion_matrix")
        node_report = classification_report(
            node_test_true,
            node_test_pred,
            labels=list(range(num_node_classes or 0)),
            target_names=node_class_names,
            digits=4,
            zero_division=0,
        )
        with open(reports_dir / "final_test_node_classification_report.txt", "w", encoding="utf-8") as f:
            f.write(node_report)
        logger.info(
            f"BEST TEST NODE | acc={test_metrics['node']['acc']*100:.2f}% | "
            f"macro-F1={test_metrics['node']['macro_f1']:.4f} | "
            f"weighted-F1={test_metrics['node']['weighted_f1']:.4f}"
        )
        logger.info(f"\nFinal NODE classification report:\n{node_report}")

    best_summary = {
        "model_type": opt.model_type,
        "model_name": opt.model_name,
        "best_checkpoint": str(best_ckpt_path),
        "best_epoch": best_epoch,
        "selection_metric": opt.selection_metric,
        "best_val_score": best_val_score,
        "final_test_metrics": test_metrics,
        "best_epoch_record": best_epoch_record,
    }
    save_json(best_summary, save_folder / "best_summary.json")

    logger.info("\nTraining finished.")
    logger.info(f"Best validation {opt.selection_metric}: {best_val_score:.4f} at epoch {best_epoch}")
    logger.info(f"Best checkpoint saved at: {best_ckpt_path}")
    logger.info(f"Reports saved in: {reports_dir}")


if __name__ == "__main__":
    main()
