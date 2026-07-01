import os
import json
import time
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)

import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import warnings
warnings.filterwarnings("ignore", message=r".*torch-scatter.*")

# ============================================================
# Optional external losses
# ============================================================
HAS_CUSTOM_LOSSES = True
try:
    from losses import FocalLoss, LDAMLoss, PolyFocalLoss
except Exception:
    HAS_CUSTOM_LOSSES = False
    FocalLoss = None
    LDAMLoss = None
    PolyFocalLoss = None

# ============================================================
# Your model
# ============================================================
from networks.graph_attention_ffn_kan_multitask_updated import GraphAttentionKAN
from utils import get_class_names, format_confusion_matrix_df, save_confusion_matrix_artifacts


# ============================================================
# CLI
# ============================================================

def parse_option():
    parser = argparse.ArgumentParser("Train multitask GraphAttentionKAN for IVN IDS")

    # Paths
    parser.add_argument("--graph_folder", type=str, required=True,
                        help="Folder containing graph shards and graph_index_*.parquet/csv")
    parser.add_argument("--save_folder", type=str, default="./save/graph_attention_ffn_kan_multitask",
                        help="Folder to save checkpoints and logs")
    parser.add_argument("--model_name", type=str, default="graph_attention_ffn_kan_multitask",
                        help="Base name for saved models")

    # Data loading
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--max_shards_train", type=int, default=0, help="0 means use all")
    parser.add_argument("--max_shards_val", type=int, default=0, help="0 means use all")
    parser.add_argument("--max_shards_test", type=int, default=0, help="0 means use all")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda",
                        help="Training device. Use 'cpu', 'cuda', or explicit 'cuda:N'.")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU index used when --device cuda is selected.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume")
    parser.add_argument("--print_val_node_cm_every", type=int, default=1,
                        help="Print val node confusion matrix every N epochs (0 to disable).")
    parser.add_argument("--save_val_node_cm", action="store_true",
                        help="Save val node confusion matrix per epoch to CSV/TXT.")


    # Scheduler
    parser.add_argument("--scheduler", type=str, default="cosine_restart",
                        choices=["none", "cosine", "cosine_restart"])
    parser.add_argument("--t0", type=int, default=10, help="T_0 for CosineAnnealingWarmRestarts")
    parser.add_argument("--tmult", type=int, default=2, help="T_mult for CosineAnnealingWarmRestarts")
    parser.add_argument("--eta_min", type=float, default=0.00001)

    # Sampling / imbalance
    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--use_node_class_weights", action="store_true")

    # Multi-task
    parser.add_argument("--enable_node_task", action="store_true",
                        help="Enable node classification loss/metrics")
    parser.add_argument("--node_loss_weight", type=float, default=1.0,
                        help="Weight for node classification loss")
    parser.add_argument("--selection_metric", type=str, default="joint",
                        choices=["joint", "graph_macro_f1", "node_macro_f1"],
                        help="Validation metric used to select best checkpoint")
    parser.add_argument("--node_target", type=str, default="node_y",
                        choices=["node_y", "node_is_attack"],
                        help="Which node label to use for node task")
    parser.add_argument("--node_head_from_layer", type=int, default=-2,
                        help="Which encoder block output to use for node head. Negative values are allowed, e.g. -1=last block, -2=one block before last")
    parser.add_argument("--save_epoch_checkpoints", action="store_true",
                        help="If set, save a checkpoint after each epoch under save_folder/epoch_save")
    parser.add_argument("--epoch_save_every", type=int, default=1,
                        help="Save one epoch checkpoint every N epochs when --save_epoch_checkpoints is enabled")

    # Loss (graph)
    parser.add_argument("--loss_name", type=str, default="ce",
                        choices=["ce", "focal", "ldam", "polyfocal"])
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--ldam_max_m", type=float, default=0.5)
    parser.add_argument("--ldam_s", type=float, default=30.0)

    # Optional separate node loss
    parser.add_argument("--node_loss_name", type=str, default="same_as_graph",
                        choices=["same_as_graph", "ce", "focal", "ldam", "polyfocal"])
    parser.add_argument("--node_label_smoothing", type=float, default=0.0)

    # KAN regularization
    parser.add_argument("--kan_reg_lambda", type=float, default=0.00001)
    parser.add_argument("--kan_reg_activation", type=float, default=1.0)
    parser.add_argument("--kan_reg_entropy", type=float, default=1.0)

    # Model
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--id_emb_dim", type=int, default=32)
    parser.add_argument("--rel_emb_dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--ffn_ratio", type=float, default=2.0)
    parser.add_argument("--num_relations", type=int, default=4)

    # Block KAN config (for FFN in MultiRelationGATBlock)
    parser.add_argument("--block_kan_grid_size", type=int, default=5)
    parser.add_argument("--block_kan_spline_order", type=int, default=3)
    parser.add_argument("--block_kan_scale_noise", type=float, default=0.1)
    parser.add_argument("--block_kan_scale_base", type=float, default=1.0)
    parser.add_argument("--block_kan_scale_spline", type=float, default=1.0)

    # KAN head config
    parser.add_argument("--kan_hidden", type=int, default=128)
    parser.add_argument("--kan_grid_size", type=int, default=5)
    parser.add_argument("--kan_spline_order", type=int, default=3)
    parser.add_argument("--kan_scale_noise", type=float, default=0.1)
    parser.add_argument("--kan_scale_base", type=float, default=1.0)
    parser.add_argument("--kan_scale_spline", type=float, default=1.0)

    opt = parser.parse_args()

    os.makedirs(opt.save_folder, exist_ok=True)
    if opt.save_epoch_checkpoints:
        os.makedirs(os.path.join(opt.save_folder, "epoch_save"), exist_ok=True)
    return opt


# ============================================================
# Logger
# ============================================================

def set_logger(log_path: str):
    logger = logging.getLogger("train_graph_attention_ffn_kan_multitask")
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


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cm_df_to_aligned_string(cm_df: pd.DataFrame, index_label: str = "true\pred", min_col_width: int = 12) -> str:
    """Render a confusion matrix DataFrame with evenly spaced columns for clean terminal logs."""
    values = cm_df.values
    value_width = max(len(str(int(v))) for v in values.flatten()) if values.size > 0 else 1
    label_width = max([len(str(c)) for c in cm_df.columns] + [len(str(i)) for i in cm_df.index] + [len(index_label)])
    cell_w = max(min_col_width, value_width + 2, label_width + 2)
    row_label_w = max(len(index_label), max(len(str(i)) for i in cm_df.index)) + 2

    header = " " * row_label_w + "".join(f"{str(col):^{cell_w}}" for col in cm_df.columns)
    lines = [header]
    for idx, row in cm_df.iterrows():
        row_str = f"{str(idx):<{row_label_w}}" + "".join(f"{int(v):^{cell_w}d}" for v in row.values)
        lines.append(row_str)
    return "\n".join(lines)


# ============================================================
# Helpers
# ============================================================

def load_table(base: Path) -> pd.DataFrame:
    pq = base.with_suffix(".parquet")
    csv = base.with_suffix(".csv")
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Cannot find {pq} or {csv}")


def infer_label_mapping(graph_folder: Path) -> Dict[int, str]:
    mapping = {}
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


def graph_dict_to_data(graph: dict) -> Data:
    data = Data(
        x=graph["x"].float(),
        edge_index=graph["edge_index"].long(),
        edge_attr=graph["edge_attr"].float(),
        edge_type=graph["edge_type"].long(),
        id_token=graph["id_index"].long(),
        y=graph["y"].view(-1).long(),
    )

    # New: node-level supervision for multitask
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


def load_graph_split(graph_folder: Path, split_name: str, logger, max_shards: int = 0) -> List[Data]:
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
        node_y = getattr(data, node_target)
        if hasattr(data, "node_mask"):
            node_y = node_y[data.node_mask]
        for c in node_y.view(-1).tolist():
            counts[int(c)] += 1
    if not found:
        return [0 for _ in range(num_classes)]
    return counts.tolist()


def build_weighted_sampler(dataset: List[Data], num_classes: int):
    cls_counts = np.array(get_graph_class_counts(dataset, num_classes), dtype=np.float64)
    cls_counts = np.maximum(cls_counts, 1.0)
    cls_weights = 1.0 / cls_counts

    sample_weights = []
    for data in dataset:
        y = int(data.y.view(-1)[0].item())
        sample_weights.append(cls_weights[y])

    sample_weights = torch.tensor(sample_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler


def get_class_weights_tensor(cls_num_list: List[int], device: str) -> torch.Tensor:
    counts = np.array(cls_num_list, dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_criterion(loss_name: str, label_smoothing: float, focal_gamma: float,
                    ldam_max_m: float, ldam_s: float,
                    cls_num_list: List[int], class_weights: Optional[torch.Tensor], logger,
                    prefix: str = ""):
    loss_name = loss_name.lower()

    if loss_name == "ce":
        logger.info(f"Using {prefix}CrossEntropyLoss")
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    if loss_name == "ldam":
        if not HAS_CUSTOM_LOSSES or LDAMLoss is None:
            logger.warning(f"{prefix}LDAMLoss not available, fallback to CrossEntropyLoss")
            return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        logger.info(f"Using {prefix}LDAMLoss")
        return LDAMLoss(cls_num_list=cls_num_list, max_m=ldam_max_m, s=ldam_s)

    if loss_name == "focal":
        if not HAS_CUSTOM_LOSSES or FocalLoss is None:
            logger.warning(f"{prefix}FocalLoss not available, fallback to CrossEntropyLoss")
            return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        logger.info(f"Using {prefix}FocalLoss")
        try:
            return FocalLoss(gamma=focal_gamma)
        except TypeError:
            return FocalLoss()

    if loss_name == "polyfocal":
        if not HAS_CUSTOM_LOSSES or PolyFocalLoss is None:
            logger.warning(f"{prefix}PolyFocalLoss not available, fallback to CrossEntropyLoss")
            return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        logger.info(f"Using {prefix}PolyFocalLoss")
        try:
            return PolyFocalLoss(gamma=focal_gamma, num_classes=10)
        except TypeError:
            return PolyFocalLoss()

    raise ValueError(f"Unsupported loss_name: {loss_name}")


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    if len(y_true) == 0:
        return {
            "acc": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "balanced_acc": 0.0,
        }
    return {
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
    }


def selection_score(metrics: Dict[str, Dict[str, float]], mode: str = "joint") -> float:
    graph_f1 = metrics["graph"]["macro_f1"]
    node_f1 = metrics["node"]["macro_f1"]
    if mode == "graph_macro_f1":
        return graph_f1
    if mode == "node_macro_f1":
        return node_f1
    return 0.5 * (graph_f1 + node_f1)


def save_checkpoint(state: dict, save_path: str):
    torch.save(state, save_path)


def parse_model_outputs(output):
    graph_logits = None
    node_logits = None

    if isinstance(output, dict):
        graph_logits = output.get("graph_logits", None)
        node_logits = output.get("node_logits", None)
    elif isinstance(output, (tuple, list)):
        if len(output) >= 2:
            graph_logits, node_logits = output[0], output[1]
        elif len(output) == 1:
            graph_logits = output[0]
    else:
        graph_logits = output

    return graph_logits, node_logits


def forward_multitask(model, data):
    try:
        output = model(data, return_node_logits=True)
    except TypeError:
        output = model(data)

    graph_logits, node_logits = parse_model_outputs(output)
    if graph_logits is None:
        raise RuntimeError(
            "Model did not return graph_logits. Update GraphAttentionKAN.forward() to return both graph_logits and node_logits "
            "(or at least graph_logits)."
        )
    return graph_logits, node_logits


def get_node_targets_and_mask(data: Data, node_target: str = "node_y") -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not hasattr(data, node_target):
        return None, None
    node_y = getattr(data, node_target).view(-1).long()
    if hasattr(data, "node_mask"):
        mask = data.node_mask.view(-1).bool()
    else:
        mask = torch.ones_like(node_y, dtype=torch.bool)
    return node_y, mask


def compute_multitask_loss(
    graph_logits: torch.Tensor,
    graph_y: torch.Tensor,
    node_logits: Optional[torch.Tensor],
    node_y: Optional[torch.Tensor],
    node_mask: Optional[torch.Tensor],
    criterion_graph,
    criterion_node,
    model,
    opt,
):
    graph_loss = criterion_graph(graph_logits.float(), graph_y)

    node_loss = torch.tensor(0.0, device=graph_logits.device)
    if opt.enable_node_task:
        if node_logits is None:
            raise RuntimeError(
                "Node task enabled but model returned node_logits=None. "
                "Update GraphAttentionKAN to produce node_logits for multitask training."
            )
        if node_y is None or node_mask is None:
            raise RuntimeError(
                f"Node task enabled but batch does not contain '{opt.node_target}' / node_mask. "
                "Rebuild graphs and update graph_dict_to_data()."
            )
        valid = node_mask.bool()
        if valid.any():
            node_loss = criterion_node(node_logits[valid].float(), node_y[valid])

    if hasattr(model, "kan_regularization_loss"):
        kan_reg = model.kan_regularization_loss(
            regularize_activation=opt.kan_reg_activation,
            regularize_entropy=opt.kan_reg_entropy,
        )
    else:
        kan_reg = torch.tensor(0.0, device=graph_logits.device)

    loss = graph_loss + opt.node_loss_weight * node_loss + opt.kan_reg_lambda * kan_reg
    return loss, graph_loss, node_loss, kan_reg


# ============================================================
# Training / Evaluation
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion_graph,
    criterion_node,
    scaler,
    epoch,
    opt,
    logger,
):
    model.train()

    total_loss = 0.0
    total_graph_loss = 0.0
    total_node_loss = 0.0
    total_kan_reg = 0.0
    total_graph_samples = 0
    total_node_samples = 0

    graph_all_preds, graph_all_labels = [], []
    node_all_preds, node_all_labels = [], []

    start_time = time.time()

    for idx, data in enumerate(loader):
        data = data.to(opt.device)
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        data.edge_index = data.edge_index.long()
        data.edge_type = data.edge_type.long()
        if hasattr(data, "id_token"):
            data.id_token = data.id_token.long()
        y = data.y.view(-1).long()
        node_y, node_mask = get_node_targets_and_mask(data, node_target=opt.node_target)

        optimizer.zero_grad(set_to_none=True)

        if opt.amp and opt.device.startswith("cuda"):
            with torch.amp.autocast("cuda", enabled=True):
                graph_logits, node_logits = forward_multitask(model, data)
                loss, graph_loss, node_loss, kan_reg = compute_multitask_loss(
                    graph_logits=graph_logits,
                    graph_y=y,
                    node_logits=node_logits,
                    node_y=node_y,
                    node_mask=node_mask,
                    criterion_graph=criterion_graph,
                    criterion_node=criterion_node,
                    model=model,
                    opt=opt,
                )

            scaler.scale(loss).backward()
            if opt.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            graph_logits, node_logits = forward_multitask(model, data)
            loss, graph_loss, node_loss, kan_reg = compute_multitask_loss(
                graph_logits=graph_logits,
                graph_y=y,
                node_logits=node_logits,
                node_y=node_y,
                node_mask=node_mask,
                criterion_graph=criterion_graph,
                criterion_node=criterion_node,
                model=model,
                opt=opt,
            )
            loss.backward()
            if opt.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.grad_clip)
            optimizer.step()

        # Graph metrics
        graph_preds = graph_logits.argmax(dim=1)
        bs_graph = y.size(0)
        total_graph_samples += bs_graph
        graph_all_preds.extend(graph_preds.detach().cpu().numpy().tolist())
        graph_all_labels.extend(y.detach().cpu().numpy().tolist())

        # Node metrics
        valid_nodes = 0
        if opt.enable_node_task and node_logits is not None and node_y is not None and node_mask is not None:
            valid = node_mask.bool()
            if valid.any():
                node_preds = node_logits[valid].argmax(dim=1)
                valid_nodes = int(valid.sum().item())
                total_node_samples += valid_nodes
                node_all_preds.extend(node_preds.detach().cpu().numpy().tolist())
                node_all_labels.extend(node_y[valid].detach().cpu().numpy().tolist())

        total_loss += float(loss.item()) * bs_graph
        total_graph_loss += float(graph_loss.item()) * bs_graph
        total_node_loss += float(node_loss.item()) * max(valid_nodes, 1)
        total_kan_reg += float(kan_reg.item()) * bs_graph

        if (idx + 1) % opt.print_freq == 0:
            graph_metrics = compute_metrics(graph_all_labels, graph_all_preds)
            node_metrics = compute_metrics(node_all_labels, node_all_preds) if opt.enable_node_task else {"acc": 0.0, "macro_f1": 0.0}
            logger.info(
                f"Train Epoch [{epoch}] Batch [{idx+1}/{len(loader)}] | "
                f"Loss {total_loss/max(total_graph_samples,1):.4f} | "
                f"Graph {total_graph_loss/max(total_graph_samples,1):.4f} | "
                f"Node {total_node_loss/max(total_node_samples,1):.4f} | "
                f"KAN {total_kan_reg/max(total_graph_samples,1):.4f} | "
                f"Graph Acc {graph_metrics['acc']*100:.2f}% | Graph Macro-F1 {graph_metrics['macro_f1']:.4f} | "
                f"Node Acc {node_metrics['acc']*100:.2f}% | Node Macro-F1 {node_metrics['macro_f1']:.4f}"
            )

    epoch_time = time.time() - start_time
    graph_metrics = compute_metrics(graph_all_labels, graph_all_preds)
    node_metrics = compute_metrics(node_all_labels, node_all_preds) if opt.enable_node_task else compute_metrics([], [])

    return {
        "graph": {
            **graph_metrics,
            "loss": total_graph_loss / max(1, total_graph_samples),
        },
        "node": {
            **node_metrics,
            "loss": total_node_loss / max(1, total_node_samples) if total_node_samples > 0 else 0.0,
        },
        "total_loss": total_loss / max(1, total_graph_samples),
        "kan_reg": total_kan_reg / max(1, total_graph_samples),
        "epoch_time_sec": epoch_time,
    }


@torch.no_grad()
def evaluate(model, loader, criterion_graph, criterion_node, opt):
    model.eval()

    total_loss = 0.0
    total_graph_loss = 0.0
    total_node_loss = 0.0
    total_kan_reg = 0.0
    total_graph_samples = 0
    total_node_samples = 0

    graph_all_preds, graph_all_labels = [], []
    node_all_preds, node_all_labels = [], []

    for data in loader:
        data = data.to(opt.device)
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        data.edge_index = data.edge_index.long()
        data.edge_type = data.edge_type.long()
        if hasattr(data, "id_token"):
            data.id_token = data.id_token.long()
        y = data.y.view(-1).long()
        node_y, node_mask = get_node_targets_and_mask(data, node_target=opt.node_target)

        graph_logits, node_logits = forward_multitask(model, data)
        loss, graph_loss, node_loss, kan_reg = compute_multitask_loss(
            graph_logits=graph_logits,
            graph_y=y,
            node_logits=node_logits,
            node_y=node_y,
            node_mask=node_mask,
            criterion_graph=criterion_graph,
            criterion_node=criterion_node,
            model=model,
            opt=opt,
        )

        graph_preds = graph_logits.argmax(dim=1)
        bs_graph = y.size(0)
        total_graph_samples += bs_graph
        graph_all_preds.extend(graph_preds.detach().cpu().numpy().tolist())
        graph_all_labels.extend(y.detach().cpu().numpy().tolist())

        valid_nodes = 0
        if opt.enable_node_task and node_logits is not None and node_y is not None and node_mask is not None:
            valid = node_mask.bool()
            if valid.any():
                node_preds = node_logits[valid].argmax(dim=1)
                valid_nodes = int(valid.sum().item())
                total_node_samples += valid_nodes
                node_all_preds.extend(node_preds.detach().cpu().numpy().tolist())
                node_all_labels.extend(node_y[valid].detach().cpu().numpy().tolist())

        total_loss += float(loss.item()) * bs_graph
        total_graph_loss += float(graph_loss.item()) * bs_graph
        total_node_loss += float(node_loss.item()) * max(valid_nodes, 1)
        total_kan_reg += float(kan_reg.item()) * bs_graph

    graph_metrics = compute_metrics(graph_all_labels, graph_all_preds)
    node_metrics = compute_metrics(node_all_labels, node_all_preds) if opt.enable_node_task else compute_metrics([], [])

    metrics = {
        "graph": {
            **graph_metrics,
            "loss": total_graph_loss / max(1, total_graph_samples),
        },
        "node": {
            **node_metrics,
            "loss": total_node_loss / max(1, total_node_samples) if total_node_samples > 0 else 0.0,
        },
        "total_loss": total_loss / max(1, total_graph_samples),
        "kan_reg": total_kan_reg / max(1, total_graph_samples),
    }

    return metrics, graph_all_labels, graph_all_preds, node_all_labels, node_all_preds


# ============================================================
# Main
# ============================================================

def main():
    opt = parse_option()

    log_file = os.path.join(opt.save_folder, "training_log.txt")
    logger = set_logger(log_file)
    epoch_save_dir = os.path.join(opt.save_folder, "epoch_save")
    val_node_cm_dir = os.path.join(opt.save_folder, "val_node_cm")
    final_reports_dir = os.path.join(opt.save_folder, "reports")
    if opt.save_val_node_cm:
        os.makedirs(val_node_cm_dir, exist_ok=True)
    os.makedirs(final_reports_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Start training multitask GraphAttentionKAN with KAN-FFN backbone")
    logger.info(f"Options:\n{json.dumps(vars(opt), indent=2)}")
    logger.info("=" * 80)

    set_seed(opt.seed)

    requested_device = str(opt.device).lower().strip()
    if requested_device == "cpu" or not torch.cuda.is_available():
        opt.device = "cpu"
        logger.info("Using CPU")
    else:
        if requested_device == "cuda":
            opt.device = f"cuda:{opt.gpu_id}"
        else:
            # explicit device string like cuda:2 takes precedence
            opt.device = str(opt.device)
        gpu_index = int(str(opt.device).split(":")[1]) if ":" in str(opt.device) else 0
        logger.info(f"Using GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")

    graph_folder = Path(opt.graph_folder)

    # ----------------------------
    # Load label names
    # ----------------------------
    label_mapping = infer_label_mapping(graph_folder)
    label_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
    num_graph_classes = len(label_names)

    logger.info(f"Graph/window label mapping: {label_mapping}")

    # ----------------------------
    # Load datasets
    # ----------------------------
    train_dataset = load_graph_split(graph_folder, "train", logger, max_shards=opt.max_shards_train)
    val_dataset = load_graph_split(graph_folder, "val", logger, max_shards=opt.max_shards_val)
    test_dataset = load_graph_split(graph_folder, "test", logger, max_shards=opt.max_shards_test)

    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty.")

    sample = train_dataset[0]
    node_feat_dim = sample.x.size(1)
    edge_attr_dim = sample.edge_attr.size(1)
    num_ids = max(int(d.id_token.max().item()) for d in train_dataset) + 1

    # Node classes
    if opt.node_target == "node_is_attack":
        num_node_classes = 2
        node_label_mapping = {0: "normal", 1: "attack"}
    else:
        num_node_classes = num_graph_classes
        node_label_mapping = label_mapping.copy()

    graph_class_names = get_class_names(label_mapping, num_graph_classes)
    node_class_names = get_class_names(node_label_mapping, num_node_classes)

    logger.info(f"node_feat_dim     = {node_feat_dim}")
    logger.info(f"edge_attr_dim     = {edge_attr_dim}")
    logger.info(f"num_ids           = {num_ids}")
    logger.info(f"num_graph_classes = {num_graph_classes}")
    logger.info(f"num_node_classes  = {num_node_classes}")

    graph_cls_num_list = get_graph_class_counts(train_dataset, num_graph_classes)
    node_cls_num_list = get_node_class_counts(train_dataset, num_node_classes, node_target=opt.node_target)
    logger.info(f"Train graph class distribution: {dict(zip(graph_class_names, graph_cls_num_list))}")
    logger.info(f"Train node class distribution : {dict(zip(node_class_names, node_cls_num_list))}")

    train_sampler = None
    if opt.use_weighted_sampler:
        train_sampler = build_weighted_sampler(train_dataset, num_graph_classes)
        logger.info("Using WeightedRandomSampler on graph labels")

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=(train_sampler is None),
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

    # ----------------------------
    # Model
    # ----------------------------
    logger.info("Initializing model...")

    # IMPORTANT: this script assumes the model was updated for multitask and
    # accepts `num_node_classes` and returns both graph_logits and node_logits.
    try:
        model = GraphAttentionKAN(
            node_feat_dim=node_feat_dim,
            edge_attr_dim=edge_attr_dim,
            num_classes=num_graph_classes,
            num_node_classes=num_node_classes,
            num_ids=num_ids,
            hidden_dim=opt.hidden_dim,
            num_layers=opt.num_layers,
            heads=opt.heads,
            id_emb_dim=opt.id_emb_dim,
            rel_emb_dim=opt.rel_emb_dim,
            num_relations=opt.num_relations,
            dropout=opt.dropout,
            ffn_ratio=opt.ffn_ratio,
            block_kan_grid_size=opt.block_kan_grid_size,
            block_kan_spline_order=opt.block_kan_spline_order,
            block_kan_scale_noise=opt.block_kan_scale_noise,
            block_kan_scale_base=opt.block_kan_scale_base,
            block_kan_scale_spline=opt.block_kan_scale_spline,
            kan_hidden=opt.kan_hidden,
            kan_grid_size=opt.kan_grid_size,
            kan_spline_order=opt.kan_spline_order,
            kan_scale_noise=opt.kan_scale_noise,
            kan_scale_base=opt.kan_scale_base,
            kan_scale_spline=opt.kan_scale_spline,
            node_head_from_layer=opt.node_head_from_layer,
        ).to(opt.device)
    except TypeError as e:
        raise RuntimeError(
            "GraphAttentionKAN is not yet updated for multitask. Please add a node head and let forward() return "
            "both graph_logits and node_logits. Also add __init__(..., num_node_classes=...)."
        ) from e

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

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

    graph_class_weights = None
    if opt.use_class_weights:
        graph_class_weights = get_class_weights_tensor(graph_cls_num_list, opt.device)
        logger.info(f"Using graph class weights: {graph_class_weights.detach().cpu().numpy().tolist()}")

    node_class_weights = None
    if opt.enable_node_task and opt.use_node_class_weights:
        node_class_weights = get_class_weights_tensor(node_cls_num_list, opt.device)
        logger.info(f"Using node class weights: {node_class_weights.detach().cpu().numpy().tolist()}")

    criterion_graph = build_criterion(
        loss_name=opt.loss_name,
        label_smoothing=opt.label_smoothing,
        focal_gamma=opt.focal_gamma,
        ldam_max_m=opt.ldam_max_m,
        ldam_s=opt.ldam_s,
        cls_num_list=graph_cls_num_list,
        class_weights=graph_class_weights,
        logger=logger,
        prefix="graph/",
    )

    node_loss_name = opt.loss_name if opt.node_loss_name == "same_as_graph" else opt.node_loss_name
    criterion_node = build_criterion(
        loss_name=node_loss_name,
        label_smoothing=opt.node_label_smoothing,
        focal_gamma=opt.focal_gamma,
        ldam_max_m=opt.ldam_max_m,
        ldam_s=opt.ldam_s,
        cls_num_list=node_cls_num_list,
        class_weights=node_class_weights,
        logger=logger,
        prefix="node/",
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(opt.amp and opt.device.startswith("cuda")))

    # Resume
    start_epoch = 1
    best_val_selection = -1.0
    best_test_selection = -1.0
    best_val_graph_macro_f1 = -1.0
    best_test_graph_macro_f1 = -1.0
    best_val_node_macro_f1 = -1.0
    best_test_node_macro_f1 = -1.0
    history = []

    if opt.resume:
        ckpt = torch.load(opt.resume, map_location=opt.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler", None) is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_selection = ckpt.get("best_val_selection", -1.0)
        best_test_selection = ckpt.get("best_test_selection", -1.0)
        best_val_graph_macro_f1 = ckpt.get("best_val_graph_macro_f1", -1.0)
        best_test_graph_macro_f1 = ckpt.get("best_test_graph_macro_f1", -1.0)
        best_val_node_macro_f1 = ckpt.get("best_val_node_macro_f1", -1.0)
        best_test_node_macro_f1 = ckpt.get("best_test_node_macro_f1", -1.0)
        history = ckpt.get("history", [])
        logger.info(f"Resumed from {opt.resume}, start_epoch={start_epoch}")

    # ----------------------------
    # Training loop
    # ----------------------------
    for epoch in range(start_epoch, opt.epochs + 1):
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

        lr_now = optimizer.param_groups[0]["lr"]
        val_select = selection_score(val_metrics, mode=opt.selection_metric)
        test_select = selection_score(test_metrics, mode=opt.selection_metric)

        epoch_record = {
            "epoch": epoch,
            "lr": lr_now,
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "val_selection": val_select,
            "test_selection": test_select,
        }
        history.append(epoch_record)

        logger.info("\n" + "=" * 100)
        logger.info(f"Epoch {epoch}/{opt.epochs} | LR={lr_now:.8f}")
        logger.info(
            f"Train | total={train_metrics['total_loss']:.4f} | graph={train_metrics['graph']['loss']:.4f} | "
            f"node={train_metrics['node']['loss']:.4f} | kan={train_metrics['kan_reg']:.4f} | "
            f"G-acc={train_metrics['graph']['acc']*100:.2f}% | G-mF1={train_metrics['graph']['macro_f1']:.4f} | "
            f"N-acc={train_metrics['node']['acc']*100:.2f}% | N-mF1={train_metrics['node']['macro_f1']:.4f}"
        )
        logger.info(
            f"Val   | total={val_metrics['total_loss']:.4f} | graph={val_metrics['graph']['loss']:.4f} | "
            f"node={val_metrics['node']['loss']:.4f} | kan={val_metrics['kan_reg']:.4f} | "
            f"G-acc={val_metrics['graph']['acc']*100:.2f}% | G-mF1={val_metrics['graph']['macro_f1']:.4f} | "
            f"N-acc={val_metrics['node']['acc']*100:.2f}% | N-mF1={val_metrics['node']['macro_f1']:.4f} | "
            f"selection={val_select:.4f}"
        )
        logger.info(
            f"Test  | total={test_metrics['total_loss']:.4f} | graph={test_metrics['graph']['loss']:.4f} | "
            f"node={test_metrics['node']['loss']:.4f} | kan={test_metrics['kan_reg']:.4f} | "
            f"G-acc={test_metrics['graph']['acc']*100:.2f}% | G-mF1={test_metrics['graph']['macro_f1']:.4f} | "
            f"N-acc={test_metrics['node']['acc']*100:.2f}% | N-mF1={test_metrics['node']['macro_f1']:.4f} | "
            f"selection={test_select:.4f}"
        )

        val_graph_cm = confusion_matrix(y_val_true, y_val_pred, labels=list(range(num_graph_classes)))
        val_graph_cm_df = format_confusion_matrix_df(val_graph_cm, graph_class_names)
        logger.info("Val Graph Confusion Matrix:\n%s", val_graph_cm_df.to_string())

        if opt.enable_node_task and len(node_val_true) > 0:
            val_node_cm = confusion_matrix(node_val_true, node_val_pred, labels=list(range(num_node_classes)))
            if opt.print_val_node_cm_every > 0 and (epoch % opt.print_val_node_cm_every == 0):
                val_node_cm_df = format_confusion_matrix_df(val_node_cm, node_class_names)
                logger.info("=" * 100)
                logger.info("VAL Node Confusion Matrix @ epoch %d", epoch)
                logger.info("\n%s", val_node_cm_df.to_string())
                logger.info("=" * 100)
            if opt.save_val_node_cm:
                save_confusion_matrix_artifacts(
                    val_node_cm,
                    node_class_names,
                    Path(val_node_cm_dir) / f"epoch_{epoch:03d}_val_node_cm"
                )

        common_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val_selection": best_val_selection,
            "best_test_selection": best_test_selection,
            "best_val_graph_macro_f1": best_val_graph_macro_f1,
            "best_test_graph_macro_f1": best_test_graph_macro_f1,
            "best_val_node_macro_f1": best_val_node_macro_f1,
            "best_test_node_macro_f1": best_test_node_macro_f1,
            "history": history,
            "label_mapping": label_mapping,
            "node_label_mapping": node_label_mapping,
            "args": vars(opt),
        }

        # save last
        last_ckpt_path = os.path.join(opt.save_folder, f"{opt.model_name}_last.pth")
        save_checkpoint(common_state, last_ckpt_path)

        # optional save every epoch
        if opt.save_epoch_checkpoints and (epoch % max(1, opt.epoch_save_every) == 0):
            epoch_ckpt_path = os.path.join(epoch_save_dir, f"{opt.model_name}_epoch_{epoch:03d}.pth")
            save_checkpoint(common_state, epoch_ckpt_path)

        # best by selection metric
        if val_select > best_val_selection:
            best_val_selection = val_select
            state = {
                **common_state,
                "best_val_selection": best_val_selection,
                "best_test_selection": best_test_selection,
                "best_val_graph_macro_f1": best_val_graph_macro_f1,
                "best_test_graph_macro_f1": best_test_graph_macro_f1,
                "best_val_node_macro_f1": best_val_node_macro_f1,
                "best_test_node_macro_f1": best_test_node_macro_f1,
            }
            best_path = os.path.join(opt.save_folder, f"{opt.model_name}_best_joint.pth")
            save_checkpoint(state, best_path)
            logger.info(f"🔥 New best VAL {opt.selection_metric}: {best_val_selection:.4f} | saved to {best_path}")

        # best by val graph macro-F1
        if val_metrics["graph"]["macro_f1"] > best_val_graph_macro_f1:
            best_val_graph_macro_f1 = val_metrics["graph"]["macro_f1"]
            state = {
                **common_state,
                "best_val_selection": best_val_selection,
                "best_test_selection": best_test_selection,
                "best_val_graph_macro_f1": best_val_graph_macro_f1,
                "best_test_graph_macro_f1": best_test_graph_macro_f1,
                "best_val_node_macro_f1": best_val_node_macro_f1,
                "best_test_node_macro_f1": best_test_node_macro_f1,
            }
            best_path = os.path.join(opt.save_folder, f"{opt.model_name}_best_val_graph.pth")
            save_checkpoint(state, best_path)
            np.save(os.path.join(opt.save_folder, "best_val_graph_confusion_matrix.npy"), val_graph_cm)
            save_confusion_matrix_artifacts(val_graph_cm, graph_class_names, Path(opt.save_folder) / "best_val_graph_confusion_matrix")
            logger.info(f"📈 New best VAL graph macro-F1: {best_val_graph_macro_f1:.4f} | saved to {best_path}")

        # best by test graph macro-F1 (tracking only)
        if test_metrics["graph"]["macro_f1"] > best_test_graph_macro_f1:
            best_test_graph_macro_f1 = test_metrics["graph"]["macro_f1"]
            state = {
                **common_state,
                "best_val_selection": best_val_selection,
                "best_test_selection": best_test_selection,
                "best_val_graph_macro_f1": best_val_graph_macro_f1,
                "best_test_graph_macro_f1": best_test_graph_macro_f1,
                "best_val_node_macro_f1": best_val_node_macro_f1,
                "best_test_node_macro_f1": best_test_node_macro_f1,
            }
            best_path = os.path.join(opt.save_folder, f"{opt.model_name}_best_test_graph.pth")
            save_checkpoint(state, best_path)
            logger.info(f"🧪 New best TEST graph macro-F1: {best_test_graph_macro_f1:.4f} | saved to {best_path}")

        # best by val node macro-F1
        if opt.enable_node_task and val_metrics["node"]["macro_f1"] > best_val_node_macro_f1:
            best_val_node_macro_f1 = val_metrics["node"]["macro_f1"]
            state = {
                **common_state,
                "best_val_selection": best_val_selection,
                "best_test_selection": best_test_selection,
                "best_val_graph_macro_f1": best_val_graph_macro_f1,
                "best_test_graph_macro_f1": best_test_graph_macro_f1,
                "best_val_node_macro_f1": best_val_node_macro_f1,
                "best_test_node_macro_f1": best_test_node_macro_f1,
            }
            best_path = os.path.join(opt.save_folder, f"{opt.model_name}_best_val_node.pth")
            save_checkpoint(state, best_path)
            if opt.enable_node_task and len(node_val_true) > 0:
                save_confusion_matrix_artifacts(
                    confusion_matrix(node_val_true, node_val_pred, labels=list(range(num_node_classes))),
                    node_class_names,
                    Path(opt.save_folder) / "best_val_node_confusion_matrix"
                )
            logger.info(f"📍 New best VAL node macro-F1: {best_val_node_macro_f1:.4f} | saved to {best_path}")

        # best by test node macro-F1 (tracking only)
        if opt.enable_node_task and test_metrics["node"]["macro_f1"] > best_test_node_macro_f1:
            best_test_node_macro_f1 = test_metrics["node"]["macro_f1"]
            state = {
                **common_state,
                "best_val_selection": best_val_selection,
                "best_test_selection": best_test_selection,
                "best_val_graph_macro_f1": best_val_graph_macro_f1,
                "best_test_graph_macro_f1": best_test_graph_macro_f1,
                "best_val_node_macro_f1": best_val_node_macro_f1,
                "best_test_node_macro_f1": best_test_node_macro_f1,
            }
            best_path = os.path.join(opt.save_folder, f"{opt.model_name}_best_test_node.pth")
            save_checkpoint(state, best_path)
            logger.info(f"🧪 New best TEST node macro-F1: {best_test_node_macro_f1:.4f} | saved to {best_path}")

        # best by test selection (tracking only)
        if test_select > best_test_selection:
            best_test_selection = test_select
            state = {
                **common_state,
                "best_val_selection": best_val_selection,
                "best_test_selection": best_test_selection,
                "best_val_graph_macro_f1": best_val_graph_macro_f1,
                "best_test_graph_macro_f1": best_test_graph_macro_f1,
                "best_val_node_macro_f1": best_val_node_macro_f1,
                "best_test_node_macro_f1": best_test_node_macro_f1,
            }
            best_test_path = os.path.join(opt.save_folder, f"{opt.model_name}_best_test_joint.pth")
            save_checkpoint(state, best_test_path)
            logger.info(f"🧪 New best TEST {opt.selection_metric}: {best_test_selection:.4f} | saved to {best_test_path}")

        with open(os.path.join(opt.save_folder, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    # ----------------------------
    # Final evaluation using best_joint checkpoint
    # ----------------------------
    logger.info("\n" + "=" * 100)
    logger.info("Final evaluation on TEST using best_joint checkpoint")
    logger.info("=" * 100)

    best_path = os.path.join(opt.save_folder, f"{opt.model_name}_best_joint.pth")
    best_ckpt = torch.load(best_path, map_location=opt.device, weights_only=False)
    model.load_state_dict(best_ckpt["model"])

    test_metrics, y_test_true, y_test_pred, node_test_true, node_test_pred = evaluate(
        model, test_loader, criterion_graph, criterion_node, opt
    )

    test_graph_cm = confusion_matrix(y_test_true, y_test_pred, labels=list(range(num_graph_classes)))
    logger.info(
        f"BEST TEST | total={test_metrics['total_loss']:.4f} | "
        f"Graph acc={test_metrics['graph']['acc']*100:.2f}% | Graph macro_f1={test_metrics['graph']['macro_f1']:.4f} | "
        f"Node acc={test_metrics['node']['acc']*100:.2f}% | Node macro_f1={test_metrics['node']['macro_f1']:.4f} | "
        f"selection={selection_score(test_metrics, opt.selection_metric):.4f}"
    )
    test_graph_cm_df = format_confusion_matrix_df(test_graph_cm, graph_class_names)
    logger.info("Final Test Graph Confusion Matrix:\n%s", test_graph_cm_df.to_string())

    graph_report = classification_report(
        y_test_true,
        y_test_pred,
        labels=list(range(num_graph_classes)),
        target_names=graph_class_names,
        digits=4,
        zero_division=0,
    )
    logger.info(f"\nFinal GRAPH Classification Report:\n{graph_report}")

    np.save(os.path.join(opt.save_folder, "final_graph_confusion_matrix.npy"), test_graph_cm)
    save_confusion_matrix_artifacts(test_graph_cm, graph_class_names, Path(final_reports_dir) / "final_graph_confusion_matrix")
    with open(os.path.join(opt.save_folder, "final_graph_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(graph_report)
    with open(os.path.join(final_reports_dir, "final_graph_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(graph_report)

    if opt.enable_node_task and len(node_test_true) > 0:
        test_node_cm = confusion_matrix(node_test_true, node_test_pred, labels=list(range(num_node_classes)))
        node_report = classification_report(
            node_test_true,
            node_test_pred,
            labels=list(range(num_node_classes)),
            target_names=node_class_names,
            digits=4,
            zero_division=0,
        )
        test_node_cm_df = format_confusion_matrix_df(test_node_cm, node_class_names)
        logger.info("Final Test Node Confusion Matrix:\n%s", test_node_cm_df.to_string())
        logger.info(f"\nFinal NODE Classification Report:\n{node_report}")
        np.save(os.path.join(opt.save_folder, "final_node_confusion_matrix.npy"), test_node_cm)
        save_confusion_matrix_artifacts(test_node_cm, node_class_names, Path(final_reports_dir) / "final_node_confusion_matrix")
        with open(os.path.join(opt.save_folder, "final_node_classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(node_report)
        with open(os.path.join(final_reports_dir, "final_node_classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(node_report)

    logger.info("\nTraining finished.")
    logger.info(f"Best val selection ({opt.selection_metric}): {best_val_selection:.4f}")
    logger.info(f"Best val graph macro-F1            : {best_val_graph_macro_f1:.4f}")
    logger.info(f"Best val node macro-F1             : {best_val_node_macro_f1:.4f}")
    logger.info(f"Best test selection ({opt.selection_metric}): {best_test_selection:.4f}")
    logger.info(f"Artifacts saved in: {opt.save_folder}")


if __name__ == "__main__":
    main()
