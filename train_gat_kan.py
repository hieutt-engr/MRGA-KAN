import os
import json
import time
import math
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
warnings.filterwarnings("ignore", message=".*torch-scatter.*")

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

from networks.graph_attention_kan import GraphAttentionKAN


# ============================================================
# CLI
# ============================================================

def parse_option():
    parser = argparse.ArgumentParser("Train GraphAttentionKAN for IVN IDS")

    # Paths
    parser.add_argument(
        "--graph_folder",
        type=str,
        required=True,
        help="Folder containing graph shards and graph_index_*.parquet/csv",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="./save/graph_attention_kan",
        help="Folder to save checkpoints and logs",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="graph_attention_kan",
        help="Base name for saved models",
    )

    # Data loading
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--max_shards_train", type=int, default=0, help="0 means use all")
    parser.add_argument("--max_shards_val", type=int, default=0, help="0 means use all")
    parser.add_argument("--max_shards_test", type=int, default=0, help="0 means use all")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume")

    # Scheduler
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine_restart",
        choices=["none", "cosine", "cosine_restart"],
    )
    parser.add_argument("--t0", type=int, default=10, help="T_0 for CosineAnnealingWarmRestarts")
    parser.add_argument("--tmult", type=int, default=2, help="T_mult for CosineAnnealingWarmRestarts")
    parser.add_argument("--eta_min", type=float, default=1e-5)

    # Sampling / imbalance
    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--use_class_weights", action="store_true")

    # Loss
    parser.add_argument(
        "--loss_name",
        type=str,
        default="ce",
        choices=["ce", "focal", "ldam", "polyfocal"],
    )
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--ldam_max_m", type=float, default=0.5)
    parser.add_argument("--ldam_s", type=float, default=30.0)

    # KAN regularization
    parser.add_argument("--kan_reg_lambda", type=float, default=1e-4)
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

    # KAN head config
    parser.add_argument("--kan_hidden", type=int, default=128)
    parser.add_argument("--kan_grid_size", type=int, default=5)
    parser.add_argument("--kan_spline_order", type=int, default=3)
    parser.add_argument("--kan_scale_noise", type=float, default=0.1)
    parser.add_argument("--kan_scale_base", type=float, default=1.0)
    parser.add_argument("--kan_scale_spline", type=float, default=1.0)

    opt = parser.parse_args()

    os.makedirs(opt.save_folder, exist_ok=True)
    return opt


# ============================================================
# Logger
# ============================================================

def set_logger(log_path: str):
    logger = logging.getLogger("train_graph_attention_kan")
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
        id_token=graph["id_index"].long(),   # đổi tên ở đây
        y=graph["y"].view(-1).long(),
    )

    # Optional useful metadata
    meta = graph.get("meta", {})
    data.attack_count = torch.tensor([int(meta.get("attack_count", 0))], dtype=torch.long)
    data.attack_ratio = torch.tensor([float(meta.get("attack_ratio", 0.0))], dtype=torch.float32)
    data.is_mixed_window = torch.tensor([int(bool(meta.get("is_mixed_window", False)))], dtype=torch.long)

    return data


def load_graph_split(
    graph_folder: Path,
    split_name: str,
    logger,
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


def get_class_counts(dataset: List[Data], num_classes: int) -> List[int]:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for data in dataset:
        y = int(data.y.view(-1)[0].item())
        counts[y] += 1
    return counts.tolist()


def build_weighted_sampler(dataset: List[Data], num_classes: int):
    cls_counts = np.array(get_class_counts(dataset, num_classes), dtype=np.float64)
    cls_counts = np.maximum(cls_counts, 1.0)
    cls_weights = 1.0 / cls_counts

    sample_weights = []
    for data in dataset:
        y = int(data.y.view(-1)[0].item())
        sample_weights.append(cls_weights[y])

    sample_weights = torch.tensor(sample_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def get_class_weights_tensor(cls_num_list: List[int], device: str) -> torch.Tensor:
    counts = np.array(cls_num_list, dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_criterion(opt, cls_num_list: List[int], class_weights: Optional[torch.Tensor], logger):
    loss_name = opt.loss_name.lower()

    if loss_name == "ce":
        logger.info("Using CrossEntropyLoss")
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=opt.label_smoothing)

    if loss_name == "ldam":
        if not HAS_CUSTOM_LOSSES or LDAMLoss is None:
            logger.warning("LDAMLoss not available in losses.py, fallback to CrossEntropyLoss")
            return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=opt.label_smoothing)
        logger.info("Using LDAMLoss")
        return LDAMLoss(cls_num_list=cls_num_list, max_m=opt.ldam_max_m, s=opt.ldam_s)

    if loss_name == "focal":
        if not HAS_CUSTOM_LOSSES or FocalLoss is None:
            logger.warning("FocalLoss not available in losses.py, fallback to CrossEntropyLoss")
            return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=opt.label_smoothing)
        logger.info("Using FocalLoss")
        try:
            return FocalLoss(gamma=opt.focal_gamma)
        except TypeError:
            return FocalLoss()

    if loss_name == "polyfocal":
        if not HAS_CUSTOM_LOSSES or PolyFocalLoss is None:
            logger.warning("PolyFocalLoss not available in losses.py, fallback to CrossEntropyLoss")
            return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=opt.label_smoothing)
        logger.info("Using PolyFocalLoss")
        try:
            return PolyFocalLoss(gamma=opt.focal_gamma)
        except TypeError:
            return PolyFocalLoss()

    raise ValueError(f"Unsupported loss_name: {opt.loss_name}")


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    return {
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
    }


def save_checkpoint(state: dict, save_path: str):
    torch.save(state, save_path)


# ============================================================
# Training / Evaluation
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    epoch,
    opt,
    logger,
    class_weights: Optional[torch.Tensor],
):
    model.train()

    total_loss = 0.0
    total_base_loss = 0.0
    total_kan_reg = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

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

        optimizer.zero_grad(set_to_none=True)

        if opt.amp and opt.device.startswith("cuda"):
            # with torch.cuda.amp.autocast():
            with torch.amp.autocast("cuda", enabled=(opt.amp and opt.device.startswith("cuda"))):
                logits = model(data)
                base_loss = criterion(logits, y)

                if hasattr(model, "kan_regularization_loss"):
                    kan_reg = model.kan_regularization_loss(
                        regularize_activation=opt.kan_reg_activation,
                        regularize_entropy=opt.kan_reg_entropy,
                    )
                else:
                    kan_reg = torch.tensor(0.0, device=logits.device)

                loss = base_loss + opt.kan_reg_lambda * kan_reg

            scaler.scale(loss).backward()
            if opt.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(data)
            base_loss = criterion(logits, y)

            if hasattr(model, "kan_regularization_loss"):
                kan_reg = model.kan_regularization_loss(
                    regularize_activation=opt.kan_reg_activation,
                    regularize_entropy=opt.kan_reg_entropy,
                )
            else:
                kan_reg = torch.tensor(0.0, device=logits.device)

            loss = base_loss + opt.kan_reg_lambda * kan_reg

            loss.backward()
            if opt.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.grad_clip)
            optimizer.step()

        preds = logits.argmax(dim=1)

        bs = y.size(0)
        total_samples += bs
        total_loss += float(loss.item()) * bs
        total_base_loss += float(base_loss.item()) * bs
        total_kan_reg += float(kan_reg.item()) * bs

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(y.detach().cpu().numpy().tolist())

        if (idx + 1) % opt.print_freq == 0:
            metrics = compute_metrics(all_labels, all_preds)
            logger.info(
                f"Train Epoch [{epoch}] Batch [{idx+1}/{len(loader)}] | "
                f"Loss {total_loss/total_samples:.4f} | "
                f"Base {total_base_loss/total_samples:.4f} | "
                f"KAN {total_kan_reg/total_samples:.4f} | "
                f"Acc {metrics['acc']*100:.2f}% | "
                f"Macro-F1 {metrics['macro_f1']:.4f}"
            )

    epoch_time = time.time() - start_time
    metrics = compute_metrics(all_labels, all_preds)
    metrics.update({
        "loss": total_loss / max(1, total_samples),
        "base_loss": total_base_loss / max(1, total_samples),
        "kan_reg": total_kan_reg / max(1, total_samples),
        "epoch_time_sec": epoch_time,
    })
    return metrics


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    opt,
):
    model.eval()

    total_loss = 0.0
    total_base_loss = 0.0
    total_kan_reg = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    for data in loader:
        data = data.to(opt.device)
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        data.edge_index = data.edge_index.long()
        data.edge_type = data.edge_type.long()
        if hasattr(data, "id_token"):
            data.id_token = data.id_token.long()
        y = data.y.view(-1).long()

        logits = model(data)
        # base_loss = criterion(logits, y)
        base_loss = criterion(logits.float(), y)
        if hasattr(model, "kan_regularization_loss"):
            kan_reg = model.kan_regularization_loss(
                regularize_activation=opt.kan_reg_activation,
                regularize_entropy=opt.kan_reg_entropy,
            )
        else:
            kan_reg = torch.tensor(0.0, device=logits.device)

        loss = base_loss + opt.kan_reg_lambda * kan_reg

        preds = logits.argmax(dim=1)

        bs = y.size(0)
        total_samples += bs
        total_loss += float(loss.item()) * bs
        total_base_loss += float(base_loss.item()) * bs
        total_kan_reg += float(kan_reg.item()) * bs

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(y.detach().cpu().numpy().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    metrics.update({
        "loss": total_loss / max(1, total_samples),
        "base_loss": total_base_loss / max(1, total_samples),
        "kan_reg": total_kan_reg / max(1, total_samples),
    })

    return metrics, all_labels, all_preds


# ============================================================
# Main
# ============================================================

def main():
    opt = parse_option()

    log_file = os.path.join(opt.save_folder, "training_log.txt")
    logger = set_logger(log_file)

    logger.info("=" * 80)
    logger.info("Start training GraphAttentionKAN")
    logger.info(f"Options:\n{json.dumps(vars(opt), indent=2)}")
    logger.info("=" * 80)

    set_seed(opt.seed)

    if torch.cuda.is_available() and opt.device != "cpu":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        opt.device = "cpu"
        logger.info("Using CPU")

    graph_folder = Path(opt.graph_folder)

    # ----------------------------
    # Load label names
    # ----------------------------
    label_mapping = infer_label_mapping(graph_folder)
    label_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
    num_classes = len(label_names)

    logger.info(f"Label mapping: {label_mapping}")

    # ----------------------------
    # Load datasets
    # ----------------------------
    train_dataset = load_graph_split(graph_folder, "train", logger, max_shards=opt.max_shards_train)
    val_dataset = load_graph_split(graph_folder, "val", logger, max_shards=opt.max_shards_val)
    test_dataset = load_graph_split(graph_folder, "test", logger, max_shards=opt.max_shards_test)

    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty.")

    # infer dimensions from first graph
    sample = train_dataset[0]
    node_feat_dim = sample.x.size(1)
    edge_attr_dim = sample.edge_attr.size(1)
    num_ids = max(int(d.id_token.max().item()) for d in train_dataset) + 1

    logger.info(f"node_feat_dim = {node_feat_dim}")
    logger.info(f"edge_attr_dim = {edge_attr_dim}")
    logger.info(f"num_ids       = {num_ids}")
    logger.info(f"num_classes   = {num_classes}")

    # class counts
    cls_num_list = get_class_counts(train_dataset, num_classes)
    logger.info(f"Train class distribution: {dict(zip(label_names, cls_num_list))}")

    # sampler
    train_sampler = None
    if opt.use_weighted_sampler:
        train_sampler = build_weighted_sampler(train_dataset, num_classes)
        logger.info("Using WeightedRandomSampler")

    # loaders
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

    model = GraphAttentionKAN(
        node_feat_dim=node_feat_dim,
        edge_attr_dim=edge_attr_dim,
        num_classes=num_classes,
        num_ids=num_ids,
        hidden_dim=opt.hidden_dim,
        num_layers=opt.num_layers,
        heads=opt.heads,
        id_emb_dim=opt.id_emb_dim,
        rel_emb_dim=opt.rel_emb_dim,
        num_relations=opt.num_relations,
        dropout=opt.dropout,
        ffn_ratio=opt.ffn_ratio,
        kan_hidden=opt.kan_hidden,
        kan_grid_size=opt.kan_grid_size,
        kan_spline_order=opt.kan_spline_order,
        kan_scale_noise=opt.kan_scale_noise,
        kan_scale_base=opt.kan_scale_base,
        kan_scale_spline=opt.kan_scale_spline,
    ).to(opt.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt.learning_rate,
        weight_decay=opt.weight_decay,
    )

    if opt.scheduler == "cosine_restart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=opt.t0,
            T_mult=opt.tmult,
            eta_min=opt.eta_min,
        )
    elif opt.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=opt.epochs,
            eta_min=opt.eta_min,
        )
    else:
        scheduler = None

    class_weights = None
    if opt.use_class_weights:
        class_weights = get_class_weights_tensor(cls_num_list, opt.device)
        logger.info(f"Using class weights: {class_weights.detach().cpu().numpy().tolist()}")

    criterion = build_criterion(opt, cls_num_list, class_weights, logger)
    scaler = torch.amp.GradScaler("cuda", enabled=(opt.amp and opt.device.startswith("cuda")))

    # Resume
    start_epoch = 1
    best_val_macro_f1 = -1.0
    best_val_acc = -1.0
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
        best_val_macro_f1 = ckpt.get("best_val_macro_f1", -1.0)
        best_val_acc = ckpt.get("best_val_acc", -1.0)
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
            criterion=criterion,
            scaler=scaler,
            epoch=epoch,
            opt=opt,
            logger=logger,
            class_weights=class_weights,
        )

        val_metrics, y_val_true, y_val_pred = evaluate(model, val_loader, criterion, opt)
        test_metrics, y_test_true, y_test_pred = evaluate(model, test_loader, criterion, opt)

        if scheduler is not None:
            if opt.scheduler == "cosine_restart":
                scheduler.step(epoch - 1 + 1.0)
            else:
                scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch,
            "lr": lr_now,
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }
        history.append(epoch_record)

        logger.info("\n" + "=" * 80)
        logger.info(f"Epoch {epoch}/{opt.epochs}")
        logger.info(f"LR: {lr_now:.8f}")
        logger.info(
            f"Train | loss={train_metrics['loss']:.4f} | base={train_metrics['base_loss']:.4f} | "
            f"kan={train_metrics['kan_reg']:.4f} | acc={train_metrics['acc']*100:.2f}% | "
            f"macro_f1={train_metrics['macro_f1']:.4f} | weighted_f1={train_metrics['weighted_f1']:.4f}"
        )
        logger.info(
            f"Val   | loss={val_metrics['loss']:.4f} | base={val_metrics['base_loss']:.4f} | "
            f"kan={val_metrics['kan_reg']:.4f} | acc={val_metrics['acc']*100:.2f}% | "
            f"macro_f1={val_metrics['macro_f1']:.4f} | weighted_f1={val_metrics['weighted_f1']:.4f} | "
            f"bal_acc={val_metrics['balanced_acc']:.4f}"
        )
        logger.info(
            f"Test  | loss={test_metrics['loss']:.4f} | base={test_metrics['base_loss']:.4f} | "
            f"kan={test_metrics['kan_reg']:.4f} | acc={test_metrics['acc']*100:.2f}% | "
            f"macro_f1={test_metrics['macro_f1']:.4f} | weighted_f1={test_metrics['weighted_f1']:.4f} | "
            f"bal_acc={test_metrics['balanced_acc']:.4f}"
        )

        # Val confusion matrix
        val_cm = confusion_matrix(y_val_true, y_val_pred, labels=list(range(num_classes)))
        logger.info(f"Val Confusion Matrix:\n{val_cm}")

        # save last
        last_ckpt_path = os.path.join(opt.save_folder, f"{opt.model_name}_last.pth")
        save_checkpoint(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_val_macro_f1": best_val_macro_f1,
                "best_val_acc": best_val_acc,
                "history": history,
                "label_mapping": label_mapping,
                "args": vars(opt),
            },
            last_ckpt_path,
        )

        # best by val macro_f1
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]

            best_path = os.path.join(opt.save_folder, f"{opt.model_name}_best_macro_f1.pth")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "best_val_macro_f1": best_val_macro_f1,
                    "best_val_acc": best_val_acc,
                    "history": history,
                    "label_mapping": label_mapping,
                    "args": vars(opt),
                },
                best_path,
            )

            np.save(os.path.join(opt.save_folder, "best_val_confusion_matrix.npy"), val_cm)
            logger.info(f"🔥 New best macro-F1 on val: {best_val_macro_f1:.4f} | saved to {best_path}")

        # best by val acc
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]

            best_acc_path = os.path.join(opt.save_folder, f"{opt.model_name}_best_acc.pth")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "best_val_macro_f1": best_val_macro_f1,
                    "best_val_acc": best_val_acc,
                    "history": history,
                    "label_mapping": label_mapping,
                    "args": vars(opt),
                },
                best_acc_path,
            )
            logger.info(f"✅ New best accuracy on val: {best_val_acc*100:.2f}%")

        # save history every epoch
        with open(os.path.join(opt.save_folder, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    # ----------------------------
    # Final evaluation using best_macro_f1 checkpoint
    # ----------------------------
    logger.info("\n" + "=" * 80)
    logger.info("Final evaluation on TEST using best_macro_f1 checkpoint")
    logger.info("=" * 80)

    best_path = os.path.join(opt.save_folder, f"{opt.model_name}_best_macro_f1.pth")
    best_ckpt = torch.load(best_path, map_location=opt.device, weights_only=False)
    model.load_state_dict(best_ckpt["model"])

    test_metrics, y_test_true, y_test_pred = evaluate(model, test_loader, criterion, opt)
    test_cm = confusion_matrix(y_test_true, y_test_pred, labels=list(range(num_classes)))

    logger.info(
        f"BEST TEST | loss={test_metrics['loss']:.4f} | "
        f"acc={test_metrics['acc']*100:.2f}% | "
        f"macro_f1={test_metrics['macro_f1']:.4f} | "
        f"weighted_f1={test_metrics['weighted_f1']:.4f} | "
        f"balanced_acc={test_metrics['balanced_acc']:.4f}"
    )
    logger.info(f"Final Test Confusion Matrix:\n{test_cm}")

    report = classification_report(
        y_test_true,
        y_test_pred,
        labels=list(range(num_classes)),
        target_names=label_names,
        digits=4,
        zero_division=0,
    )
    logger.info(f"\nFinal Classification Report:\n{report}")

    np.save(os.path.join(opt.save_folder, "final_confusion_matrix.npy"), test_cm)
    with open(os.path.join(opt.save_folder, "final_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"\nTraining finished.")
    logger.info(f"Best val macro-F1: {best_val_macro_f1:.4f}")
    logger.info(f"Best val acc     : {best_val_acc*100:.2f}%")
    logger.info(f"Artifacts saved in: {opt.save_folder}")


if __name__ == "__main__":
    main()