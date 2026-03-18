import os
import json
import time
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)

import warnings
warnings.filterwarnings("ignore", message=".*torch-scatter.*")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

# Optional custom losses
HAS_CUSTOM_LOSSES = True
try:
    from losses import FocalLoss, LDAMLoss, PolyFocalLoss
except Exception:
    HAS_CUSTOM_LOSSES = False
    FocalLoss = None
    LDAMLoss = None
    PolyFocalLoss = None

from dataset_sequence_graph import (
    SequenceGraphDataset,
    collate_sequence_graphs,
    move_sequence_batch_to_device,
)
from networks.graph_attention_kan import GraphAttentionKAN
from networks.graph_temporal_kan import GraphTemporalKAN
from networks.graph_sequence_last_kan import GraphSequenceLastKAN

# ============================================================
# CLI
# ============================================================

def parse_option():
    parser = argparse.ArgumentParser("Train GraphTemporalKAN for IVN IDS")

    # Paths
    parser.add_argument("--sequence_folder", type=str, required=True,
                        help="Folder containing train_seq.pt / val_seq.pt / test_seq.pt")
    parser.add_argument("--save_folder", type=str, default="./save/graph_temporal_kan",
                        help="Folder to save logs and checkpoints")
    parser.add_argument("--model_name", type=str, default="graph_temporal_kan")

    # Optional pretrained graph encoder
    parser.add_argument("--graph_encoder_ckpt", type=str, default="",
                        help="Path to pretrained single-window GraphAttentionKAN checkpoint")
    parser.add_argument("--strict_graph_encoder_load", action="store_true")
    parser.add_argument("--freeze_graph_encoder_epochs", type=int, default=0)

    # Data loading
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--max_train_sequences", type=int, default=0)
    parser.add_argument("--max_val_sequences", type=int, default=0)
    parser.add_argument("--max_test_sequences", type=int, default=0)

    # Stratified subsampling at sequence level
    parser.add_argument("--subsample_train_frac", type=float, default=1.0)
    parser.add_argument("--subsample_val_frac", type=float, default=1.0)
    parser.add_argument("--subsample_test_frac", type=float, default=1.0)
    parser.add_argument("--subsample_min_per_class", type=int, default=0)
    parser.add_argument("--subsample_seed", type=int, default=42)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=str, default="")

    # Scheduler
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["none", "cosine", "cosine_restart"])
    parser.add_argument("--t0", type=int, default=10)
    parser.add_argument("--tmult", type=int, default=2)
    parser.add_argument("--eta_min", type=float, default=1e-6)

    # Imbalance handling
    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--use_class_weights", action="store_true")

    # Loss
    parser.add_argument("--loss_name", type=str, default="ce",
                        choices=["ce", "focal", "ldam", "polyfocal"])
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--ldam_max_m", type=float, default=0.5)
    parser.add_argument("--ldam_s", type=float, default=30.0)

    # Graph encoder config
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--id_emb_dim", type=int, default=32)
    parser.add_argument("--rel_emb_dim", type=int, default=8)
    parser.add_argument("--num_relations", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--ffn_ratio", type=float, default=2.0)

    # Temporal transformer
    parser.add_argument("--temporal_hidden_dim", type=int, default=256)
    parser.add_argument("--num_transformer_layers", type=int, default=2)
    parser.add_argument("--num_transformer_heads", type=int, default=4)
    parser.add_argument(
        "--sequence_model_mode",
        type=str,
        default="transformer",
        choices=["transformer", "last_only"],
        help="transformer = GraphTemporalKAN, last_only = use only the last graph in the sequence",
    )

    # KAN head
    parser.add_argument("--kan_hidden", type=int, default=128)
    parser.add_argument("--kan_grid_size", type=int, default=5)
    parser.add_argument("--kan_spline_order", type=int, default=3)
    parser.add_argument("--kan_scale_noise", type=float, default=0.1)
    parser.add_argument("--kan_scale_base", type=float, default=1.0)
    parser.add_argument("--kan_scale_spline", type=float, default=1.0)

    # KAN regularization
    parser.add_argument("--kan_reg_lambda", type=float, default=1e-4)
    parser.add_argument("--kan_reg_activation", type=float, default=1.0)
    parser.add_argument("--kan_reg_entropy", type=float, default=1.0)

    opt = parser.parse_args()
    os.makedirs(opt.save_folder, exist_ok=True)
    return opt


# ============================================================
# Logger / seed
# ============================================================

def set_logger(log_path):
    logger = logging.getLogger("train_graph_temporal_kan")
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset helpers
# ============================================================

def get_base_dataset(dataset):
    if isinstance(dataset, Subset):
        return dataset.dataset
    return dataset


def get_labels_from_any_dataset(dataset):
    if isinstance(dataset, Subset):
        base = dataset.dataset
        labels = [int(base.samples[idx]["y"]) for idx in dataset.indices]
        return labels
    else:
        return [int(s["y"]) for s in dataset.samples]


def get_class_counts_from_any_dataset(dataset, num_classes):
    labels = get_labels_from_any_dataset(dataset)
    counts = [0 for _ in range(num_classes)]
    for y in labels:
        counts[y] += 1
    return counts


def infer_num_ids_from_any_dataset(dataset):
    base = get_base_dataset(dataset)
    return base.infer_num_ids()


def infer_node_feat_dim_from_any_dataset(dataset):
    base = get_base_dataset(dataset)
    return base.infer_node_feat_dim()


def infer_edge_attr_dim_from_any_dataset(dataset):
    base = get_base_dataset(dataset)
    return base.infer_edge_attr_dim()


def infer_seq_len_from_any_dataset(dataset):
    base = get_base_dataset(dataset)
    return base.seq_len


def infer_label_mapping_from_any_dataset(dataset):
    base = get_base_dataset(dataset)
    return base.label_mapping


def stratified_subsample_sequence_dataset(dataset, frac=1.0, min_per_class=0, seed=42):
    """
    Stratified subsampling by sequence label y.
    Returns:
      subset_dataset, selected_indices, class_counts_dict
    """
    if frac >= 1.0:
        labels = get_labels_from_any_dataset(dataset)
        counts = dict(sorted(dict((y, labels.count(y)) for y in set(labels)).items()))
        return dataset, list(range(len(dataset))), counts

    rng = np.random.default_rng(seed)
    base = get_base_dataset(dataset)

    if isinstance(dataset, Subset):
        candidate_indices = dataset.indices
    else:
        candidate_indices = list(range(len(base.samples)))

    label_to_indices = defaultdict(list)
    for idx in candidate_indices:
        y = int(base.samples[idx]["y"])
        label_to_indices[y].append(idx)

    selected = []
    class_counts = {}

    for y, idxs in sorted(label_to_indices.items(), key=lambda kv: kv[0]):
        n = len(idxs)
        k = int(round(n * frac))
        k = max(min_per_class, k)
        k = min(k, n)

        chosen = rng.choice(idxs, size=k, replace=False).tolist()
        selected.extend(chosen)
        class_counts[y] = k

    selected = sorted(selected)
    subset = Subset(base, selected)
    return subset, selected, class_counts


# ============================================================
# Metrics / weights / loss
# ============================================================

def compute_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
    }


def get_class_weights_tensor(cls_num_list, device):
    counts = np.array(cls_num_list, dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_weighted_sampler(dataset, num_classes):
    labels = get_labels_from_any_dataset(dataset)

    cls_counts = np.zeros(num_classes, dtype=np.float64)
    for y in labels:
        cls_counts[y] += 1
    cls_counts = np.maximum(cls_counts, 1.0)

    cls_weights = 1.0 / cls_counts
    sample_weights = torch.tensor([cls_weights[y] for y in labels], dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def build_criterion(opt, cls_num_list, class_weights, logger):
    loss_name = opt.loss_name.lower()

    if loss_name == "ce":
        logger.info("Using CrossEntropyLoss")
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=opt.label_smoothing)

    if loss_name == "ldam":
        if not HAS_CUSTOM_LOSSES or LDAMLoss is None:
            logger.warning("LDAMLoss not available, fallback to CrossEntropyLoss")
            return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=opt.label_smoothing)
        logger.info("Using LDAMLoss")
        return LDAMLoss(cls_num_list=cls_num_list, max_m=opt.ldam_max_m, s=opt.ldam_s)

    if loss_name == "focal":
        if not HAS_CUSTOM_LOSSES or FocalLoss is None:
            logger.warning("FocalLoss not available, fallback to CrossEntropyLoss")
            return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=opt.label_smoothing)
        logger.info("Using FocalLoss")
        try:
            return FocalLoss(gamma=opt.focal_gamma)
        except TypeError:
            return FocalLoss()

    if loss_name == "polyfocal":
        if not HAS_CUSTOM_LOSSES or PolyFocalLoss is None:
            logger.warning("PolyFocalLoss not available, fallback to CrossEntropyLoss")
            return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=opt.label_smoothing)
        logger.info("Using PolyFocalLoss")
        try:
            return PolyFocalLoss(gamma=opt.focal_gamma)
        except TypeError:
            return PolyFocalLoss()

    raise ValueError("Unsupported loss_name: {}".format(opt.loss_name))


def save_checkpoint(state, save_path):
    torch.save(state, save_path)


def set_requires_grad(module, requires_grad):
    for p in module.parameters():
        p.requires_grad = requires_grad


def load_graph_encoder_checkpoint(model, ckpt_path, logger, strict=False):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)

    try:
        model.graph_encoder.load_state_dict(state, strict=strict)
        logger.info("Loaded pretrained graph encoder directly from checkpoint.")
        return
    except Exception:
        pass

    ge_state = {}
    for k, v in state.items():
        if k.startswith("graph_encoder."):
            ge_state[k.replace("graph_encoder.", "", 1)] = v

    if len(ge_state) > 0:
        model.graph_encoder.load_state_dict(ge_state, strict=strict)
        logger.info("Loaded pretrained graph encoder from graph_encoder.* checkpoint keys.")
        return

    raise RuntimeError("Could not load graph encoder weights from {}".format(ckpt_path))


# ============================================================
# Train / Eval
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
):
    model.train()

    total_loss = 0.0
    total_base_loss = 0.0
    total_kan_reg = 0.0
    total_samples = 0

    all_preds = []
    all_labels = []

    num_skipped = 0
    num_valid_batches = 0
    start_time = time.time()

    for idx, batch in enumerate(loader):
        batch = move_sequence_batch_to_device(batch, opt.device)
        graph_batches = batch["graph_batches"]
        y = batch["y"]

        optimizer.zero_grad(set_to_none=True)

        if opt.amp and opt.device.startswith("cuda"):
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(graph_batches, update_grid=False)

            if not torch.isfinite(logits).all():
                logger.warning("[Epoch {} Batch {}] Non-finite logits. Skip.".format(epoch, idx + 1))
                num_skipped += 1
                continue

            base_loss = criterion(logits.float(), y)
            if not torch.isfinite(base_loss):
                logger.warning("[Epoch {} Batch {}] Non-finite base loss. Skip.".format(epoch, idx + 1))
                num_skipped += 1
                continue

            if hasattr(model, "kan_regularization_loss"):
                kan_reg = model.kan_regularization_loss(
                    regularize_activation=opt.kan_reg_activation,
                    regularize_entropy=opt.kan_reg_entropy,
                )
            else:
                kan_reg = torch.tensor(0.0, device=logits.device)

            loss = base_loss + opt.kan_reg_lambda * kan_reg
            if not torch.isfinite(loss):
                logger.warning("[Epoch {} Batch {}] Non-finite total loss. Skip.".format(epoch, idx + 1))
                num_skipped += 1
                continue

            scaler.scale(loss).backward()
            if opt.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(graph_batches, update_grid=False)

            if not torch.isfinite(logits).all():
                logger.warning("[Epoch {} Batch {}] Non-finite logits. Skip.".format(epoch, idx + 1))
                num_skipped += 1
                continue

            base_loss = criterion(logits.float(), y)
            if not torch.isfinite(base_loss):
                logger.warning("[Epoch {} Batch {}] Non-finite base loss. Skip.".format(epoch, idx + 1))
                num_skipped += 1
                continue

            if hasattr(model, "kan_regularization_loss"):
                kan_reg = model.kan_regularization_loss(
                    regularize_activation=opt.kan_reg_activation,
                    regularize_entropy=opt.kan_reg_entropy,
                )
            else:
                kan_reg = torch.tensor(0.0, device=logits.device)

            loss = base_loss + opt.kan_reg_lambda * kan_reg
            if not torch.isfinite(loss):
                logger.warning("[Epoch {} Batch {}] Non-finite total loss. Skip.".format(epoch, idx + 1))
                num_skipped += 1
                continue

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
        num_valid_batches += 1

        if (idx + 1) % opt.print_freq == 0:
            metrics = compute_metrics(all_labels, all_preds)
            logger.info(
                "Train Epoch [{}] Batch [{}/{}] | CurLoss {:.4f} | AvgLoss {:.4f} | "
                "Base {:.4f} | KAN {:.4f} | Acc {:.2f}% | Macro-F1 {:.4f} | Skipped {}".format(
                    epoch,
                    idx + 1,
                    len(loader),
                    float(loss.item()),
                    total_loss / max(1, total_samples),
                    total_base_loss / max(1, total_samples),
                    total_kan_reg / max(1, total_samples),
                    metrics["acc"] * 100.0,
                    metrics["macro_f1"],
                    num_skipped,
                )
            )

    epoch_time = time.time() - start_time
    metrics = compute_metrics(all_labels, all_preds)
    metrics.update({
        "loss": total_loss / max(1, total_samples),
        "base_loss": total_base_loss / max(1, total_samples),
        "kan_reg": total_kan_reg / max(1, total_samples),
        "epoch_time_sec": epoch_time,
        "num_skipped_batches": num_skipped,
        "num_valid_batches": num_valid_batches,
    })
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, opt):
    model.eval()

    total_loss = 0.0
    total_base_loss = 0.0
    total_kan_reg = 0.0
    total_samples = 0

    all_preds = []
    all_labels = []

    for batch in loader:
        batch = move_sequence_batch_to_device(batch, opt.device)
        graph_batches = batch["graph_batches"]
        y = batch["y"]

        logits = model(graph_batches, update_grid=False)
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

    logger.info("=" * 100)
    logger.info("Start training GraphTemporalKAN")
    logger.info(json.dumps(vars(opt), indent=2))
    logger.info("=" * 100)

    set_seed(opt.seed)

    if torch.cuda.is_available() and opt.device != "cpu":
        logger.info("Using GPU: {}".format(torch.cuda.get_device_name(0)))
    else:
        opt.device = "cpu"
        logger.info("Using CPU.")

    seq_folder = Path(opt.sequence_folder)

    # --------------------------------------------------------
    # Load full datasets first
    # --------------------------------------------------------
    train_dataset_full = SequenceGraphDataset(
        seq_path=str(seq_folder / "train_seq.pt"),
        max_samples=opt.max_train_sequences,
        return_meta=True,
    )
    val_dataset_full = SequenceGraphDataset(
        seq_path=str(seq_folder / "val_seq.pt"),
        max_samples=opt.max_val_sequences,
        return_meta=True,
    )
    test_dataset_full = SequenceGraphDataset(
        seq_path=str(seq_folder / "test_seq.pt"),
        max_samples=opt.max_test_sequences,
        return_meta=True,
    )

    # Metadata inferred before subsampling
    label_mapping = infer_label_mapping_from_any_dataset(train_dataset_full)
    if not label_mapping:
        label_mapping = infer_label_mapping_from_any_dataset(val_dataset_full)
    if not label_mapping:
        label_mapping = infer_label_mapping_from_any_dataset(test_dataset_full)

    label_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
    num_classes = len(label_names)

    seq_len = infer_seq_len_from_any_dataset(train_dataset_full)
    node_feat_dim = infer_node_feat_dim_from_any_dataset(train_dataset_full)
    edge_attr_dim = infer_edge_attr_dim_from_any_dataset(train_dataset_full)
    num_ids = infer_num_ids_from_any_dataset(train_dataset_full)

    logger.info("Label mapping: {}".format(label_mapping))
    logger.info("Full train sequences: {}".format(len(train_dataset_full)))
    logger.info("Full val sequences  : {}".format(len(val_dataset_full)))
    logger.info("Full test sequences : {}".format(len(test_dataset_full)))
    logger.info("seq_len             : {}".format(seq_len))
    logger.info("node_feat_dim       : {}".format(node_feat_dim))
    logger.info("edge_attr_dim       : {}".format(edge_attr_dim))
    logger.info("num_ids             : {}".format(num_ids))
    logger.info("Full train class distribution: {}".format(
        dict(zip(label_names, get_class_counts_from_any_dataset(train_dataset_full, num_classes)))
    ))

    # --------------------------------------------------------
    # Stratified subsampling at sequence level
    # --------------------------------------------------------
    train_dataset, train_selected, train_counts_sub = stratified_subsample_sequence_dataset(
        train_dataset_full,
        frac=opt.subsample_train_frac,
        min_per_class=opt.subsample_min_per_class,
        seed=opt.subsample_seed,
    )
    val_dataset, val_selected, val_counts_sub = stratified_subsample_sequence_dataset(
        val_dataset_full,
        frac=opt.subsample_val_frac,
        min_per_class=opt.subsample_min_per_class,
        seed=opt.subsample_seed + 1,
    )
    test_dataset, test_selected, test_counts_sub = stratified_subsample_sequence_dataset(
        test_dataset_full,
        frac=opt.subsample_test_frac,
        min_per_class=opt.subsample_min_per_class,
        seed=opt.subsample_seed + 2,
    )

    logger.info("Train sequences after stratified subsample: {}".format(len(train_dataset)))
    logger.info("Val sequences after stratified subsample  : {}".format(len(val_dataset)))
    logger.info("Test sequences after stratified subsample : {}".format(len(test_dataset)))
    logger.info("Train class distribution after subsample: {}".format(
        {label_mapping[k]: v for k, v in sorted(train_counts_sub.items())}
    ))
    logger.info("Val class distribution after subsample  : {}".format(
        {label_mapping[k]: v for k, v in sorted(val_counts_sub.items())}
    ))
    logger.info("Test class distribution after subsample : {}".format(
        {label_mapping[k]: v for k, v in sorted(test_counts_sub.items())}
    ))

    # Save selected indices for reproducibility
    with open(os.path.join(opt.save_folder, "subsample_indices.json"), "w", encoding="utf-8") as f:
        json.dump({
            "train_selected_indices": train_selected,
            "val_selected_indices": val_selected,
            "test_selected_indices": test_selected,
        }, f)

    # --------------------------------------------------------
    # Sampler / loaders
    # --------------------------------------------------------
    train_sampler = None
    if opt.use_weighted_sampler:
        train_sampler = build_weighted_sampler(train_dataset, num_classes)
        logger.info("Using WeightedRandomSampler")

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        collate_fn=collate_sequence_graphs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        collate_fn=collate_sequence_graphs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        collate_fn=collate_sequence_graphs,
    )

    # --------------------------------------------------------
    # Build graph encoder + temporal model
    # --------------------------------------------------------
    logger.info("Initializing model...")

    graph_encoder = GraphAttentionKAN(
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
    )

    graph_emb_dim = opt.hidden_dim * 2

    # model = GraphTemporalKAN(
    #     graph_encoder=graph_encoder,
    #     num_classes=num_classes,
    #     seq_len=seq_len,
    #     graph_emb_dim=graph_emb_dim,
    #     temporal_hidden_dim=opt.temporal_hidden_dim,
    #     num_transformer_layers=opt.num_transformer_layers,
    #     num_transformer_heads=opt.num_transformer_heads,
    #     dropout=opt.dropout,
    #     kan_hidden=opt.kan_hidden,
    #     kan_grid_size=opt.kan_grid_size,
    #     kan_spline_order=opt.kan_spline_order,
    # ).to(opt.device)

    if opt.sequence_model_mode == "transformer":
        model = GraphTemporalKAN(
            graph_encoder=graph_encoder,
            num_classes=num_classes,
            seq_len=seq_len,
            graph_emb_dim=graph_emb_dim,
            temporal_hidden_dim=opt.temporal_hidden_dim,
            num_transformer_layers=opt.num_transformer_layers,
            num_transformer_heads=opt.num_transformer_heads,
            dropout=opt.dropout,
            kan_hidden=opt.kan_hidden,
            kan_grid_size=opt.kan_grid_size,
            kan_spline_order=opt.kan_spline_order,
        ).to(opt.device)

    elif opt.sequence_model_mode == "last_only":
        model = GraphSequenceLastKAN(
            graph_encoder=graph_encoder,
            num_classes=num_classes,
            graph_emb_dim=graph_emb_dim,
            kan_hidden=opt.kan_hidden,
            kan_grid_size=opt.kan_grid_size,
            kan_spline_order=opt.kan_spline_order,
        ).to(opt.device)


    if opt.graph_encoder_ckpt:
        load_graph_encoder_checkpoint(
            model=model,
            ckpt_path=opt.graph_encoder_ckpt,
            logger=logger,
            strict=opt.strict_graph_encoder_load,
        )
    else:
        logger.info("No pretrained graph encoder checkpoint provided. Training from scratch.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: {:,}".format(n_params))

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
    cls_num_list = get_class_counts_from_any_dataset(train_dataset, num_classes)
    if opt.use_class_weights:
        class_weights = get_class_weights_tensor(cls_num_list, opt.device)
        logger.info("Using class weights: {}".format(class_weights.detach().cpu().numpy().tolist()))

    criterion = build_criterion(opt, cls_num_list, class_weights, logger)
    scaler = torch.amp.GradScaler("cuda", enabled=(opt.amp and opt.device.startswith("cuda")))

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------
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
        logger.info("Resumed from {}, start_epoch={}".format(opt.resume, start_epoch))

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    for epoch in range(start_epoch, opt.epochs + 1):
        if opt.freeze_graph_encoder_epochs > 0 and epoch <= opt.freeze_graph_encoder_epochs:
            set_requires_grad(model.graph_encoder, False)
            logger.info("Epoch {}: graph encoder frozen".format(epoch))
        else:
            set_requires_grad(model.graph_encoder, True)

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            epoch=epoch,
            opt=opt,
            logger=logger,
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

        logger.info("\n" + "=" * 100)
        logger.info("Epoch {}/{}".format(epoch, opt.epochs))
        logger.info("LR: {:.8f}".format(lr_now))
        logger.info(
            "Train | loss={:.4f} | base={:.4f} | kan={:.4f} | acc={:.2f}% | macro_f1={:.4f} | weighted_f1={:.4f} | skipped={}".format(
                train_metrics["loss"],
                train_metrics["base_loss"],
                train_metrics["kan_reg"],
                train_metrics["acc"] * 100.0,
                train_metrics["macro_f1"],
                train_metrics["weighted_f1"],
                train_metrics["num_skipped_batches"],
            )
        )
        logger.info(
            "Val   | loss={:.4f} | base={:.4f} | kan={:.4f} | acc={:.2f}% | macro_f1={:.4f} | weighted_f1={:.4f} | bal_acc={:.4f}".format(
                val_metrics["loss"],
                val_metrics["base_loss"],
                val_metrics["kan_reg"],
                val_metrics["acc"] * 100.0,
                val_metrics["macro_f1"],
                val_metrics["weighted_f1"],
                val_metrics["balanced_acc"],
            )
        )
        logger.info(
            "Test  | loss={:.4f} | base={:.4f} | kan={:.4f} | acc={:.2f}% | macro_f1={:.4f} | weighted_f1={:.4f} | bal_acc={:.4f}".format(
                test_metrics["loss"],
                test_metrics["base_loss"],
                test_metrics["kan_reg"],
                test_metrics["acc"] * 100.0,
                test_metrics["macro_f1"],
                test_metrics["weighted_f1"],
                test_metrics["balanced_acc"],
            )
        )

        val_cm = confusion_matrix(y_val_true, y_val_pred, labels=list(range(num_classes)))
        logger.info("Val Confusion Matrix:\n{}".format(val_cm))

        last_ckpt_path = os.path.join(opt.save_folder, "{}_last.pth".format(opt.model_name))
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

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]

            best_path = os.path.join(opt.save_folder, "{}_best_macro_f1.pth".format(opt.model_name))
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
            logger.info("🔥 New best val macro-F1: {:.4f} | saved to {}".format(best_val_macro_f1, best_path))

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]

            best_acc_path = os.path.join(opt.save_folder, "{}_best_acc.pth".format(opt.model_name))
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
            logger.info("✅ New best val accuracy: {:.2f}%".format(best_val_acc * 100.0))

        with open(os.path.join(opt.save_folder, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------
    logger.info("\n" + "=" * 100)
    logger.info("Final evaluation on TEST using best_macro_f1 checkpoint")
    logger.info("=" * 100)

    best_path = os.path.join(opt.save_folder, "{}_best_macro_f1.pth".format(opt.model_name))
    best_ckpt = torch.load(best_path, map_location=opt.device, weights_only=False)
    model.load_state_dict(best_ckpt["model"])

    test_metrics, y_test_true, y_test_pred = evaluate(model, test_loader, criterion, opt)
    test_cm = confusion_matrix(y_test_true, y_test_pred, labels=list(range(num_classes)))

    logger.info(
        "BEST TEST | loss={:.4f} | acc={:.2f}% | macro_f1={:.4f} | weighted_f1={:.4f} | balanced_acc={:.4f}".format(
            test_metrics["loss"],
            test_metrics["acc"] * 100.0,
            test_metrics["macro_f1"],
            test_metrics["weighted_f1"],
            test_metrics["balanced_acc"],
        )
    )
    logger.info("Final Test Confusion Matrix:\n{}".format(test_cm))

    report = classification_report(
        y_test_true,
        y_test_pred,
        labels=list(range(num_classes)),
        target_names=label_names,
        digits=4,
        zero_division=0,
    )
    logger.info("\nFinal Classification Report:\n{}".format(report))

    np.save(os.path.join(opt.save_folder, "final_confusion_matrix.npy"), test_cm)
    with open(os.path.join(opt.save_folder, "final_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    logger.info("\nTraining finished.")
    logger.info("Best val macro-F1: {:.4f}".format(best_val_macro_f1))
    logger.info("Best val acc     : {:.2f}%".format(best_val_acc * 100.0))
    logger.info("Artifacts saved in: {}".format(opt.save_folder))


if __name__ == "__main__":
    main()