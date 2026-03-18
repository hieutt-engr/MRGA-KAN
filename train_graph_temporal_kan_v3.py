
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
from networks.graph_attention_encoder import GraphAttentionEncoder
from networks.graph_temporal_kan_v3 import GraphTemporalKANv3
from networks.efficient_kan import KAN


class GraphSequenceLastKANv2(nn.Module):
    def __init__(
        self,
        graph_encoder,
        num_classes,
        graph_emb_dim,
        kan_hidden=128,
        kan_grid_size=5,
        kan_spline_order=3,
    ):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.head = KAN(
            layers_hidden=[graph_emb_dim, kan_hidden, num_classes],
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
            base_activation=torch.nn.SiLU,
        )

    def forward(self, batched_graph_list, update_grid=False, return_aux=False):
        last_graph_batch = batched_graph_list[-1]
        g = self.graph_encoder(last_graph_batch)

        if g.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                logits = self.head(g.float(), update_grid=update_grid)
        else:
            logits = self.head(g.float(), update_grid=update_grid)

        if return_aux:
            return logits, {"last_graph_embedding": g}
        return logits

    def kan_regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return self.head.regularization_loss(
            regularize_activation=regularize_activation,
            regularize_entropy=regularize_entropy,
        )


def parse_option():
    parser = argparse.ArgumentParser("Train GraphTemporalKANv3 for IVN IDS")
    parser.add_argument("--sequence_folder", type=str, required=True)
    parser.add_argument("--save_folder", type=str, default="./save/graph_temporal_kan_v3")
    parser.add_argument("--model_name", type=str, default="graph_temporal_kan_v3")

    parser.add_argument("--graph_encoder_ckpt", type=str, default="")
    parser.add_argument("--strict_graph_encoder_load", action="store_true")
    parser.add_argument("--freeze_graph_encoder_epochs", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--max_train_sequences", type=int, default=0)
    parser.add_argument("--max_val_sequences", type=int, default=0)
    parser.add_argument("--max_test_sequences", type=int, default=0)

    parser.add_argument("--subsample_train_frac", type=float, default=1.0)
    parser.add_argument("--subsample_val_frac", type=float, default=1.0)
    parser.add_argument("--subsample_test_frac", type=float, default=1.0)
    parser.add_argument("--subsample_min_per_class", type=int, default=0)
    parser.add_argument("--subsample_seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=str, default="")

    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "cosine_restart"])
    parser.add_argument("--t0", type=int, default=10)
    parser.add_argument("--tmult", type=int, default=2)
    parser.add_argument("--eta_min", type=float, default=1e-6)

    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--use_class_weights", action="store_true")

    parser.add_argument("--loss_name", type=str, default="ce", choices=["ce", "focal", "ldam", "polyfocal"])
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--ldam_max_m", type=float, default=0.5)
    parser.add_argument("--ldam_s", type=float, default=30.0)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--id_emb_dim", type=int, default=32)
    parser.add_argument("--rel_emb_dim", type=int, default=8)
    parser.add_argument("--num_relations", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--ffn_ratio", type=float, default=2.0)

    parser.add_argument("--temporal_hidden_dim", type=int, default=256)
    parser.add_argument("--num_transformer_layers", type=int, default=2)
    parser.add_argument("--num_transformer_heads", type=int, default=4)
    parser.add_argument("--sequence_model_mode", type=str, default="transformer", choices=["transformer", "last_only"])
    parser.add_argument("--use_cls_token", dest="use_cls_token", action="store_true")
    parser.add_argument("--no_cls_token", dest="use_cls_token", action="store_false")
    parser.set_defaults(use_cls_token=True)

    parser.add_argument("--kan_hidden", type=int, default=128)
    parser.add_argument("--kan_grid_size", type=int, default=5)
    parser.add_argument("--kan_spline_order", type=int, default=3)

    parser.add_argument("--kan_reg_lambda", type=float, default=1e-4)
    parser.add_argument("--kan_reg_activation", type=float, default=1.0)
    parser.add_argument("--kan_reg_entropy", type=float, default=1.0)

    # new: auxiliary supervision on last graph embedding
    parser.add_argument("--aux_last_loss_weight", type=float, default=0.2,
                        help="Weight for auxiliary CE loss on the last-graph embedding.")

    # new: attention logging
    parser.add_argument("--attention_log_classes", type=str, default="4,6",
                        help="Comma-separated class ids for attention logging. Example: 4,6")
    parser.add_argument("--attention_log_max_sequences", type=int, default=2,
                        help="How many example sequences per target class to print in eval logs.")
    parser.add_argument("--attention_log_split", type=str, default="val", choices=["none", "val", "test"])

    opt = parser.parse_args()
    os.makedirs(opt.save_folder, exist_ok=True)
    return opt


def set_logger(log_path):
    logger = logging.getLogger("train_graph_temporal_kan_v3")
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


def get_base_dataset(dataset):
    return dataset.dataset if isinstance(dataset, Subset) else dataset


def get_labels_from_any_dataset(dataset):
    if isinstance(dataset, Subset):
        base = dataset.dataset
        return [int(base.samples[idx]["y"]) for idx in dataset.indices]
    return [int(s["y"]) for s in dataset.samples]


def get_class_counts_from_any_dataset(dataset, num_classes):
    labels = get_labels_from_any_dataset(dataset)
    counts = [0 for _ in range(num_classes)]
    for y in labels:
        counts[y] += 1
    return counts


def infer_num_ids_from_any_dataset(dataset):
    return get_base_dataset(dataset).infer_num_ids()


def infer_node_feat_dim_from_any_dataset(dataset):
    return get_base_dataset(dataset).infer_node_feat_dim()


def infer_edge_attr_dim_from_any_dataset(dataset):
    return get_base_dataset(dataset).infer_edge_attr_dim()


def infer_seq_len_from_any_dataset(dataset):
    return get_base_dataset(dataset).seq_len


def infer_label_mapping_from_any_dataset(dataset):
    return get_base_dataset(dataset).label_mapping


def stratified_subsample_sequence_dataset(dataset, frac=1.0, min_per_class=0, seed=42):
    if frac >= 1.0:
        labels = get_labels_from_any_dataset(dataset)
        counts = dict(sorted(dict((y, labels.count(y)) for y in set(labels)).items()))
        return dataset, list(range(len(dataset))), counts

    rng = np.random.default_rng(seed)
    base = get_base_dataset(dataset)
    candidate_indices = dataset.indices if isinstance(dataset, Subset) else list(range(len(base.samples)))
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
    return Subset(base, selected), selected, class_counts


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
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


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
    raise ValueError(f"Unsupported loss_name: {opt.loss_name}")


def save_checkpoint(state, save_path):
    torch.save(state, save_path)


def set_requires_grad(module, requires_grad):
    for p in module.parameters():
        p.requires_grad = requires_grad


def load_graph_encoder_checkpoint(model, ckpt_path, logger, strict=False):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    filtered = {}
    for k, v in state.items():
        if k.startswith("head."):
            continue
        if k.startswith("graph_encoder."):
            new_k = k.replace("graph_encoder.", "", 1)
            if not new_k.startswith("head."):
                filtered[new_k] = v
        else:
            if not k.startswith("head."):
                filtered[k] = v
    missing, unexpected = model.graph_encoder.load_state_dict(filtered, strict=strict)
    logger.info(f"Loaded graph encoder. Missing={missing}, Unexpected={unexpected}")


def parse_attention_classes(opt):
    if not opt.attention_log_classes.strip():
        return []
    return [int(x.strip()) for x in opt.attention_log_classes.split(",") if x.strip() != ""]


def _format_attention_summary(attn_info, label_mapping):
    if not attn_info:
        return ""
    lines = []
    avg_by_class = attn_info.get("avg_attention_by_class", {})
    examples = attn_info.get("examples", {})
    for cls_id, avg_weights in sorted(avg_by_class.items(), key=lambda kv: kv[0]):
        cls_name = label_mapping.get(cls_id, str(cls_id))
        arr = np.array(avg_weights, dtype=np.float64)
        lines.append(f"Attention avg for class {cls_id} ({cls_name}): {np.round(arr, 4).tolist()}")
        for ex in examples.get(cls_id, []):
            seq_id = ex.get("sequence_id", "NA")
            pred = ex.get("pred", -1)
            pred_name = label_mapping.get(pred, str(pred))
            weights = ex.get("attn_weights", [])
            graph_labels = ex.get("graph_labels", [])
            lines.append(
                f"  Example seq={seq_id} pred={pred}({pred_name}) attn={np.round(np.array(weights), 4).tolist()} graph_labels={graph_labels}"
            )
    return "\n".join(lines)


def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch, opt, logger):
    model.train()
    total_loss = total_base_loss = total_kan_reg = total_aux_last_loss = 0.0
    total_samples = 0
    all_preds, all_labels = [], []
    num_skipped = 0
    num_valid_batches = 0
    start_time = time.time()

    use_aux = (opt.aux_last_loss_weight > 0.0 and opt.sequence_model_mode == "transformer")

    for idx, batch in enumerate(loader):
        batch = move_sequence_batch_to_device(batch, opt.device)
        graph_batches = batch["graph_batches"]
        y = batch["y"]
        optimizer.zero_grad(set_to_none=True)

        def forward_and_loss():
            if use_aux:
                logits, aux = model(graph_batches, update_grid=False, return_aux=True)
                aux_last_logits = aux.get("aux_last_logits", None)
            else:
                logits = model(graph_batches, update_grid=False)
                aux_last_logits = None

            if not torch.isfinite(logits).all():
                return None, None, None, None, None

            seq_loss = criterion(logits.float(), y)
            if not torch.isfinite(seq_loss):
                return None, None, None, None, None

            aux_last_loss = torch.tensor(0.0, device=logits.device)
            if aux_last_logits is not None:
                aux_last_loss = criterion(aux_last_logits.float(), y)

            if hasattr(model, "kan_regularization_loss"):
                kan_reg = model.kan_regularization_loss(
                    regularize_activation=opt.kan_reg_activation,
                    regularize_entropy=opt.kan_reg_entropy,
                )
            else:
                kan_reg = torch.tensor(0.0, device=logits.device)

            base_loss = seq_loss + opt.aux_last_loss_weight * aux_last_loss
            loss = base_loss + opt.kan_reg_lambda * kan_reg
            if not torch.isfinite(loss):
                return None, None, None, None, None
            return logits, loss, base_loss, kan_reg, aux_last_loss

        if opt.amp and opt.device.startswith("cuda"):
            with torch.amp.autocast("cuda", enabled=True):
                logits, loss, base_loss, kan_reg, aux_last_loss = forward_and_loss()
            if logits is None:
                logger.warning(f"[Epoch {epoch} Batch {idx + 1}] Non-finite batch. Skip.")
                num_skipped += 1
                continue
            scaler.scale(loss).backward()
            if opt.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, loss, base_loss, kan_reg, aux_last_loss = forward_and_loss()
            if logits is None:
                logger.warning(f"[Epoch {epoch} Batch {idx + 1}] Non-finite batch. Skip.")
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
        total_aux_last_loss += float(aux_last_loss.item()) * bs
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(y.detach().cpu().numpy().tolist())
        num_valid_batches += 1

        if (idx + 1) % opt.print_freq == 0:
            metrics = compute_metrics(all_labels, all_preds)
            logger.info(
                "Train Epoch [{}] Batch [{}/{}] | CurLoss {:.4f} | AvgLoss {:.4f} | Base {:.4f} | AuxLast {:.4f} | KAN {:.4f} | Acc {:.2f}% | Macro-F1 {:.4f} | Skipped {}".format(
                    epoch, idx + 1, len(loader), float(loss.item()), total_loss / max(1, total_samples),
                    total_base_loss / max(1, total_samples), total_aux_last_loss / max(1, total_samples),
                    total_kan_reg / max(1, total_samples), metrics["acc"] * 100.0, metrics["macro_f1"], num_skipped,
                )
            )

    epoch_time = time.time() - start_time
    metrics = compute_metrics(all_labels, all_preds)
    metrics.update({
        "loss": total_loss / max(1, total_samples),
        "base_loss": total_base_loss / max(1, total_samples),
        "kan_reg": total_kan_reg / max(1, total_samples),
        "aux_last_loss": total_aux_last_loss / max(1, total_samples),
        "epoch_time_sec": epoch_time,
        "num_skipped_batches": num_skipped,
        "num_valid_batches": num_valid_batches,
    })
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, opt, collect_attention=False, target_attention_classes=None):
    model.eval()
    total_loss = total_base_loss = total_kan_reg = total_aux_last_loss = 0.0
    total_samples = 0
    all_preds, all_labels = [], []

    attn_sum = {}
    attn_count = {}
    attn_examples = defaultdict(list)
    target_attention_classes = target_attention_classes or []
    use_aux = (opt.aux_last_loss_weight > 0.0 and opt.sequence_model_mode == "transformer")

    for batch in loader:
        batch = move_sequence_batch_to_device(batch, opt.device)
        graph_batches = batch["graph_batches"]
        y = batch["y"]

        need_aux = use_aux or collect_attention
        if need_aux:
            logits, aux = model(graph_batches, update_grid=False, return_aux=True)
            aux_last_logits = aux.get("aux_last_logits", None)
            attn_weights = aux.get("attn_weights", None)
        else:
            logits = model(graph_batches, update_grid=False)
            aux_last_logits = None
            attn_weights = None

        seq_loss = criterion(logits.float(), y)
        aux_last_loss = torch.tensor(0.0, device=logits.device)
        if aux_last_logits is not None and use_aux:
            aux_last_loss = criterion(aux_last_logits.float(), y)

        if hasattr(model, "kan_regularization_loss"):
            kan_reg = model.kan_regularization_loss(
                regularize_activation=opt.kan_reg_activation,
                regularize_entropy=opt.kan_reg_entropy,
            )
        else:
            kan_reg = torch.tensor(0.0, device=logits.device)

        base_loss = seq_loss + opt.aux_last_loss_weight * aux_last_loss
        loss = base_loss + opt.kan_reg_lambda * kan_reg
        preds = logits.argmax(dim=1)

        bs = y.size(0)
        total_samples += bs
        total_loss += float(loss.item()) * bs
        total_base_loss += float(base_loss.item()) * bs
        total_kan_reg += float(kan_reg.item()) * bs
        total_aux_last_loss += float(aux_last_loss.item()) * bs

        y_cpu = y.detach().cpu().numpy().tolist()
        pred_cpu = preds.detach().cpu().numpy().tolist()
        all_preds.extend(pred_cpu)
        all_labels.extend(y_cpu)

        if collect_attention and attn_weights is not None:
            attn_cpu = attn_weights.detach().cpu().numpy()
            seq_ids = batch.get("sequence_ids", [f"seq_{i}" for i in range(len(y_cpu))])
            graph_labels_batch = batch.get("graph_labels", [[] for _ in range(len(y_cpu))])
            for i, cls_id in enumerate(y_cpu):
                if cls_id not in target_attention_classes:
                    continue
                vec = attn_cpu[i].tolist()
                if cls_id not in attn_sum:
                    attn_sum[cls_id] = np.zeros(len(vec), dtype=np.float64)
                    attn_count[cls_id] = 0
                attn_sum[cls_id] += np.asarray(vec, dtype=np.float64)
                attn_count[cls_id] += 1
                if len(attn_examples[cls_id]) < opt.attention_log_max_sequences:
                    attn_examples[cls_id].append({
                        "sequence_id": seq_ids[i],
                        "pred": int(pred_cpu[i]),
                        "attn_weights": vec,
                        "graph_labels": graph_labels_batch[i],
                    })

    metrics = compute_metrics(all_labels, all_preds)
    metrics.update({
        "loss": total_loss / max(1, total_samples),
        "base_loss": total_base_loss / max(1, total_samples),
        "kan_reg": total_kan_reg / max(1, total_samples),
        "aux_last_loss": total_aux_last_loss / max(1, total_samples),
    })

    attn_info = None
    if collect_attention:
        avg_attention_by_class = {}
        for cls_id in attn_sum:
            avg_attention_by_class[cls_id] = (attn_sum[cls_id] / max(1, attn_count[cls_id])).tolist()
        attn_info = {
            "avg_attention_by_class": avg_attention_by_class,
            "examples": dict(attn_examples),
        }

    return metrics, all_labels, all_preds, attn_info


def main():
    opt = parse_option()
    log_file = os.path.join(opt.save_folder, "training_log.txt")
    logger = set_logger(log_file)

    logger.info("=" * 100)
    logger.info("Start training GraphTemporalKANv3")
    logger.info(json.dumps(vars(opt), indent=2))
    logger.info("=" * 100)

    set_seed(opt.seed)
    if torch.cuda.is_available() and opt.device != "cpu":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        opt.device = "cpu"
        logger.info("Using CPU.")

    seq_folder = Path(opt.sequence_folder)
    train_dataset_full = SequenceGraphDataset(str(seq_folder / "train_seq.pt"), max_samples=opt.max_train_sequences, return_meta=True)
    val_dataset_full = SequenceGraphDataset(str(seq_folder / "val_seq.pt"), max_samples=opt.max_val_sequences, return_meta=True)
    test_dataset_full = SequenceGraphDataset(str(seq_folder / "test_seq.pt"), max_samples=opt.max_test_sequences, return_meta=True)

    label_mapping = infer_label_mapping_from_any_dataset(train_dataset_full) or infer_label_mapping_from_any_dataset(val_dataset_full) or infer_label_mapping_from_any_dataset(test_dataset_full)
    label_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
    num_classes = len(label_names)
    seq_len = infer_seq_len_from_any_dataset(train_dataset_full)
    node_feat_dim = infer_node_feat_dim_from_any_dataset(train_dataset_full)
    edge_attr_dim = infer_edge_attr_dim_from_any_dataset(train_dataset_full)
    num_ids = infer_num_ids_from_any_dataset(train_dataset_full)

    logger.info(f"Label mapping: {label_mapping}")
    logger.info(f"Full train sequences: {len(train_dataset_full)}")
    logger.info(f"Full val sequences  : {len(val_dataset_full)}")
    logger.info(f"Full test sequences : {len(test_dataset_full)}")
    logger.info(f"seq_len             : {seq_len}")
    logger.info(f"node_feat_dim       : {node_feat_dim}")
    logger.info(f"edge_attr_dim       : {edge_attr_dim}")
    logger.info(f"num_ids             : {num_ids}")
    logger.info("Full train class distribution: {}".format(dict(zip(label_names, get_class_counts_from_any_dataset(train_dataset_full, num_classes)))))

    train_dataset, train_selected, train_counts_sub = stratified_subsample_sequence_dataset(train_dataset_full, frac=opt.subsample_train_frac, min_per_class=opt.subsample_min_per_class, seed=opt.subsample_seed)
    val_dataset, val_selected, val_counts_sub = stratified_subsample_sequence_dataset(val_dataset_full, frac=opt.subsample_val_frac, min_per_class=opt.subsample_min_per_class, seed=opt.subsample_seed + 1)
    test_dataset, test_selected, test_counts_sub = stratified_subsample_sequence_dataset(test_dataset_full, frac=opt.subsample_test_frac, min_per_class=opt.subsample_min_per_class, seed=opt.subsample_seed + 2)

    logger.info(f"Train sequences after stratified subsample: {len(train_dataset)}")
    logger.info(f"Val sequences after stratified subsample  : {len(val_dataset)}")
    logger.info(f"Test sequences after stratified subsample : {len(test_dataset)}")
    logger.info("Train class distribution after subsample: {}".format({label_mapping[k]: v for k, v in sorted(train_counts_sub.items())}))
    logger.info("Val class distribution after subsample  : {}".format({label_mapping[k]: v for k, v in sorted(val_counts_sub.items())}))
    logger.info("Test class distribution after subsample : {}".format({label_mapping[k]: v for k, v in sorted(test_counts_sub.items())}))

    with open(os.path.join(opt.save_folder, "subsample_indices.json"), "w", encoding="utf-8") as f:
        json.dump({"train_selected_indices": train_selected, "val_selected_indices": val_selected, "test_selected_indices": test_selected}, f)

    train_sampler = build_weighted_sampler(train_dataset, num_classes) if opt.use_weighted_sampler else None
    if train_sampler is not None:
        logger.info("Using WeightedRandomSampler")

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=opt.num_workers, pin_memory=opt.pin_memory, collate_fn=collate_sequence_graphs)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=opt.pin_memory, collate_fn=collate_sequence_graphs)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                             pin_memory=opt.pin_memory, collate_fn=collate_sequence_graphs)

    logger.info("Initializing model...")
    graph_encoder = GraphAttentionEncoder(
        node_feat_dim=node_feat_dim,
        edge_attr_dim=edge_attr_dim,
        num_ids=num_ids,
        hidden_dim=opt.hidden_dim,
        num_layers=opt.num_layers,
        heads=opt.heads,
        id_emb_dim=opt.id_emb_dim,
        rel_emb_dim=opt.rel_emb_dim,
        num_relations=opt.num_relations,
        dropout=opt.dropout,
        ffn_ratio=opt.ffn_ratio,
    )
    graph_emb_dim = opt.hidden_dim * 2

    if opt.sequence_model_mode == "transformer":
        model = GraphTemporalKANv3(
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
            use_cls_token=opt.use_cls_token,
        ).to(opt.device)
    else:
        model = GraphSequenceLastKANv2(
            graph_encoder=graph_encoder,
            num_classes=num_classes,
            graph_emb_dim=graph_emb_dim,
            kan_hidden=opt.kan_hidden,
            kan_grid_size=opt.kan_grid_size,
            kan_spline_order=opt.kan_spline_order,
        ).to(opt.device)

    if opt.graph_encoder_ckpt:
        load_graph_encoder_checkpoint(model, opt.graph_encoder_ckpt, logger, strict=opt.strict_graph_encoder_load)
    else:
        logger.info("No pretrained graph encoder checkpoint provided. Training from scratch.")

    logger.info("Trainable parameters: {:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    if opt.scheduler == "cosine_restart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt.t0, T_mult=opt.tmult, eta_min=opt.eta_min)
    elif opt.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=opt.eta_min)
    else:
        scheduler = None

    class_weights = None
    cls_num_list = get_class_counts_from_any_dataset(train_dataset, num_classes)
    if opt.use_class_weights:
        class_weights = get_class_weights_tensor(cls_num_list, opt.device)
        logger.info("Using class weights: {}".format(class_weights.detach().cpu().numpy().tolist()))

    criterion = build_criterion(opt, cls_num_list, class_weights, logger)
    scaler = torch.amp.GradScaler("cuda", enabled=(opt.amp and opt.device.startswith("cuda")))

    start_epoch = 1
    best_val_macro_f1 = -1.0
    best_val_acc = -1.0
    history = []
    if opt.resume:
        ckpt = torch.load(opt.resume, map_location=opt.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_macro_f1 = ckpt.get("best_val_macro_f1", -1.0)
        best_val_acc = ckpt.get("best_val_acc", -1.0)
        history = ckpt.get("history", [])
        logger.info(f"Resumed from {opt.resume}, start_epoch={start_epoch}")

    target_attention_classes = parse_attention_classes(opt)

    for epoch in range(start_epoch, opt.epochs + 1):
        if opt.freeze_graph_encoder_epochs > 0 and epoch <= opt.freeze_graph_encoder_epochs:
            set_requires_grad(model.graph_encoder, False)
            logger.info(f"Epoch {epoch}: graph encoder frozen")
        else:
            set_requires_grad(model.graph_encoder, True)

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, epoch, opt, logger)

        collect_val_attn = (opt.attention_log_split == "val" and opt.sequence_model_mode == "transformer")
        collect_test_attn = (opt.attention_log_split == "test" and opt.sequence_model_mode == "transformer")

        val_metrics, y_val_true, y_val_pred, val_attn_info = evaluate(
            model, val_loader, criterion, opt,
            collect_attention=collect_val_attn,
            target_attention_classes=target_attention_classes,
        )
        test_metrics, y_test_true, y_test_pred, test_attn_info = evaluate(
            model, test_loader, criterion, opt,
            collect_attention=collect_test_attn,
            target_attention_classes=target_attention_classes,
        )

        if scheduler is not None:
            if opt.scheduler == "cosine_restart":
                scheduler.step(epoch - 1 + 1.0)
            else:
                scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "lr": lr_now, "train": train_metrics, "val": val_metrics, "test": test_metrics})

        logger.info("\n" + "=" * 100)
        logger.info(f"Epoch {epoch}/{opt.epochs}")
        logger.info(f"LR: {lr_now:.8f}")
        logger.info(
            "Train | loss={:.4f} | base={:.4f} | aux_last={:.4f} | kan={:.4f} | acc={:.2f}% | macro_f1={:.4f} | weighted_f1={:.4f} | skipped={}".format(
                train_metrics["loss"], train_metrics["base_loss"], train_metrics["aux_last_loss"], train_metrics["kan_reg"],
                train_metrics["acc"] * 100.0, train_metrics["macro_f1"], train_metrics["weighted_f1"], train_metrics["num_skipped_batches"]
            )
        )
        logger.info(
            "Val   | loss={:.4f} | base={:.4f} | aux_last={:.4f} | kan={:.4f} | acc={:.2f}% | macro_f1={:.4f} | weighted_f1={:.4f} | bal_acc={:.4f}".format(
                val_metrics["loss"], val_metrics["base_loss"], val_metrics["aux_last_loss"], val_metrics["kan_reg"],
                val_metrics["acc"] * 100.0, val_metrics["macro_f1"], val_metrics["weighted_f1"], val_metrics["balanced_acc"]
            )
        )
        logger.info(
            "Test  | loss={:.4f} | base={:.4f} | aux_last={:.4f} | kan={:.4f} | acc={:.2f}% | macro_f1={:.4f} | weighted_f1={:.4f} | bal_acc={:.4f}".format(
                test_metrics["loss"], test_metrics["base_loss"], test_metrics["aux_last_loss"], test_metrics["kan_reg"],
                test_metrics["acc"] * 100.0, test_metrics["macro_f1"], test_metrics["weighted_f1"], test_metrics["balanced_acc"]
            )
        )

        val_cm = confusion_matrix(y_val_true, y_val_pred, labels=list(range(num_classes)))
        logger.info("Val Confusion Matrix:\n{}".format(val_cm))
        if val_attn_info is not None:
            logger.info("Val attention summary:\n{}".format(_format_attention_summary(val_attn_info, label_mapping)))
        if test_attn_info is not None:
            logger.info("Test attention summary:\n{}".format(_format_attention_summary(test_attn_info, label_mapping)))

        last_ckpt_path = os.path.join(opt.save_folder, f"{opt.model_name}_last.pth")
        save_checkpoint({
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val_macro_f1": best_val_macro_f1, "best_val_acc": best_val_acc,
            "history": history, "label_mapping": label_mapping, "args": vars(opt),
        }, last_ckpt_path)

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_path = os.path.join(opt.save_folder, f"{opt.model_name}_best_macro_f1.pth")
            save_checkpoint({
                "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_val_macro_f1": best_val_macro_f1, "best_val_acc": best_val_acc,
                "history": history, "label_mapping": label_mapping, "args": vars(opt),
            }, best_path)
            np.save(os.path.join(opt.save_folder, "best_val_confusion_matrix.npy"), val_cm)
            logger.info(f"🔥 New best val macro-F1: {best_val_macro_f1:.4f} | saved to {best_path}")

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_acc_path = os.path.join(opt.save_folder, f"{opt.model_name}_best_acc.pth")
            save_checkpoint({
                "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_val_macro_f1": best_val_macro_f1, "best_val_acc": best_val_acc,
                "history": history, "label_mapping": label_mapping, "args": vars(opt),
            }, best_acc_path)
            logger.info(f"✅ New best val accuracy: {best_val_acc * 100.0:.2f}%")

        with open(os.path.join(opt.save_folder, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    logger.info("\n" + "=" * 100)
    logger.info("Final evaluation on TEST using best_macro_f1 checkpoint")
    logger.info("=" * 100)

    best_path = os.path.join(opt.save_folder, f"{opt.model_name}_best_macro_f1.pth")
    best_ckpt = torch.load(best_path, map_location=opt.device, weights_only=False)
    model.load_state_dict(best_ckpt["model"])

    final_collect_attention = (opt.attention_log_split in ["val", "test"] and opt.sequence_model_mode == "transformer")
    test_metrics, y_test_true, y_test_pred, test_attn_info = evaluate(
        model, test_loader, criterion, opt,
        collect_attention=final_collect_attention,
        target_attention_classes=target_attention_classes,
    )
    test_cm = confusion_matrix(y_test_true, y_test_pred, labels=list(range(num_classes)))

    logger.info(
        "BEST TEST | loss={:.4f} | acc={:.2f}% | macro_f1={:.4f} | weighted_f1={:.4f} | balanced_acc={:.4f}".format(
            test_metrics["loss"], test_metrics["acc"] * 100.0, test_metrics["macro_f1"],
            test_metrics["weighted_f1"], test_metrics["balanced_acc"]
        )
    )
    logger.info("Final Test Confusion Matrix:\n{}".format(test_cm))
    if test_attn_info is not None:
        logger.info("Final attention summary:\n{}".format(_format_attention_summary(test_attn_info, label_mapping)))

    report = classification_report(y_test_true, y_test_pred, labels=list(range(num_classes)), target_names=label_names, digits=4, zero_division=0)
    logger.info("\nFinal Classification Report:\n{}".format(report))

    np.save(os.path.join(opt.save_folder, "final_confusion_matrix.npy"), test_cm)
    with open(os.path.join(opt.save_folder, "final_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    logger.info("\nTraining finished.")
    logger.info(f"Best val macro-F1: {best_val_macro_f1:.4f}")
    logger.info(f"Best val acc     : {best_val_acc * 100.0:.2f}%")
    logger.info(f"Artifacts saved in: {opt.save_folder}")


if __name__ == "__main__":
    main()
