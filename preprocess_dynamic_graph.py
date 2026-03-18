#!/usr/bin/env python3
"""
preprocess_ivn_dynamic_graph.py

V3: Window-first preprocessing for IVN IDS
Target pipeline:
Dynamic multi-relational message graph
+ graph attention encoder
+ temporal transformer
+ KAN classifier head

Main idea:
1) Read CSV files.
2) Build multiclass message labels:
      attack == 0 -> normal
      attack == 1 -> <source filename stem>
3) Parse payload into 8 bytes.
4) Compute message-level engineered features.
5) Build candidate windows FIRST over full files.
6) Assign attack-aware window labels:
      no attack message -> normal
      else -> dominant attack class among attack messages
7) Split at WINDOW level (stratified by source_class + window_label).
8) Infer message splits from selected windows.
9) Fit train statistics on train messages only.
10) Save messages + windows + metadata + summary.

Important:
- `combined` is treated as an independent attack class.
- This version intentionally prefers safer splitting over dense overlapping windows.
- For leakage control, the default candidate sampling stride is:
      sampling_stride = max(window_size, stride)
  unless explicitly overridden.

Example:
python preprocess_ivn_dynamic_graph.py \
    --input_dir ./data/2017-subaru-forester/merged \
    --output_dir ./data/2017-subaru-forester/preprocessed_v3 \
    --window_size 64 \
    --stride 16 \
    --sampling_stride 64 \
    --min_attack_count 1 \
    --min_attack_ratio 0.0 \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


NORMAL_LABEL = "normal"


# ============================================================
# Config dataclasses
# ============================================================

@dataclass
class SplitConfig:
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    seed: int = 42


@dataclass
class WindowConfig:
    window_size: int = 64
    stride: int = 16
    sampling_stride: int = 0   # 0 means auto -> max(window_size, stride)
    min_attack_count: int = 1
    min_attack_ratio: float = 0.0
    keep_mixed_windows: bool = True


# ============================================================
# Utilities
# ============================================================

def safe_entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def robust_stats(series: pd.Series) -> Dict[str, float]:
    vals = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy()
    )
    if len(vals) == 0:
        return {"median": 0.0, "iqr": 1.0}

    q1 = np.percentile(vals, 25)
    med = np.percentile(vals, 50)
    q3 = np.percentile(vals, 75)
    iqr = float(q3 - q1)
    if iqr <= 1e-12:
        iqr = 1.0
    return {"median": float(med), "iqr": iqr}


def robust_scale(series: pd.Series, median: float, iqr: float) -> pd.Series:
    return (series - median) / max(iqr, 1e-12)


def ensure_hex_payload(x: object, expected_hex_len: int = 16) -> str:
    if pd.isna(x):
        s = ""
    else:
        s = str(x).strip().lower()

    if s.startswith("0x"):
        s = s[2:]

    s = "".join(ch for ch in s if ch in "0123456789abcdef")

    if len(s) < expected_hex_len:
        s = s.zfill(expected_hex_len)
    elif len(s) > expected_hex_len:
        s = s[:expected_hex_len]

    return s


def hex_to_8bytes(hex_str: str) -> List[int]:
    hex_str = ensure_hex_payload(hex_str, expected_hex_len=16)
    return [int(hex_str[i:i + 2], 16) for i in range(0, 16, 2)]


def bytes_to_bits(byte_values: List[int]) -> np.ndarray:
    arr = np.array(byte_values, dtype=np.uint8)
    return np.unpackbits(arr).astype(np.uint8)


def save_dataframe(df: pd.DataFrame, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = path_base.with_suffix(".parquet")
    csv_path = path_base.with_suffix(".csv")

    try:
        df.to_parquet(parquet_path, index=False)
        print(f"[Saved] {parquet_path}")
    except Exception as e:
        print(f"[Warn] Could not save parquet ({parquet_path.name}): {e}")
        df.to_csv(csv_path, index=False)
        print(f"[Saved] {csv_path}")


def save_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {path}")


def deterministic_split_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    """
    Split n items with a deterministic rule.
    If n is very small:
    - n=1 -> train
    - n=2 -> train, val
    - n=3 -> train, val, test
    """
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0
    if n == 3:
        return 1, 1, 1

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    # guarantee non-empty splits if possible
    if n_train <= 0:
        n_train = 1
    if n_val <= 0:
        n_val = 1
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        # take one from the largest of train/val
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1

    # final correction
    while n_train + n_val + n_test > n:
        if n_train >= n_val and n_train >= n_test and n_train > 1:
            n_train -= 1
        elif n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break

    while n_train + n_val + n_test < n:
        n_train += 1

    return n_train, n_val, n_test


# ============================================================
# Read dataset
# ============================================================

def discover_csv_files(input_dir: Path, explicit_files: Optional[List[str]] = None) -> List[Path]:
    if explicit_files:
        files = [(input_dir / f) for f in explicit_files if (input_dir / f).exists()]
    else:
        files = sorted(input_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    return files


def read_single_csv(csv_path: Path) -> pd.DataFrame:
    source_class = csv_path.stem.lower()
    df = pd.read_csv(csv_path)

    required_cols = {"timestamp", "arbitration_id", "data_field", "attack"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {sorted(missing)}")

    df = df[["timestamp", "arbitration_id", "data_field", "attack"]].copy()
    df["source_file"] = csv_path.name
    df["source_class"] = source_class
    df["orig_row_in_file"] = np.arange(len(df), dtype=np.int64)

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["arbitration_id"] = pd.to_numeric(df["arbitration_id"], errors="coerce").astype("Int64")
    df["attack"] = pd.to_numeric(df["attack"], errors="coerce").fillna(0).astype(int)
    df["data_field"] = df["data_field"].map(lambda x: ensure_hex_payload(x, expected_hex_len=16))

    df = df.sort_values(["timestamp", "orig_row_in_file"], kind="mergesort").reset_index(drop=True)

    df = df[df["timestamp"].notna()].copy()
    df = df[df["arbitration_id"].notna()].copy()
    df["arbitration_id"] = df["arbitration_id"].astype(np.int64)

    df["multiclass_label"] = np.where(df["attack"] == 0, NORMAL_LABEL, source_class)
    df["msg_idx_in_file"] = np.arange(len(df), dtype=np.int64)

    return df


def read_dataset(input_dir: Path, explicit_files: Optional[List[str]] = None) -> pd.DataFrame:
    csv_files = discover_csv_files(input_dir, explicit_files)
    dfs = []

    print(f"[Info] Reading CSV files from: {input_dir}")
    for csv_path in csv_files:
        df = read_single_csv(csv_path)
        dfs.append(df)
        print(f"[Loaded] {csv_path.name}: {len(df)} rows")

    full = pd.concat(dfs, ignore_index=True)
    full["global_msg_index"] = np.arange(len(full), dtype=np.int64)
    print(f"[Info] Total rows after concatenation: {len(full)}")
    return full


# ============================================================
# Feature engineering
# ============================================================

def add_payload_features(df: pd.DataFrame) -> pd.DataFrame:
    bytes_arr = np.array([hex_to_8bytes(x) for x in df["data_field"].tolist()], dtype=np.uint8)

    for i in range(8):
        df[f"byte_{i}"] = bytes_arr[:, i].astype(np.int64)
        df[f"byte_{i}_norm"] = bytes_arr[:, i].astype(np.float32) / 255.0

    return df


def add_timing_and_payload_diffs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["source_class", "msg_idx_in_file"], kind="mergesort").copy()

    df["delta_t_global"] = df.groupby("source_class")["timestamp"].diff().fillna(0.0)
    df["delta_t_global"] = df["delta_t_global"].clip(lower=0.0)

    delta_same = np.zeros(len(df), dtype=np.float64)
    l1_same = np.zeros(len(df), dtype=np.float64)
    ham_same = np.zeros(len(df), dtype=np.float64)

    prev_state: Dict[Tuple[str, int], Tuple[float, np.ndarray, np.ndarray]] = {}

    for pos, row in enumerate(df.itertuples(index=False)):
        key = (row.source_class, int(row.arbitration_id))

        cur_bytes = np.array([getattr(row, f"byte_{i}") for i in range(8)], dtype=np.uint8)
        cur_bits = bytes_to_bits(cur_bytes.tolist())

        if key in prev_state:
            prev_ts, prev_bytes, prev_bits = prev_state[key]

            dt = max(0.0, float(row.timestamp) - float(prev_ts))
            delta_same[pos] = dt
            l1_same[pos] = float(np.abs(cur_bytes.astype(np.int16) - prev_bytes.astype(np.int16)).sum())
            ham_same[pos] = float((cur_bits != prev_bits).sum())
        else:
            delta_same[pos] = 0.0
            l1_same[pos] = 0.0
            ham_same[pos] = 0.0

        prev_state[key] = (float(row.timestamp), cur_bytes.copy(), cur_bits.copy())

    df["delta_t_same_id"] = delta_same
    df["payload_l1_to_prev_same_id"] = l1_same
    df["payload_hamming_to_prev_same_id"] = ham_same

    df["log_delta_t_global"] = np.log1p(df["delta_t_global"].astype(np.float64))
    df["log_delta_t_same_id"] = np.log1p(df["delta_t_same_id"].astype(np.float64))

    return df


def add_label_indices(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    classes = sorted(df["multiclass_label"].unique().tolist())
    if NORMAL_LABEL in classes:
        classes = [NORMAL_LABEL] + [c for c in classes if c != NORMAL_LABEL]

    label_to_index = {label: idx for idx, label in enumerate(classes)}
    df = df.copy()
    df["multiclass_label_index"] = df["multiclass_label"].map(label_to_index).astype(np.int64)
    return df, label_to_index


# ============================================================
# Window creation
# ============================================================

def attack_aware_window_label(
    labels: List[str],
    window_size: int,
    min_attack_count: int,
    min_attack_ratio: float,
) -> Dict[str, object]:
    labels_arr = np.array(labels, dtype=object)
    attack_mask = labels_arr != NORMAL_LABEL
    attack_labels = labels_arr[attack_mask]

    attack_count = int(attack_mask.sum())
    attack_ratio = float(attack_count / max(1, window_size))

    if attack_count == 0:
        return {
            "window_label": NORMAL_LABEL,
            "attack_count": 0,
            "attack_ratio": 0.0,
            "dominant_attack_ratio_inside_attacks": 0.0,
            "is_mixed_window": False,
            "has_attack": False,
        }

    vals, counts = np.unique(attack_labels, return_counts=True)
    idx = int(np.argmax(counts))
    dominant_attack = str(vals[idx])
    dominant_attack_count = int(counts[idx])
    dominant_attack_ratio_inside_attacks = float(dominant_attack_count / attack_count)

    if attack_count < min_attack_count or attack_ratio < min_attack_ratio:
        label = NORMAL_LABEL
    else:
        label = dominant_attack

    return {
        "window_label": label,
        "attack_count": attack_count,
        "attack_ratio": attack_ratio,
        "dominant_attack_ratio_inside_attacks": dominant_attack_ratio_inside_attacks,
        "is_mixed_window": attack_count > 0 and attack_count < window_size,
        "has_attack": True,
    }


def build_candidate_windows(df_all: pd.DataFrame, window_cfg: WindowConfig) -> pd.DataFrame:
    rows = []

    effective_sampling_stride = (
        window_cfg.sampling_stride
        if window_cfg.sampling_stride > 0
        else max(window_cfg.window_size, window_cfg.stride)
    )

    print(f"[Info] Candidate window sampling stride = {effective_sampling_stride}")

    for source_class, g in df_all.groupby("source_class", sort=True):
        g = g.sort_values("msg_idx_in_file", kind="mergesort").reset_index(drop=True)

        n = len(g)
        W = window_cfg.window_size
        step = effective_sampling_stride

        if n < W:
            print(f"[Warn] {source_class}: n={n} < window_size={W}, skip.")
            continue

        window_order = 0
        for start in range(0, n - W + 1, step):
            end = start + W
            w = g.iloc[start:end].copy()

            label_info = attack_aware_window_label(
                labels=w["multiclass_label"].tolist(),
                window_size=W,
                min_attack_count=window_cfg.min_attack_count,
                min_attack_ratio=window_cfg.min_attack_ratio,
            )

            if not window_cfg.keep_mixed_windows and label_info["is_mixed_window"]:
                continue

            t0 = float(w["timestamp"].iloc[0])
            t1 = float(w["timestamp"].iloc[-1])
            duration = max(0.0, t1 - t0)
            msg_rate = float(len(w) / (duration + 1e-9))
            unique_ids = int(w["arbitration_id"].nunique())
            id_entropy = safe_entropy_from_counts(w["arbitration_id"].value_counts().to_numpy())

            row = {
                "source_class": source_class,
                "window_id": f"{source_class}__{start:08d}_{end:08d}",
                "window_order_in_file": int(window_order),
                "start_msg_idx_in_file": int(w["msg_idx_in_file"].iloc[0]),
                "end_msg_idx_in_file": int(w["msg_idx_in_file"].iloc[-1]),
                "start_global_msg_index": int(w["global_msg_index"].iloc[0]),
                "end_global_msg_index": int(w["global_msg_index"].iloc[-1]),
                "num_messages": int(len(w)),
                "t_start": t0,
                "t_end": t1,
                "window_duration": duration,
                "msg_rate": msg_rate,
                "unique_id_count": unique_ids,
                "id_entropy": id_entropy,
                "window_label": label_info["window_label"],
                "attack_count": int(label_info["attack_count"]),
                "attack_ratio": float(label_info["attack_ratio"]),
                "dominant_attack_ratio_inside_attacks": float(label_info["dominant_attack_ratio_inside_attacks"]),
                "is_mixed_window": bool(label_info["is_mixed_window"]),
                "has_attack": bool(label_info["has_attack"]),
            }
            rows.append(row)
            window_order += 1

    windows = pd.DataFrame(rows)
    return windows


# ============================================================
# Window-level split
# ============================================================

def split_windows_stratified(
    windows: pd.DataFrame,
    split_cfg: SplitConfig,
) -> pd.DataFrame:
    """
    Split windows at the window level.
    Stratify by (source_class, window_label) to preserve attack coverage.
    """
    rng = np.random.default_rng(split_cfg.seed)
    windows = windows.copy()
    windows["split"] = "unset"

    group_cols = ["source_class", "window_label"]

    stats_rows = []

    for keys, g in windows.groupby(group_cols, sort=True):
        idx = g.index.to_numpy()
        n = len(idx)

        shuffled = idx.copy()
        rng.shuffle(shuffled)

        n_train, n_val, n_test = deterministic_split_counts(
            n=n,
            train_ratio=split_cfg.train_ratio,
            val_ratio=split_cfg.val_ratio,
            test_ratio=split_cfg.test_ratio,
        )

        train_idx = shuffled[:n_train]
        val_idx = shuffled[n_train:n_train + n_val]
        test_idx = shuffled[n_train + n_val:n_train + n_val + n_test]

        windows.loc[train_idx, "split"] = "train"
        windows.loc[val_idx, "split"] = "val"
        windows.loc[test_idx, "split"] = "test"

        stats_rows.append(
            {
                "source_class": keys[0],
                "window_label": keys[1],
                "n_total": int(n),
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    print("[Info] Window-level split statistics:")
    if len(stats_df) > 0:
        print(stats_df.sort_values(["source_class", "window_label"]).to_string(index=False))

    return windows


# ============================================================
# Infer message splits from windows
# ============================================================

def infer_message_splits_from_windows(df_all: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """
    Assign message split based on selected windows.

    Because candidate windows are non-overlapping by construction (default safe setting),
    each message should belong to at most one chosen window.
    Messages not covered by any window remain 'unused'.
    """
    parts = []

    for source_class, g in df_all.groupby("source_class", sort=True):
        g = g.sort_values("msg_idx_in_file", kind="mergesort").copy().reset_index(drop=True)
        msg_split = np.array(["unused"] * len(g), dtype=object)

        wsub = windows[windows["source_class"] == source_class].copy()
        for row in wsub.itertuples(index=False):
            start = int(row.start_msg_idx_in_file)
            end = int(row.end_msg_idx_in_file)
            split = row.split
            msg_split[start:end + 1] = split

        g["split"] = msg_split
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    return out


# ============================================================
# Train-only stats
# ============================================================

def fit_train_statistics(df_train: pd.DataFrame) -> Dict[str, object]:
    stats: Dict[str, object] = {}

    unique_ids = sorted(df_train["arbitration_id"].astype(int).unique().tolist())
    id_to_index = {cid: idx for idx, cid in enumerate(unique_ids)}

    stats["id_to_index"] = id_to_index
    stats["num_ids"] = len(id_to_index)
    stats["id_min"] = int(min(unique_ids)) if unique_ids else 0
    stats["id_max"] = int(max(unique_ids)) if unique_ids else 1

    id_counts = df_train["arbitration_id"].value_counts().to_dict()
    total_train = max(1, len(df_train))
    id_freq_train = {int(k): float(v) / total_train for k, v in id_counts.items()}
    stats["id_freq_train"] = id_freq_train

    normal_train = df_train[df_train["multiclass_label"] == NORMAL_LABEL].copy()

    if len(normal_train) == 0:
        period_mean = {}
        period_std = {}
        global_period_mean = 0.0
        global_period_std = 1.0
    else:
        grp = normal_train.groupby("arbitration_id")["delta_t_same_id"]
        period_mean = grp.mean().to_dict()
        period_std = grp.std().fillna(0.0).to_dict()

        global_period_mean = float(normal_train["delta_t_same_id"].mean())
        global_period_std = float(normal_train["delta_t_same_id"].std()) if len(normal_train) > 1 else 1.0
        if global_period_std <= 1e-12:
            global_period_std = 1.0

    stats["periodicity_mean_normal_train"] = {int(k): float(v) for k, v in period_mean.items()}
    stats["periodicity_std_normal_train"] = {int(k): float(v) for k, v in period_std.items()}
    stats["global_period_mean"] = global_period_mean
    stats["global_period_std"] = global_period_std

    cont_cols = [
        "log_delta_t_global",
        "log_delta_t_same_id",
        "payload_l1_to_prev_same_id",
        "payload_hamming_to_prev_same_id",
    ]
    scale_params = {col: robust_stats(df_train[col]) for col in cont_cols}
    stats["robust_scale_params"] = scale_params

    return stats


def apply_train_statistics(df: pd.DataFrame, stats: Dict[str, object]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    id_to_index: Dict[int, int] = stats["id_to_index"]  # type: ignore[assignment]
    id_freq_train: Dict[int, float] = stats["id_freq_train"]  # type: ignore[assignment]
    period_mean: Dict[int, float] = stats["periodicity_mean_normal_train"]  # type: ignore[assignment]
    period_std: Dict[int, float] = stats["periodicity_std_normal_train"]  # type: ignore[assignment]
    scale_params: Dict[str, Dict[str, float]] = stats["robust_scale_params"]  # type: ignore[assignment]

    df = df.copy()

    unknown_index = int(stats["num_ids"])
    df["id_index"] = df["arbitration_id"].map(lambda x: id_to_index.get(int(x), unknown_index)).astype(np.int64)

    id_min = float(stats["id_min"])
    id_max = float(stats["id_max"])
    denom = max(1.0, id_max - id_min)
    df["id_norm"] = (df["arbitration_id"].astype(float) - id_min) / denom

    df["id_freq_train"] = df["arbitration_id"].map(lambda x: id_freq_train.get(int(x), 0.0)).astype(np.float64)
    df["id_rarity_score"] = -np.log(df["id_freq_train"] + 1e-12)

    global_period_mean = float(stats["global_period_mean"])
    global_period_std = max(1e-6, float(stats["global_period_std"]))

    mean_series = df["arbitration_id"].map(lambda x: period_mean.get(int(x), global_period_mean)).astype(np.float64)
    std_series = df["arbitration_id"].map(lambda x: max(period_std.get(int(x), global_period_std), 1e-6)).astype(np.float64)

    df["periodicity_residual"] = (df["delta_t_same_id"].astype(np.float64) - mean_series).abs()
    df["periodicity_residual_z"] = df["periodicity_residual"] / std_series
    df["log_periodicity_residual"] = np.log1p(df["periodicity_residual"])

    train_mask = df["split"] == "train"
    log_period_stats = robust_stats(df.loc[train_mask, "log_periodicity_residual"])
    stats["log_periodicity_residual_scale_params"] = log_period_stats

    for col, pars in scale_params.items():
        df[f"{col}_scaled"] = robust_scale(df[col].astype(np.float64), pars["median"], pars["iqr"])

    df["log_periodicity_residual_scaled"] = robust_scale(
        df["log_periodicity_residual"].astype(np.float64),
        log_period_stats["median"],
        log_period_stats["iqr"],
    )

    return df, stats


# ============================================================
# Summary
# ============================================================

def generate_summary(df_all: pd.DataFrame, windows: pd.DataFrame) -> Dict[str, object]:
    summary: Dict[str, object] = {}

    summary["num_messages_total"] = int(len(df_all))
    summary["num_messages_used"] = int((df_all["split"] != "unused").sum())
    summary["num_ids_total"] = int(df_all["arbitration_id"].nunique())
    summary["source_files"] = sorted(df_all["source_file"].unique().tolist())
    summary["source_classes"] = sorted(df_all["source_class"].unique().tolist())

    summary["classes_message_level_full"] = df_all["multiclass_label"].value_counts().to_dict()
    summary["classes_message_level_used"] = df_all[df_all["split"] != "unused"]["multiclass_label"].value_counts().to_dict()
    summary["classes_by_split_message_level"] = {
        split: sub["multiclass_label"].value_counts().to_dict()
        for split, sub in df_all[df_all["split"] != "unused"].groupby("split")
    }

    summary["windows"] = {}
    for split, wdf in windows.groupby("split"):
        summary["windows"][split] = {
            "num_windows": int(len(wdf)),
            "window_labels": wdf["window_label"].value_counts().to_dict(),
            "windows_with_attack": int((wdf["has_attack"] == True).sum()),
            "mixed_windows": int((wdf["is_mixed_window"] == True).sum()),
            "attack_count_stats": {
                "min": int(wdf["attack_count"].min()),
                "max": int(wdf["attack_count"].max()),
                "mean": float(wdf["attack_count"].mean()),
            },
            "attack_ratio_stats": {
                "min": float(wdf["attack_ratio"].min()),
                "max": float(wdf["attack_ratio"].max()),
                "mean": float(wdf["attack_ratio"].mean()),
            },
        }

    return summary


# ============================================================
# Main pipeline
# ============================================================

def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    explicit_files: Optional[List[str]],
    split_cfg: SplitConfig,
    window_cfg: WindowConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Read full dataset
    df = read_dataset(input_dir=input_dir, explicit_files=explicit_files)

    # 2) Raw feature engineering on all messages
    df = add_payload_features(df)
    df = add_timing_and_payload_diffs(df)
    df, label_to_index = add_label_indices(df)

    # 3) Build candidate windows first
    windows = build_candidate_windows(df, window_cfg=window_cfg)
    if len(windows) == 0:
        raise ValueError("No windows were created. Check window_size / sampling_stride.")

    # attach label indices after window labels exist
    windows["window_label_index"] = windows["window_label"].map(label_to_index).astype(np.int64)

    # 4) Split at window level
    windows = split_windows_stratified(windows, split_cfg=split_cfg)

    # 5) Infer message split membership from windows
    df = infer_message_splits_from_windows(df, windows)

    # keep only messages used by some window
    df_used = df[df["split"] != "unused"].copy().reset_index(drop=True)

    # 6) Fit train-only statistics on messages covered by train windows
    df_train = df_used[df_used["split"] == "train"].copy()
    if len(df_train) == 0:
        raise ValueError("Train messages are empty after window-level split.")
    stats = fit_train_statistics(df_train)

    # 7) Apply train-only statistics
    df_used, stats = apply_train_statistics(df_used, stats)

    # 8) Save outputs
    save_dataframe(df_used, output_dir / "messages_used")
    save_dataframe(df_used[df_used["split"] == "train"].reset_index(drop=True), output_dir / "messages_train")
    save_dataframe(df_used[df_used["split"] == "val"].reset_index(drop=True), output_dir / "messages_val")
    save_dataframe(df_used[df_used["split"] == "test"].reset_index(drop=True), output_dir / "messages_test")

    save_dataframe(windows[windows["split"] == "train"].reset_index(drop=True), output_dir / "windows_train")
    save_dataframe(windows[windows["split"] == "val"].reset_index(drop=True), output_dir / "windows_val")
    save_dataframe(windows[windows["split"] == "test"].reset_index(drop=True), output_dir / "windows_test")
    save_dataframe(windows.reset_index(drop=True), output_dir / "windows_all")

    metadata = {
        "split_config": asdict(split_cfg),
        "window_config": asdict(window_cfg),
        "label_to_index": label_to_index,
        "index_to_label": {str(v): k for k, v in label_to_index.items()},
        "train_stats": stats,
    }
    save_json(metadata, output_dir / "preprocessing_metadata.json")

    summary = generate_summary(df_used, windows)
    save_json(summary, output_dir / "summary.json")

    print("\n[Done] Preprocessing finished.")
    print(f"Input dir : {input_dir}")
    print(f"Output dir: {output_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Window-first preprocessing for IVN IDS dynamic graph learning."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing CSV files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed outputs.")
    parser.add_argument(
        "--files",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of csv filenames. Example: dos.csv rpm.csv combined.csv",
    )

    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument(
        "--sampling_stride",
        type=int,
        default=0,
        help="Stride used to generate candidate windows for splitting. "
             "0 means auto=max(window_size, stride) to reduce overlap leakage.",
    )

    parser.add_argument("--min_attack_count", type=int, default=1)
    parser.add_argument("--min_attack_ratio", type=float, default=0.0)
    parser.add_argument(
        "--drop_mixed_windows",
        action="store_true",
        help="If set, windows containing both normal and attack messages will be dropped.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {ratio_sum}")

    split_cfg = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    window_cfg = WindowConfig(
        window_size=args.window_size,
        stride=args.stride,
        sampling_stride=args.sampling_stride,
        min_attack_count=args.min_attack_count,
        min_attack_ratio=args.min_attack_ratio,
        keep_mixed_windows=not args.drop_mixed_windows,
    )

    run_pipeline(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        explicit_files=args.files,
        split_cfg=split_cfg,
        window_cfg=window_cfg,
    )


if __name__ == "__main__":
    main()