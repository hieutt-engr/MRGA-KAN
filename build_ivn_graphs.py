#!/usr/bin/env python3
"""
build_ivn_graphs.py

Build single-window dynamic multi-relational message graphs from
preprocessed parquet files.

Input:
- messages_train.parquet / messages_val.parquet / messages_test.parquet
- windows_train.parquet / windows_val.parquet / windows_test.parquet
- preprocessing_metadata.json

Output:
- graphs_train_shard00000.pt, ...
- graphs_val_shard00000.pt, ...
- graphs_test_shard00000.pt, ...
- graph_index_train.parquet / val / test
- graph_summary.json
- graph_metadata.json

Graph definition:
- one graph per window
- node = one CAN message
- relations:
    0: temporal adjacency
    1: same-ID recurrence
    2: payload similarity
    3: timing affinity

Each saved graph is a Python dict with:
{
    "graph_id": str,
    "split": str,
    "source_class": str,
    "window_label": str,
    "y": LongTensor[1],                  # graph label index
    "x": FloatTensor[N, F],              # continuous node features
    "id_index": LongTensor[N],           # ID embedding indices
    "arbitration_id": LongTensor[N],
    "timestamp": FloatTensor[N],
    "msg_idx_in_file": LongTensor[N],
    "edge_index": LongTensor[2, E],
    "edge_type": LongTensor[E],
    "edge_attr": FloatTensor[E, D],
    "meta": {...}
}

Recommended usage:
python build_ivn_graphs.py \
  --input_dir ./data/2017-subaru-forester/preprocessed_v3 \
  --output_dir ./data/2017-subaru-forester/graphs_v1 \
  --graphs_per_shard 2000 \
  --temporal_k 1 \
  --same_id_k 1 \
  --payload_topk 1 \
  --timing_topk 1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch


# ------------------------------------------------------------
# Relation mapping
# ------------------------------------------------------------

RELATION_TO_INDEX = {
    "temporal": 0,
    "same_id": 1,
    "payload_sim": 2,
    "timing_aff": 3,
}

INDEX_TO_RELATION = {v: k for k, v in RELATION_TO_INDEX.items()}


# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------

def load_table(base: Path) -> pd.DataFrame:
    parquet_path = base.with_suffix(".parquet")
    csv_path = base.with_suffix(".csv")

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(f"Neither {parquet_path.name} nor {csv_path.name} exists.")


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {path}")


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


# ------------------------------------------------------------
# Feature config
# ------------------------------------------------------------

def get_default_node_feature_cols() -> List[str]:
    return [
        "id_norm",
        "byte_0_norm",
        "byte_1_norm",
        "byte_2_norm",
        "byte_3_norm",
        "byte_4_norm",
        "byte_5_norm",
        "byte_6_norm",
        "byte_7_norm",
        "log_delta_t_global_scaled",
        "log_delta_t_same_id_scaled",
        "payload_l1_to_prev_same_id_scaled",
        "payload_hamming_to_prev_same_id_scaled",
        "id_freq_train",
        "id_rarity_score",
        "log_periodicity_residual_scaled",
    ]


def validate_columns(df: pd.DataFrame, required_cols: List[str], table_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {table_name}: {missing}")


# ------------------------------------------------------------
# Edge attribute builder
# ------------------------------------------------------------

def build_edge_attr(
    i: int,
    j: int,
    timestamps: np.ndarray,
    arbitration_ids: np.ndarray,
    bytes_norm: np.ndarray,      # [N, 8] float
    bits_uint8: np.ndarray,      # [N, 64] uint8
    timing_scalar: np.ndarray,   # [N]
) -> List[float]:
    """
    Edge attribute vector:
    [0] log_dt_abs
    [1] same_id_flag
    [2] payload_l1_norm     in [0,1]
    [3] payload_hamming_norm in [0,1]
    [4] index_gap_norm      in [0,1]
    [5] timing_diff_abs
    """
    dt_abs = abs(float(timestamps[i]) - float(timestamps[j]))
    log_dt_abs = float(np.log1p(dt_abs))

    same_id_flag = 1.0 if int(arbitration_ids[i]) == int(arbitration_ids[j]) else 0.0

    payload_l1 = float(np.abs(bytes_norm[i] - bytes_norm[j]).sum() / 8.0)   # normalized to [0,1]
    payload_ham = float((bits_uint8[i] != bits_uint8[j]).sum() / 64.0)

    # index gap normalized using simple formula robust for small N
    # actual N-dependent normalization is done later if needed; here keep raw bounded style
    index_gap = abs(i - j)
    index_gap_norm = float(index_gap / max(1, len(timestamps) - 1))

    timing_diff_abs = float(abs(float(timing_scalar[i]) - float(timing_scalar[j])))

    return [
        log_dt_abs,
        same_id_flag,
        payload_l1,
        payload_ham,
        index_gap_norm,
        timing_diff_abs,
    ]


# ------------------------------------------------------------
# Graph builder for one window
# ------------------------------------------------------------

def subsample_windows(
    windows: pd.DataFrame,
    split_name: str,
    seed: int = 42,
    mode: str = "keep_all_attacks_downsample_normal",
    frac: float = 0.2,
    min_per_class: int = 200,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []

    if mode == "keep_all_attacks_downsample_normal":
        for label, g in windows.groupby("window_label", sort=True):
            if label == "normal":
                n = len(g)
                k = max(min_per_class, int(round(n * frac)))
                k = min(k, n)
                idx = rng.choice(g.index.to_numpy(), size=k, replace=False)
                parts.append(g.loc[idx].copy())
            else:
                parts.append(g.copy())

    elif mode == "stratified_fraction":
        for label, g in windows.groupby("window_label", sort=True):
            n = len(g)
            k = max(min_per_class, int(round(n * frac)))
            k = min(k, n)
            idx = rng.choice(g.index.to_numpy(), size=k, replace=False)
            parts.append(g.loc[idx].copy())

    else:
        raise ValueError(f"Unknown subsample mode: {mode}")

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["source_class", "start_msg_idx_in_file"], kind="mergesort").reset_index(drop=True)

    print(f"[Info] Subsampled windows for {split_name}: {len(windows)} -> {len(out)}")
    print(out["window_label"].value_counts())

    return out

def build_graph_for_window(
    msg_window: pd.DataFrame,
    window_row: pd.Series,
    node_feature_cols: List[str],
    temporal_k: int = 1,
    same_id_k: int = 1,
    payload_topk: int = 1,
    timing_topk: int = 1,
) -> Dict[str, Any]:
    """
    Build one graph dict from one window.
    """
    msg_window = msg_window.sort_values("msg_idx_in_file", kind="mergesort").reset_index(drop=True)
    N = len(msg_window)
    if N <= 1:
        raise ValueError(f"Window {window_row['window_id']} has too few nodes: N={N}")

    # ---------- Node tensors ----------
    x = torch.tensor(msg_window[node_feature_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
    id_index = torch.tensor(msg_window["id_index"].to_numpy(dtype=np.int64), dtype=torch.long)
    arbitration_id = torch.tensor(msg_window["arbitration_id"].to_numpy(dtype=np.int64), dtype=torch.long)
    timestamp = torch.tensor(msg_window["timestamp"].to_numpy(dtype=np.float64), dtype=torch.float32)
    msg_idx_in_file = torch.tensor(msg_window["msg_idx_in_file"].to_numpy(dtype=np.int64), dtype=torch.long)

    # numpy helpers
    timestamps_np = msg_window["timestamp"].to_numpy(dtype=np.float64)
    arbitration_ids_np = msg_window["arbitration_id"].to_numpy(dtype=np.int64)
    bytes_norm_np = msg_window[[f"byte_{i}_norm" for i in range(8)]].to_numpy(dtype=np.float32)
    timing_scalar_np = msg_window["log_delta_t_same_id_scaled"].to_numpy(dtype=np.float32)

    bits_uint8 = []
    for r in msg_window.itertuples(index=False):
        b = [int(getattr(r, f"byte_{i}")) for i in range(8)]
        bits_uint8.append(np.unpackbits(np.array(b, dtype=np.uint8)))
    bits_uint8 = np.stack(bits_uint8, axis=0).astype(np.uint8)

    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_type: List[int] = []
    edge_attr: List[List[float]] = []

    def add_edge(i: int, j: int, rel_name: str) -> None:
        edge_src.append(i)
        edge_dst.append(j)
        edge_type.append(RELATION_TO_INDEX[rel_name])
        edge_attr.append(
            build_edge_attr(
                i=i,
                j=j,
                timestamps=timestamps_np,
                arbitration_ids=arbitration_ids_np,
                bytes_norm=bytes_norm_np,
                bits_uint8=bits_uint8,
                timing_scalar=timing_scalar_np,
            )
        )

    # --------------------------------------------------------
    # Relation 1: temporal adjacency
    # --------------------------------------------------------
    for i in range(N):
        for step in range(1, temporal_k + 1):
            j = i + step
            if j < N:
                add_edge(i, j, "temporal")
                add_edge(j, i, "temporal")

    # --------------------------------------------------------
    # Relation 2: same-ID recurrence
    # connect consecutive same-ID nodes within window
    # --------------------------------------------------------
    id_to_positions: Dict[int, List[int]] = defaultdict(list)
    for idx, cid in enumerate(arbitration_ids_np.tolist()):
        id_to_positions[int(cid)].append(idx)

    for cid, positions in id_to_positions.items():
        m = len(positions)
        if m <= 1:
            continue

        for p_idx, i in enumerate(positions):
            for step in range(1, same_id_k + 1):
                q_idx = p_idx + step
                if q_idx < m:
                    j = positions[q_idx]
                    add_edge(i, j, "same_id")
                    add_edge(j, i, "same_id")

    # --------------------------------------------------------
    # Relation 3: payload similarity
    # connect top-k closest nodes by payload distance
    # restricted to same-ID groups to keep semantics tighter
    # --------------------------------------------------------
    if payload_topk > 0:
        for cid, positions in id_to_positions.items():
            if len(positions) <= 1:
                continue

            group_bytes = bytes_norm_np[positions]  # [M, 8]
            # pairwise L1
            dmat = np.abs(group_bytes[:, None, :] - group_bytes[None, :, :]).sum(axis=-1)  # [M, M]
            np.fill_diagonal(dmat, np.inf)

            for local_i, i in enumerate(positions):
                k = min(payload_topk, len(positions) - 1)
                if k <= 0:
                    continue
                nn_local = np.argpartition(dmat[local_i], kth=k - 1)[:k]
                for local_j in nn_local:
                    j = positions[int(local_j)]
                    add_edge(i, j, "payload_sim")

    # --------------------------------------------------------
    # Relation 4: timing affinity
    # connect top-k closest nodes globally by timing scalar
    # --------------------------------------------------------
    if timing_topk > 0:
        ts = timing_scalar_np.reshape(-1, 1)
        dmat = np.abs(ts - ts.T)
        np.fill_diagonal(dmat, np.inf)

        for i in range(N):
            k = min(timing_topk, N - 1)
            if k <= 0:
                continue
            nn = np.argpartition(dmat[i], kth=k - 1)[:k]
            for j in nn.tolist():
                add_edge(i, int(j), "timing_aff")

    # final tensors
    edge_index = torch.tensor(np.array([edge_src, edge_dst], dtype=np.int64), dtype=torch.long)
    edge_type_t = torch.tensor(np.array(edge_type, dtype=np.int64), dtype=torch.long)
    edge_attr_t = torch.tensor(np.array(edge_attr, dtype=np.float32), dtype=torch.float32)

    y = torch.tensor([int(window_row["window_label_index"])], dtype=torch.long)

    graph = {
        "graph_id": str(window_row["window_id"]),
        "split": str(window_row["split"]),
        "source_class": str(window_row["source_class"]),
        "window_label": str(window_row["window_label"]),
        "y": y,
        "x": x,
        "id_index": id_index,
        "arbitration_id": arbitration_id,
        "timestamp": timestamp,
        "msg_idx_in_file": msg_idx_in_file,
        "edge_index": edge_index,
        "edge_type": edge_type_t,
        "edge_attr": edge_attr_t,
        "meta": {
            "start_msg_idx_in_file": int(window_row["start_msg_idx_in_file"]),
            "end_msg_idx_in_file": int(window_row["end_msg_idx_in_file"]),
            "num_messages": int(window_row["num_messages"]),
            "t_start": float(window_row["t_start"]),
            "t_end": float(window_row["t_end"]),
            "window_duration": float(window_row["window_duration"]),
            "msg_rate": float(window_row["msg_rate"]),
            "unique_id_count": int(window_row["unique_id_count"]),
            "id_entropy": float(window_row["id_entropy"]),
            "attack_count": int(window_row["attack_count"]),
            "attack_ratio": float(window_row["attack_ratio"]),
            "dominant_attack_ratio_inside_attacks": float(window_row["dominant_attack_ratio_inside_attacks"]),
            "is_mixed_window": bool(window_row["is_mixed_window"]),
            "has_attack": bool(window_row["has_attack"]),
            "window_order_in_file": int(window_row["window_order_in_file"]) if "window_order_in_file" in window_row else -1,
        },
    }

    return graph


# ------------------------------------------------------------
# Split processor
# ------------------------------------------------------------

def process_split(
    split_name: str,
    input_dir: Path,
    output_dir: Path,
    node_feature_cols: List[str],
    graphs_per_shard: int,
    temporal_k: int,
    same_id_k: int,
    payload_topk: int,
    timing_topk: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    print(f"\n[Process split] {split_name}")

    if split_name == "all":
        messages = load_table(input_dir / "messages_used")
        windows = load_table(input_dir / "windows_all")
    else:
        messages = load_table(input_dir / f"messages_{split_name}")
        windows = load_table(input_dir / f"windows_{split_name}")

        windows = load_table(input_dir / f"windows_{split_name}")
        #Subsample windows if needed
        windows = subsample_windows(
            windows=windows,
            split_name=split_name,
            seed=args.subsample_seed,
            mode=args.subsample_mode,
            frac=args.subsample_frac,
            min_per_class=args.subsample_min_per_class,
        )

    validate_columns(
        messages,
        required_cols=node_feature_cols + [
            "id_index",
            "arbitration_id",
            "timestamp",
            "msg_idx_in_file",
            "log_delta_t_same_id_scaled",
            "byte_0", "byte_1", "byte_2", "byte_3",
            "byte_4", "byte_5", "byte_6", "byte_7",
            "byte_0_norm", "byte_1_norm", "byte_2_norm", "byte_3_norm",
            "byte_4_norm", "byte_5_norm", "byte_6_norm", "byte_7_norm",
        ],
        table_name=f"messages_{split_name}",
    )
    validate_columns(
        windows,
        required_cols=[
            "window_id",
            "source_class",
            "split",
            "window_label",
            "window_label_index",
            "start_msg_idx_in_file",
            "end_msg_idx_in_file",
            "num_messages",
            "t_start",
            "t_end",
            "window_duration",
            "msg_rate",
            "unique_id_count",
            "id_entropy",
            "attack_count",
            "attack_ratio",
            "dominant_attack_ratio_inside_attacks",
            "is_mixed_window",
            "has_attack",
        ],
        table_name=f"windows_{split_name}",
    )

    # index message tables by source_class and msg_idx_in_file
    source_tables: Dict[str, pd.DataFrame] = {}
    for source_class, g in messages.groupby("source_class", sort=True):
        gg = g.sort_values("msg_idx_in_file", kind="mergesort").copy()
        gg = gg.set_index("msg_idx_in_file", drop=False)
        source_tables[source_class] = gg

    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    graph_records = []
    shard_graphs: List[Dict[str, Any]] = []
    shard_id = 0

    label_counter = Counter()
    total_nodes = 0
    total_edges = 0
    total_graphs = 0
    relation_counter = Counter()

    windows = windows.sort_values(["source_class", "start_msg_idx_in_file"], kind="mergesort").reset_index(drop=True)

    for row_idx, w in enumerate(windows.itertuples(index=False), start=1):
        source_class = str(w.source_class)
        start_idx = int(w.start_msg_idx_in_file)
        end_idx = int(w.end_msg_idx_in_file)

        if source_class not in source_tables:
            raise KeyError(f"source_class={source_class} not found in messages_{split_name}")

        msg_source = source_tables[source_class]

        # slice by file-local message index
        sub = msg_source.loc[start_idx:end_idx].copy()
        if len(sub) != int(w.num_messages):
            raise ValueError(
                f"Window {w.window_id}: expected {int(w.num_messages)} messages, got {len(sub)}"
            )

        graph = build_graph_for_window(
            msg_window=sub.reset_index(drop=True),
            window_row=pd.Series(w._asdict()),
            node_feature_cols=node_feature_cols,
            temporal_k=temporal_k,
            same_id_k=same_id_k,
            payload_topk=payload_topk,
            timing_topk=timing_topk,
        )

        shard_graphs.append(graph)

        n_nodes = int(graph["x"].shape[0])
        n_edges = int(graph["edge_index"].shape[1])

        total_graphs += 1
        total_nodes += n_nodes
        total_edges += n_edges
        label_counter[graph["window_label"]] += 1

        etypes = graph["edge_type"].tolist()
        for et in etypes:
            relation_counter[INDEX_TO_RELATION[int(et)]] += 1

        graph_records.append(
            {
                "graph_id": graph["graph_id"],
                "split": graph["split"],
                "source_class": graph["source_class"],
                "window_label": graph["window_label"],
                "y": int(graph["y"].item()),
                "num_nodes": n_nodes,
                "num_edges": n_edges,
                "attack_count": graph["meta"]["attack_count"],
                "attack_ratio": graph["meta"]["attack_ratio"],
                "is_mixed_window": graph["meta"]["is_mixed_window"],
                "has_attack": graph["meta"]["has_attack"],
                "start_msg_idx_in_file": graph["meta"]["start_msg_idx_in_file"],
                "end_msg_idx_in_file": graph["meta"]["end_msg_idx_in_file"],
                "t_start": graph["meta"]["t_start"],
                "t_end": graph["meta"]["t_end"],
                "shard_id": shard_id,
            }
        )

        if len(shard_graphs) >= graphs_per_shard:
            shard_path = split_dir / f"graphs_{split_name}_shard{shard_id:05d}.pt"
            torch.save(shard_graphs, shard_path)
            print(f"[Saved] {shard_path} | graphs={len(shard_graphs)}")
            shard_graphs = []
            shard_id += 1

        if row_idx % 5000 == 0:
            print(f"[{split_name}] processed {row_idx}/{len(windows)} windows")

    # save remaining graphs
    if shard_graphs:
        shard_path = split_dir / f"graphs_{split_name}_shard{shard_id:05d}.pt"
        torch.save(shard_graphs, shard_path)
        print(f"[Saved] {shard_path} | graphs={len(shard_graphs)}")
        shard_id += 1

    graph_index = pd.DataFrame(graph_records)
    save_dataframe(graph_index, output_dir / f"graph_index_{split_name}")

    split_summary = {
        "num_graphs": total_graphs,
        "num_shards": shard_id,
        "label_counts": dict(label_counter),
        "avg_num_nodes": float(total_nodes / max(1, total_graphs)),
        "avg_num_edges": float(total_edges / max(1, total_graphs)),
        "edge_relation_counts": dict(relation_counter),
    }

    return split_summary


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build window-level graphs for IVN IDS.")

    parser.add_argument("--input_dir", type=str, required=True, help="Directory from preprocess_v3 outputs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save graph .pt shards.")

    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=["train", "val", "test"],
        help="Splits to process. Default: train val test",
    )

    parser.add_argument("--graphs_per_shard", type=int, default=2000)

    parser.add_argument("--temporal_k", type=int, default=1)
    parser.add_argument("--same_id_k", type=int, default=1)
    parser.add_argument("--payload_topk", type=int, default=1)
    parser.add_argument("--timing_topk", type=int, default=1)

    #subsampling options
    parser.add_argument("--subsample_mode", type=str, default="keep_all_attacks_downsample_normal",
                    choices=["keep_all_attacks_downsample_normal", "stratified_fraction"])
    parser.add_argument("--subsample_frac", type=float, default=0.2)
    parser.add_argument("--subsample_min_per_class", type=int, default=200)
    parser.add_argument("--subsample_seed", type=int, default=42)

    parser.add_argument(
        "--node_feature_cols",
        type=str,
        nargs="*",
        default=None,
        help="Optional override for node feature columns.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    node_feature_cols = args.node_feature_cols if args.node_feature_cols else get_default_node_feature_cols()

    metadata = {
        "relation_to_index": RELATION_TO_INDEX,
        "index_to_relation": INDEX_TO_RELATION,
        "node_feature_cols": node_feature_cols,
        "graphs_per_shard": args.graphs_per_shard,
        "temporal_k": args.temporal_k,
        "same_id_k": args.same_id_k,
        "payload_topk": args.payload_topk,
        "timing_topk": args.timing_topk,
    }
    save_json(metadata, output_dir / "graph_metadata.json")

    all_summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "splits": {},
    }

    for split_name in args.splits:
        split_summary = process_split(
            split_name=split_name,
            input_dir=input_dir,
            output_dir=output_dir,
            node_feature_cols=node_feature_cols,
            graphs_per_shard=args.graphs_per_shard,
            temporal_k=args.temporal_k,
            same_id_k=args.same_id_k,
            payload_topk=args.payload_topk,
            timing_topk=args.timing_topk,
            args=args,
        )
        all_summary["splits"][split_name] = split_summary

    save_json(all_summary, output_dir / "graph_summary.json")
    print("\n[Done] Graph construction finished.")
    print(json.dumps(all_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()