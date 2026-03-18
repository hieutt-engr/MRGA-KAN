import os
import json
import math
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch


NORMAL_LABEL = "normal"


# ============================================================
# IO helpers
# ============================================================

def load_table(base: Path) -> pd.DataFrame:
    pq = base.with_suffix(".parquet")
    csv = base.with_suffix(".csv")

    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)

    raise FileNotFoundError("Cannot find {} or {}".format(pq, csv))


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print("[Saved] {}".format(path))


# ============================================================
# Split helpers
# ============================================================

def deterministic_split_counts(n, train_ratio, val_ratio, test_ratio):
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

    if n_train <= 0:
        n_train = 1
    if n_val <= 0:
        n_val = 1
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1

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
# Chunk/sequence labeling
# ============================================================

def dominant_attack_aware_label(labels):
    """
    If any non-normal labels exist:
      return dominant non-normal label
    else:
      return normal
    """
    labels = list(labels)
    attack_labels = [x for x in labels if x != NORMAL_LABEL]

    if len(attack_labels) == 0:
        return NORMAL_LABEL

    c = Counter(attack_labels)
    return c.most_common(1)[0][0]


# ============================================================
# Graph loading cache
# ============================================================

class GraphShardCache(object):
    def __init__(self, graph_folder: Path, split_name: str):
        self.graph_folder = graph_folder
        self.split_name = split_name
        self.cache = {}

    def load_shard(self, shard_id: int):
        if shard_id not in self.cache:
            shard_path = self.graph_folder / self.split_name / "graphs_{}_shard{:05d}.pt".format(
                self.split_name, shard_id
            )
            self.cache[shard_id] = torch.load(shard_path, map_location="cpu", weights_only=False)
        return self.cache[shard_id]

    def get_graph_by_id(self, shard_id: int, graph_id: str):
        graphs = self.load_shard(shard_id)
        for g in graphs:
            if g["graph_id"] == graph_id:
                return g
        raise KeyError("Graph {} not found in shard {}".format(graph_id, shard_id))


# ============================================================
# Main builder logic
# ============================================================

def infer_graph_stride(graph_index: pd.DataFrame) -> int:
    """
    Infer expected graph stride from positive differences of start_msg_idx_in_file.
    """
    diffs = []
    for source_class, g in graph_index.groupby("source_class", sort=True):
        g = g.sort_values("start_msg_idx_in_file", kind="mergesort")
        d = g["start_msg_idx_in_file"].diff().dropna()
        d = d[d > 0]
        diffs.extend(d.tolist())

    if len(diffs) == 0:
        raise ValueError("Could not infer graph stride from graph_index_all.")

    stride = int(np.median(diffs))
    return stride


def assign_contiguous_runs(graph_index: pd.DataFrame, graph_stride: int) -> pd.DataFrame:
    """
    A new run starts if start_msg_idx_in_file does not increase exactly by graph_stride.
    """
    parts = []

    for source_class, g in graph_index.groupby("source_class", sort=True):
        g = g.sort_values("start_msg_idx_in_file", kind="mergesort").copy().reset_index(drop=True)

        diffs = g["start_msg_idx_in_file"].diff()
        new_run = diffs.ne(graph_stride)
        new_run.iloc[0] = True
        g["graph_run_id"] = new_run.cumsum().astype(np.int64)

        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    return out


def build_chunk_specs(
    graph_index: pd.DataFrame,
    chunk_len_graphs: int,
    chunk_step_graphs: int,
    min_chunk_graphs: int,
) -> pd.DataFrame:
    """
    Build chunk specs from contiguous graph runs.
    """
    rows = []

    for (source_class, run_id), g in graph_index.groupby(["source_class", "graph_run_id"], sort=True):
        g = g.sort_values("start_msg_idx_in_file", kind="mergesort").reset_index(drop=True)
        n = len(g)

        if n < min_chunk_graphs:
            continue

        chunk_id_local = 0
        start = 0
        while start < n:
            end = start + chunk_len_graphs
            if end > n:
                # tail chunk
                if (n - start) < min_chunk_graphs:
                    break
                end = n

            sub = g.iloc[start:end].copy()
            labels = sub["window_label"].tolist()
            chunk_label = dominant_attack_aware_label(labels)

            rows.append({
                "source_class": source_class,
                "graph_run_id": int(run_id),
                "chunk_id": "{}__run{:08d}__chunk{:08d}".format(source_class, int(run_id), chunk_id_local),
                "chunk_order_in_run": int(chunk_id_local),
                "chunk_label": chunk_label,
                "num_graphs": int(len(sub)),
                "start_graph_pos_in_run": int(start),
                "end_graph_pos_in_run": int(end - 1),
                "start_msg_idx_in_file": int(sub["start_msg_idx_in_file"].iloc[0]),
                "end_msg_idx_in_file": int(sub["end_msg_idx_in_file"].iloc[-1]),
                "graph_ids": sub["graph_id"].tolist(),
                "shard_ids": sub["shard_id"].tolist(),
                "window_labels": sub["window_label"].tolist(),
                "y_last": int(sub["y"].iloc[-1]),
            })

            chunk_id_local += 1
            start += chunk_step_graphs

    chunks = pd.DataFrame(rows)
    return chunks


def stratified_split_chunks(
    chunks: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """
    Split at chunk level, stratified by (source_class, chunk_label).
    """
    rng = np.random.default_rng(seed)
    chunks = chunks.copy()
    chunks["split"] = "unset"

    stats_rows = []

    for (source_class, chunk_label), g in chunks.groupby(["source_class", "chunk_label"], sort=True):
        idx = g.index.to_numpy()
        shuffled = idx.copy()
        rng.shuffle(shuffled)

        n_train, n_val, n_test = deterministic_split_counts(
            len(shuffled), train_ratio, val_ratio, test_ratio
        )

        tr_idx = shuffled[:n_train]
        va_idx = shuffled[n_train:n_train + n_val]
        te_idx = shuffled[n_train + n_val:n_train + n_val + n_test]

        chunks.loc[tr_idx, "split"] = "train"
        chunks.loc[va_idx, "split"] = "val"
        chunks.loc[te_idx, "split"] = "test"

        stats_rows.append({
            "source_class": source_class,
            "chunk_label": chunk_label,
            "n_total": int(len(shuffled)),
            "train": int(len(tr_idx)),
            "val": int(len(va_idx)),
            "test": int(len(te_idx)),
        })

    stats_df = pd.DataFrame(stats_rows)
    print("[Chunk split statistics]")
    if len(stats_df) > 0:
        print(stats_df.sort_values(["source_class", "chunk_label"]).to_string(index=False))

    return chunks


def build_sequences_from_chunks(
    chunks: pd.DataFrame,
    graph_folder: Path,
    split_name: str,
    seq_len: int,
    seq_stride: int,
    label_mode: str = "last",
):
    """
    Materialize full graph sequences from chunk specs.
    """
    cache = GraphShardCache(graph_folder=graph_folder, split_name="all")
    sequences = []

    sub_chunks = chunks[chunks["split"] == split_name].copy()
    sub_chunks = sub_chunks.sort_values(["source_class", "start_msg_idx_in_file"], kind="mergesort").reset_index(drop=True)

    for row in sub_chunks.itertuples(index=False):
        graph_ids = list(row.graph_ids)
        shard_ids = list(row.shard_ids)
        window_labels = list(row.window_labels)

        n = len(graph_ids)
        if n < seq_len:
            continue

        for start in range(0, n - seq_len + 1, seq_stride):
            end = start + seq_len

            seq_graph_ids = graph_ids[start:end]
            seq_shard_ids = shard_ids[start:end]
            seq_window_labels = window_labels[start:end]

            graphs = []
            for gid, sid in zip(seq_graph_ids, seq_shard_ids):
                g = cache.get_graph_by_id(shard_id=int(sid), graph_id=str(gid))
                graphs.append(g)

            if label_mode == "last":
                y = int(graphs[-1]["y"].view(-1)[0].item())
            elif label_mode == "majority":
                ys = [int(g["y"].view(-1)[0].item()) for g in graphs]
                y = Counter(ys).most_common(1)[0][0]
            else:
                raise ValueError("Unsupported label_mode: {}".format(label_mode))

            seq_item = {
                "sequence_id": "{}__{}__seq{:08d}".format(split_name, row.chunk_id, start),
                "split": split_name,
                "graphs": graphs,
                "y": y,
                "graph_ids": seq_graph_ids,
                "graph_labels": seq_window_labels,
                "meta": {
                    "source_class": row.source_class,
                    "seq_len": seq_len,
                    "graph_run_id": int(row.graph_run_id),
                    "chunk_id": row.chunk_id,
                    "chunk_label": row.chunk_label,
                    "chunk_num_graphs": int(row.num_graphs),
                    "start_msg_idx_in_file": int(row.start_msg_idx_in_file),
                    "end_msg_idx_in_file": int(row.end_msg_idx_in_file),
                },
            }
            sequences.append(seq_item)

    return sequences


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser("Build clean sequence-of-graphs dataset for temporal transformer")

    parser.add_argument("--graph_folder", type=str, required=True, help="Folder containing graph_index_all and /all shards")
    parser.add_argument("--output_folder", type=str, required=True)

    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--seq_stride", type=int, default=1)

    parser.add_argument("--graph_stride", type=int, default=0, help="0 means infer from graph_index_all")
    parser.add_argument("--chunk_len_graphs", type=int, default=64)
    parser.add_argument(
        "--purge_graphs",
        type=int,
        default=-1,
        help="Gap between chunks in graph units. -1 means auto=max(seq_len, ceil(window_size/graph_stride))",
    )
    parser.add_argument(
        "--chunk_step_graphs",
        type=int,
        default=0,
        help="0 means chunk_len_graphs + purge_graphs",
    )
    parser.add_argument("--min_chunk_graphs", type=int, default=8)

    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--label_mode", type=str, default="last", choices=["last", "majority"])

    args = parser.parse_args()

    graph_folder = Path(args.graph_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("train/val/test ratios must sum to 1.0")

    graph_index = load_table(graph_folder / "graph_index_all")
    print("[Loaded] graph_index_all rows = {}".format(len(graph_index)))

    required_cols = [
        "graph_id",
        "source_class",
        "window_label",
        "y",
        "start_msg_idx_in_file",
        "end_msg_idx_in_file",
        "shard_id",
    ]
    missing = [c for c in required_cols if c not in graph_index.columns]
    if missing:
        raise ValueError("graph_index_all missing columns: {}".format(missing))

    if args.graph_stride > 0:
        graph_stride = args.graph_stride
    else:
        graph_stride = infer_graph_stride(graph_index)

    print("[Info] graph_stride = {}".format(graph_stride))

    # infer window span in messages
    first_span = int(graph_index["end_msg_idx_in_file"].iloc[0] - graph_index["start_msg_idx_in_file"].iloc[0] + 1)

    if args.purge_graphs >= 0:
        purge_graphs = args.purge_graphs
    else:
        purge_graphs = max(args.seq_len, int(math.ceil(float(first_span) / float(graph_stride))))

    if args.chunk_step_graphs > 0:
        chunk_step_graphs = args.chunk_step_graphs
    else:
        chunk_step_graphs = args.chunk_len_graphs + purge_graphs

    print("[Info] window_span_msgs = {}".format(first_span))
    print("[Info] purge_graphs = {}".format(purge_graphs))
    print("[Info] chunk_step_graphs = {}".format(chunk_step_graphs))

    # 1) contiguous graph runs
    graph_index = assign_contiguous_runs(graph_index, graph_stride=graph_stride)

    # 2) chunk specs
    chunks = build_chunk_specs(
        graph_index=graph_index,
        chunk_len_graphs=args.chunk_len_graphs,
        chunk_step_graphs=chunk_step_graphs,
        min_chunk_graphs=max(args.min_chunk_graphs, args.seq_len),
    )
    if len(chunks) == 0:
        raise ValueError("No chunks were built. Try smaller chunk_len_graphs or min_chunk_graphs.")

    print("[Info] num_chunks = {}".format(len(chunks)))
    print(chunks["chunk_label"].value_counts())

    # 3) split chunks
    chunks = stratified_split_chunks(
        chunks=chunks,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # save chunk table
    chunk_table = chunks.drop(columns=["graph_ids", "shard_ids", "window_labels"])
    if len(chunk_table) > 0:
        try:
            chunk_table.to_parquet(output_folder / "chunk_index.parquet", index=False)
        except Exception:
            chunk_table.to_csv(output_folder / "chunk_index.csv", index=False)
    print("[Saved] chunk_index")

    # 4) build sequences
    summary = {
        "graph_stride": graph_stride,
        "window_span_msgs": first_span,
        "purge_graphs": purge_graphs,
        "chunk_step_graphs": chunk_step_graphs,
        "seq_len": args.seq_len,
        "seq_stride": args.seq_stride,
        "splits": {},
    }

    for split_name in ["train", "val", "test"]:
        print("[Build sequences] split={}".format(split_name))
        seqs = build_sequences_from_chunks(
            chunks=chunks,
            graph_folder=graph_folder,
            split_name=split_name,
            seq_len=args.seq_len,
            seq_stride=args.seq_stride,
            label_mode=args.label_mode,
        )

        save_path = output_folder / "{}_seq.pt".format(split_name)
        torch.save(seqs, save_path)
        print("[Saved] {} | num_sequences={}".format(save_path, len(seqs)))

        y_counter = Counter([int(s["y"]) for s in seqs])
        summary["splits"][split_name] = {
            "num_sequences": len(seqs),
            "label_counts": dict(sorted(y_counter.items(), key=lambda kv: kv[0])),
        }

    save_json(summary, output_folder / "sequence_summary.json")
    print("[Done] Clean sequence dataset built.")


if __name__ == "__main__":
    main()