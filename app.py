import marimo

__generated_with = "0.17.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multitask Evaluation: Graph + Node Classification
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path
    # Update absolute path to avoid FileNotFoundError
    sys.path.append('/home/hieutt/MRGA-KAN')
    print("System Paths appended:", sys.path[-3:])
    return (Path,)


@app.cell
def _():
    import os
    import json
    import math
    import random
    from typing import Dict, List, Optional, Tuple
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score
    from sklearn.manifold import TSNE
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from networks.graph_attention_ffn_kan_multitask_updated import GraphAttentionKAN

    # Group marimo and altair imports here
    import marimo as mo
    import altair as alt

    plt.rcParams['figure.figsize'] = (8, 5)
    return (
        Data,
        DataLoader,
        Dict,
        F,
        GraphAttentionKAN,
        List,
        Optional,
        TSNE,
        accuracy_score,
        alt,
        balanced_accuracy_score,
        classification_report,
        f1_score,
        mo,
        np,
        pd,
        plt,
        sns,
        torch,
    )


@app.cell
def _(Path, torch):
    # ========= CONFIG =========
    # Use absolute path so the system always finds the directory regardless of where it is run
    PROJECT_ROOT = Path("/home/hieutt/MRGA-KAN").resolve()

    GRAPH_FOLDER = PROJECT_ROOT / "data/2017-subaru-forester/graphs_subsample_node_classification_v2"
    CKPT_PATH = PROJECT_ROOT / "save/graph_attention_ffn_kan_multitask_turning/graph_attention_ffn_kan_multitask_turning_best_joint.pth"

    SPLIT = "test"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 128
    NUM_WORKERS = 8
    MAX_SHARDS = 0      # 0 = load all shards
    MAX_SAMPLES = 0     # 0 = use all graphs in split

    # node task config
    NODE_TARGET_OVERRIDE = None
    PRINT_TOP_NODE_ERRORS = 20

    # t-SNE config (Reduce max_points for smoother WebUI brushing)
    TSNE_MAX_POINTS = 1500
    TSNE_PERPLEXITY = 30
    TSNE_RANDOM_STATE = 42

    print("GRAPH_FOLDER:", GRAPH_FOLDER)
    print("CKPT_PATH   :", CKPT_PATH)
    print("DEVICE      :", DEVICE)
    print("graph exists:", GRAPH_FOLDER.exists())
    print("ckpt exists :", CKPT_PATH.exists())
    return (
        BATCH_SIZE,
        CKPT_PATH,
        DEVICE,
        GRAPH_FOLDER,
        MAX_SAMPLES,
        MAX_SHARDS,
        NODE_TARGET_OVERRIDE,
        NUM_WORKERS,
        SPLIT,
        TSNE_MAX_POINTS,
        TSNE_PERPLEXITY,
        TSNE_RANDOM_STATE,
    )


@app.cell
def _(
    Data,
    Dict,
    F,
    GraphAttentionKAN,
    List,
    Optional,
    Path,
    TSNE,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    np,
    pd,
    plt,
    sns,
    torch,
):
    # ========= HELPERS =========
    def load_table(base: Path) -> pd.DataFrame:
        pq = base.with_suffix('.parquet')
        csv = base.with_suffix('.csv')
        if pq.exists():
            return pd.read_parquet(pq)
        if csv.exists():
            return pd.read_csv(csv)
        raise FileNotFoundError(f'Cannot find {pq} or {csv}')

    def infer_label_mapping(graph_folder: Path) -> Dict[int, str]:
        mapping = {}
        for split in ['train', 'val', 'test']:
            try:
                df = load_table(graph_folder / f'graph_index_{split}')
                if 'y' in df.columns and 'window_label' in df.columns:
                    pairs = df[['y', 'window_label']].drop_duplicates()
                    for (_, row) in pairs.iterrows():
                        mapping[int(row['y'])] = str(row['window_label'])
            except Exception:
                continue
        if not mapping:
            raise RuntimeError('Could not infer label mapping from graph_index files.')
        return dict(sorted(mapping.items(), key=lambda kv: kv[0]))

    def graph_dict_to_data(graph: dict, fallback_graph_id: Optional[str]=None) -> Data:
        data = Data(x=graph['x'].float(), edge_index=graph['edge_index'].long(), edge_attr=graph['edge_attr'].float(), edge_type=graph['edge_type'].long(), id_token=graph['id_index'].long(), y=graph['y'].view(-1).long())
        if 'node_y' in graph:
            data.node_y = graph['node_y'].long()
        if 'node_mask' in graph:
            data.node_mask = graph['node_mask'].bool()
        if 'node_is_attack' in graph:
            data.node_is_attack = graph['node_is_attack'].long()
        meta = graph.get('meta', {})
        data.attack_count = torch.tensor([int(meta.get('attack_count', 0))], dtype=torch.long)
        data.attack_ratio = torch.tensor([float(meta.get('attack_ratio', 0.0))], dtype=torch.float32)
        data.is_mixed_window = torch.tensor([int(bool(meta.get('is_mixed_window', False)))], dtype=torch.long)
        graph_id = graph.get('graph_id', fallback_graph_id if fallback_graph_id is not None else 'unknown_graph')
        label_name = graph.get('window_label', None)
        data.graph_id = graph_id
        data.window_label = label_name if label_name is not None else None
        return data

    def load_graph_split(graph_folder: Path, split_name: str, max_shards: int=0, max_samples: int=0) -> List[Data]:
        split_dir = graph_folder / split_name
        shard_paths = sorted(split_dir.glob(f'graphs_{split_name}_shard*.pt'))
        if len(shard_paths) == 0:
            raise FileNotFoundError(f'No shard files found in {split_dir}')
        if max_shards > 0:
            shard_paths = shard_paths[:max_shards]
        dataset: List[Data] = []
        count = 0
        for shard_path in shard_paths:
            shard_graphs = torch.load(shard_path, map_location='cpu', weights_only=False)
            for (i, g) in enumerate(shard_graphs):
                fallback_graph_id = f'{split_name}:{shard_path.stem}:{i}'
                dataset.append(graph_dict_to_data(g, fallback_graph_id=fallback_graph_id))
                count += 1
                if max_samples > 0 and count >= max_samples:
                    return dataset
        return dataset

    def build_model_from_ckpt(ckpt: dict, num_classes: int, num_node_classes: int, num_ids: int, sample_data: Data, device: str='cpu'):
        args = ckpt.get('args', {})
        model = GraphAttentionKAN(node_feat_dim=sample_data.x.size(1), edge_attr_dim=sample_data.edge_attr.size(1), num_classes=num_classes, num_node_classes=num_node_classes, num_ids=num_ids, hidden_dim=args.get('hidden_dim', 128), num_layers=args.get('num_layers', 3), heads=args.get('heads', 4), id_emb_dim=args.get('id_emb_dim', 32), rel_emb_dim=args.get('rel_emb_dim', 8), num_relations=args.get('num_relations', 4), dropout=args.get('dropout', 0.2), ffn_ratio=args.get('ffn_ratio', 2.0), block_kan_grid_size=args.get('block_kan_grid_size', 5), block_kan_spline_order=args.get('block_kan_spline_order', 3), block_kan_scale_noise=args.get('block_kan_scale_noise', 0.1), block_kan_scale_base=args.get('block_kan_scale_base', 1.0), block_kan_scale_spline=args.get('block_kan_scale_spline', 1.0), kan_hidden=args.get('kan_hidden', 128), kan_grid_size=args.get('kan_grid_size', 5), kan_spline_order=args.get('kan_spline_order', 3), kan_scale_noise=args.get('kan_scale_noise', 0.1), kan_scale_base=args.get('kan_scale_base', 1.0), kan_scale_spline=args.get('kan_scale_spline', 1.0), node_head_from_layer=args.get('node_head_from_layer', -1)).to(device)
        (missing, unexpected) = model.load_state_dict(ckpt['model'], strict=False)
        model.eval()
        return model

    @torch.no_grad()
    def evaluate_multitask_model(model, loader, graph_label_mapping: Dict[int, str], node_target: str='node_y', node_label_mapping: Optional[Dict[int, str]]=None, device: str='cpu'):
        model.eval()
        (graph_y_true, graph_y_pred) = ([], [])
        graph_probs_all = []
        graph_logits_all = []
        embeddings_all = []
        graph_ids = []
        (node_y_true, node_y_pred) = ([], [])

        def _get_node_targets_and_mask(batch: Data, node_target: str):
            if not hasattr(batch, node_target): return (None, None)
            node_y = getattr(batch, node_target).view(-1).long()
            node_mask = getattr(batch, 'node_mask', None)
            if node_mask is None: node_mask = torch.ones_like(node_y, dtype=torch.bool)
            else: node_mask = node_mask.view(-1).bool()
            return (node_y, node_mask)

        for batch in loader:
            batch = batch.to(device)
            out = model(batch, return_graph_embedding=True, return_node_logits=True)
            graph_logits = out['graph_logits']
            node_logits = out.get('node_logits', None)
            graph_emb = out['graph_embedding']

            graph_probs = F.softmax(graph_logits, dim=1)
            graph_pred = graph_probs.argmax(dim=1)
            graph_y_true.extend(batch.y.view(-1).detach().cpu().numpy().tolist())
            graph_y_pred.extend(graph_pred.detach().cpu().numpy().tolist())
            graph_probs_all.append(graph_probs.detach().cpu().numpy())
            graph_logits_all.append(graph_logits.detach().cpu().numpy())
            embeddings_all.append(graph_emb.detach().cpu().numpy())

            gid = getattr(batch, 'graph_id', None)
            if gid is None: graph_ids.extend([f'graph_{len(graph_ids) + i}' for i in range(batch.y.numel())])
            elif isinstance(gid, list): graph_ids.extend(gid)
            else: graph_ids.extend(list(gid) if hasattr(gid, '__iter__') and not isinstance(gid, str) else [str(gid)] * batch.y.numel())

            if node_logits is not None:
                (node_target_y, node_mask) = _get_node_targets_and_mask(batch, node_target=node_target)
                if node_target_y is not None and node_mask is not None and node_mask.any():
                    node_pred = node_logits[node_mask].argmax(dim=1)
                    node_y_true.extend(node_target_y[node_mask].detach().cpu().numpy().tolist())
                    node_y_pred.extend(node_pred.detach().cpu().numpy().tolist())

        graph_y_true = np.asarray(graph_y_true)
        graph_y_pred = np.asarray(graph_y_pred)
        graph_probs = np.concatenate(graph_probs_all, axis=0) if len(graph_probs_all) else np.empty((0, len(graph_label_mapping)))
        graph_logits = np.concatenate(graph_logits_all, axis=0) if len(graph_logits_all) else np.empty((0, len(graph_label_mapping)))
        graph_embs = np.concatenate(embeddings_all, axis=0) if len(embeddings_all) else np.empty((0, 1))
        graph_metrics = {'acc': accuracy_score(graph_y_true, graph_y_pred), 'macro_f1': f1_score(graph_y_true, graph_y_pred, average='macro', zero_division=0), 'weighted_f1': f1_score(graph_y_true, graph_y_pred, average='weighted', zero_division=0), 'balanced_acc': balanced_accuracy_score(graph_y_true, graph_y_pred)}

        if len(node_y_true) > 0:
            node_y_true = np.asarray(node_y_true)
            node_y_pred = np.asarray(node_y_pred)
            node_metrics = {'acc': accuracy_score(node_y_true, node_y_pred), 'macro_f1': f1_score(node_y_true, node_y_pred, average='macro', zero_division=0), 'weighted_f1': f1_score(node_y_true, node_y_pred, average='weighted', zero_division=0), 'balanced_acc': balanced_accuracy_score(node_y_true, node_y_pred)}
        else:
            node_y_true, node_y_pred = np.asarray([]), np.asarray([])
            node_metrics = {'acc': 0.0, 'macro_f1': 0.0, 'weighted_f1': 0.0, 'balanced_acc': 0.0}

        return {'graph': {'metrics': graph_metrics, 'y_true': graph_y_true, 'y_pred': graph_y_pred, 'probs': graph_probs, 'logits': graph_logits, 'embeddings': graph_embs, 'graph_ids': graph_ids}, 'node': {'metrics': node_metrics, 'y_true': node_y_true, 'y_pred': node_y_pred, 'label_mapping': node_label_mapping}}

    def report_dataframe(y_true, y_pred, label_mapping: Dict[int, str]) -> pd.DataFrame:
        _labels = sorted(label_mapping.keys())
        target_names = [label_mapping[i] for i in _labels]
        return classification_report(y_true, y_pred, labels=_labels, target_names=target_names, digits=4)

    def draw_confusion_matrix(cm, classes, vmax=10000):
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, fmt='d', annot=True, cmap='YlGnBu', cbar=False, linewidths=0.5, vmin=0, vmax=vmax)
        plt.xticks(rotation=45, ha='right')
        plt.title('Confusion Matrix - MRGAT-KAN(Full model)')
        plt.ylabel('Actual Values')
        plt.xlabel('Predicted Values')
        plt.show()

    def run_tsne(embeddings: np.ndarray, max_points: int=3000, perplexity: int=30, random_state: int=42):
        if len(embeddings) == 0: raise ValueError('No embeddings to visualize.')
        n = len(embeddings)
        if max_points > 0 and n > max_points:
            rng = np.random.default_rng(random_state)
            keep_idx = np.sort(rng.choice(n, size=max_points, replace=False))
            emb = embeddings[keep_idx]
        else:
            keep_idx = np.arange(n)
            emb = embeddings
        tsne = TSNE(n_components=2, perplexity=min(perplexity, max(5, len(emb) - 1)), init='pca', learning_rate='auto', random_state=random_state)
        emb2d = tsne.fit_transform(emb)
        return (emb2d, keep_idx)
    return (
        build_model_from_ckpt,
        evaluate_multitask_model,
        infer_label_mapping,
        load_graph_split,
        report_dataframe,
        run_tsne,
    )


@app.cell
def _(
    BATCH_SIZE,
    CKPT_PATH,
    DEVICE,
    DataLoader,
    GRAPH_FOLDER,
    MAX_SAMPLES,
    MAX_SHARDS,
    NODE_TARGET_OVERRIDE,
    NUM_WORKERS,
    SPLIT,
    infer_label_mapping,
    load_graph_split,
    torch,
):
    # ========= LOAD DATA =========
    ckpt_preview = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    ckpt_args = ckpt_preview.get("args", {})

    label_mapping = infer_label_mapping(GRAPH_FOLDER)

    node_target = NODE_TARGET_OVERRIDE if NODE_TARGET_OVERRIDE is not None else ckpt_args.get("node_target", "node_y")
    if node_target == "node_is_attack":
        node_label_mapping = {0: "normal", 1: "attack"}
        num_node_classes = 2
    else:
        node_label_mapping = label_mapping.copy()
        num_node_classes = len(node_label_mapping)

    test_dataset = load_graph_split(
        GRAPH_FOLDER,
        split_name=SPLIT,
        max_shards=MAX_SHARDS,
        max_samples=MAX_SAMPLES,
    )

    print("Loaded test graphs:", len(test_dataset))
    print("num graph classes :", len(label_mapping))
    print("node_target       :", node_target)
    print("num node classes  :", num_node_classes)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )
    return (
        label_mapping,
        node_label_mapping,
        node_target,
        num_node_classes,
        test_dataset,
        test_loader,
    )


@app.cell
def _(
    CKPT_PATH,
    DEVICE,
    build_model_from_ckpt,
    label_mapping,
    num_node_classes,
    test_dataset,
    torch,
):
    # ========= LOAD CHECKPOINT + BUILD MODEL =========
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    num_classes = len(label_mapping)
    num_ids = int(max([d.id_token.max().item() for d in test_dataset])) + 1
    sample_data = test_dataset[0]

    model = build_model_from_ckpt(
        ckpt=ckpt, num_classes=num_classes, num_node_classes=num_node_classes,
        num_ids=num_ids, sample_data=sample_data, device=DEVICE
    )
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Run evaluation
    """)
    return


@app.cell
def _(np):
    def compute_fnr_percent(y_true, y_pred, normal_class=0):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        true_attack, pred_attack = y_true != normal_class, y_pred != normal_class
        fn, tp = np.sum(true_attack & ~pred_attack), np.sum(true_attack & pred_attack)
        denom = tp + fn
        return 0.0 if denom == 0 else fn / denom * 100.0
    return


@app.cell
def _(
    DEVICE,
    evaluate_multitask_model,
    label_mapping,
    model,
    node_label_mapping,
    node_target,
    test_loader,
):
    outputs = evaluate_multitask_model(model=model, loader=test_loader, graph_label_mapping=label_mapping, node_target=node_target, node_label_mapping=node_label_mapping, device=DEVICE)
    graph_out, node_out = outputs['graph'], outputs['node']

    print('=== GRAPH CLASSIFICATION ===')
    print('ACC         :', f"{graph_out['metrics']['acc'] * 100:.2f}%")
    print('Macro-F1    :', f"{graph_out['metrics']['macro_f1']:.4f}")

    graph_classes = ['Normal', 'Combined', 'DoS', 'Fuzzy', 'Gear', 'Interval', 'RPM', 'Speed', 'Standstill', 'Systematic']
    return (graph_out,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Classification Reports
    """)
    return


@app.cell
def _(graph_out, label_mapping, report_dataframe):
    print("Graph Classification Report:\n", report_dataframe(graph_out["y_true"], graph_out["y_pred"], label_mapping))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Interactive Graph Analysis (t-SNE & Topology)
    """)
    return


@app.cell
def _(
    TSNE_MAX_POINTS,
    TSNE_PERPLEXITY,
    TSNE_RANDOM_STATE,
    alt,
    graph_out,
    label_mapping,
    mo,
    pd,
    run_tsne,
):
    # 1. Calculate t-SNE dimensionality reduction
    emb2d, kept_idx = run_tsne(
        graph_out["embeddings"], max_points=TSNE_MAX_POINTS, 
        perplexity=TSNE_PERPLEXITY, random_state=TSNE_RANDOM_STATE
    )

    y_true_tsne = graph_out["y_true"][kept_idx]
    y_pred_tsne = graph_out["y_pred"][kept_idx]
    graph_ids_tsne = [graph_out["graph_ids"][i] for i in kept_idx]
    text_labels = [label_mapping[int(y)] for y in y_true_tsne]

    # 2. Altair DataFrame
    df_altair = pd.DataFrame({
        "t-SNE 1": emb2d[:, 0],
        "t-SNE 2": emb2d[:, 1],
        "Label": text_labels,
        "Graph ID": graph_ids_tsne,
        "Prediction": [label_mapping[int(p)] for p in y_pred_tsne]
    })

    brush = alt.selection_interval(name="brush")
    base_chart = alt.Chart(df_altair).mark_circle(size=50, opacity=0.8).encode(
        x=alt.X("t-SNE 1:Q", scale=alt.Scale(zero=False)),
        y=alt.Y("t-SNE 2:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("Label:N", scale=alt.Scale(scheme="tableau10"), title="Actual Class"),
        tooltip=["Graph ID", "Label", "Prediction"]
    ).add_params(
        brush
    ).properties(
        width=800, height=500,
        title="Interactive t-SNE: Click and drag to brush and inspect data clusters"
    )

    chart_ui = mo.ui.altair_chart(base_chart)

    chart_ui
    return (chart_ui,)


@app.cell
def _(alt, chart_ui, mo):
    # --- BRUSH ANALYSIS AND GRAPH DROPDOWN SELECTION CELL ---
    selected_data = chart_ui.value
    graph_selector = None
    ui_display = None

    if selected_data is not None and not selected_data.empty:
        total_selected = len(selected_data)

        # Label distribution chart in selected area
        counts = selected_data["Label"].value_counts().reset_index()
        counts.columns = ["Class", "Count"]
        bar_chart = alt.Chart(counts).mark_bar().encode(
            x=alt.X("Count:Q", title="Sample Count"),
            y=alt.Y("Class:N", sort="-x", title="Attack Type"),
            color=alt.Color("Class:N", scale=alt.Scale(scheme="tableau10"), legend=None)
        ).properties(width=500, height=150, title=f"Cluster Distribution ({total_selected} samples)")

        # Create Dropdown to select specific Graph ID
        graph_ids_list = selected_data["Graph ID"].tolist()
        graph_selector = mo.ui.dropdown(
            options=graph_ids_list,
            value=graph_ids_list[0] if graph_ids_list else None,
            label="🔍 SELECT GRAPH ID IN REGION TO VIEW DETAILS (TOPOLOGY):"
        )

        # Assign UI layout to ui_display variable
        ui_display = mo.vstack([
            bar_chart,
            mo.md("---"),
            mo.md("### Graph Structure & Embedding Vector Analysis (Micro-analysis)"),
            graph_selector
        ])
    else:
        ui_display = mo.md("💡 *Tip: Brush a region on the t-SNE plot to display detailed graph analysis tools.*")

    # CALL VARIABLE AT THE LAST LINE TO EXPORT DROPDOWN UI
    ui_display
    return (graph_selector,)


@app.cell
def _(alt, graph_out, graph_selector, mo, np, pd, plt, test_dataset):
    # --- GRAPH TOPOLOGY (NETWORKX) AND EMBEDDING VISUALIZATION CELL ---
    import networkx as nx
    from torch_geometric.utils import to_networkx

    # Default message to prevent invisible cell when no brush is made
    ui_details = mo.md("⏳ *System waiting: Please brush a cluster on the t-SNE plot above and select a Graph ID from the Dropdown to view details!*")

    if graph_selector is not None and graph_selector.value:
        selected_id = graph_selector.value

        # 1. Visualization Vector Embedding (Bar Chart)
        emb_chart = mo.md("Embedding vector not found.")
        try:
            idx = graph_out["graph_ids"].index(selected_id)
            emb_vector = graph_out["embeddings"][idx]
            df_emb = pd.DataFrame({"Dimension": np.arange(len(emb_vector)), "Value": emb_vector})

            emb_chart = alt.Chart(df_emb).mark_bar(size=2).encode(
                x=alt.X("Dimension:O", title="Dimension (Latent Dimension)", axis=alt.Axis(labels=False, ticks=False)),
                y=alt.Y("Value:Q", title="Activation Value"),
                tooltip=["Dimension", "Value"],
                color=alt.condition(alt.datum.Value > 0, alt.value("steelblue"), alt.value("orange"))
            ).properties(width=800, height=150, title=f"Graph Embedding Vector (Readout): {selected_id}")
        except ValueError:
            pass

        # 2. Visualization Topology (NetworkX)
        target_graph = None
        for data in test_dataset:
            if getattr(data, "graph_id", None) == selected_id:
                target_graph = data
                break

        if target_graph is not None:
            G = to_networkx(target_graph, to_undirected=False)
            fig, ax = plt.subplots(figsize=(10, 7))

            # Initialize node distribution layout
            pos = nx.spring_layout(G, seed=42)

            # --- ACCURATE AND SAFE NODE COLORING LOGIC ---
            num_nodes = G.number_of_nodes()
            node_colors = ['skyblue'] * num_nodes

            try:
                # Priority 1: Read from node_y (Move to CPU to prevent device Tensor errors)
                if hasattr(target_graph, 'node_y') and target_graph.node_y is not None:
                    labels = target_graph.node_y.cpu().detach().view(-1).numpy()
                    if len(labels) == num_nodes:
                        node_colors = ['skyblue' if l == 0 else 'red' for l in labels]

                # Priority 2: Read from node_is_attack
                elif hasattr(target_graph, 'node_is_attack') and target_graph.node_is_attack is not None:
                    labels = target_graph.node_is_attack.cpu().detach().view(-1).numpy()
                    if len(labels) == num_nodes:
                        node_colors = ['red' if l == 1 else 'skyblue' for l in labels]
            except Exception as e:
                print(f"Error reading node labels: {e}")
            # -----------------------------------------------

            nx.draw_networkx_nodes(G, pos, node_size=250, node_color=node_colors, alpha=0.9, edgecolors='gray', ax=ax)
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.4, edge_color='gray', arrows=True, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)

            ax.set_title(f"Network Topology of ID: {selected_id}\n(Red Node: Attacked / Blue Node: Normal)", fontsize=13)
            ax.axis("off")

            topo_html = mo.as_html(fig)
            plt.close(fig)

            if hasattr(target_graph, 'x'):
                feat_df = pd.DataFrame(target_graph.x[:10].cpu().detach().numpy())
                feat_df.columns = feat_df.columns.astype(str)
            else:
                feat_df = None

            ui_details = mo.vstack([
                emb_chart,
                mo.md("---"),
                topo_html,
                mo.md(f"**Total Nodes (CAN Messages):** `{target_graph.num_nodes}` | **Total Edges (Connections):** `{target_graph.num_edges}`"),
                mo.md("**Extracted Payload (x) of the first 10 nodes:**") if feat_df is not None else mo.md(""),
                feat_df.iloc[:, :15] if feat_df is not None else mo.md("")
            ])
        else:
            ui_details = mo.md("❌ *Could not find topology structure for this graph in test_dataset.*")

    ui_details
    return


if __name__ == "__main__":
    app.run()
