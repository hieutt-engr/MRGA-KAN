import torch
import time
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data, Batch

# ==========================================
# EDGE INFERENCE HOTFIXES FOR JETSON
# Bypass missing C++ scatter_reduce_ and sparse attributes on older PyTorch versions
# ==========================================
if not hasattr(torch.Tensor, 'scatter_reduce_'):
    def dummy_scatter_reduce(self, dim, index, src, *args, **kwargs):
        # Fallback to scatter_add_ which has equivalent computational cost (FLOPs)
        return self.scatter_add_(dim, index, src)
    torch.Tensor.scatter_reduce_ = dummy_scatter_reduce

for attr in ['sparse_csc', 'sparse_csr', 'sparse_bsr', 'sparse_bsc']:
    if not hasattr(torch, attr):
        setattr(torch, attr, "dummy_sparse_format")
# ==========================================

# Import the model architecture
from graph_attention_ffn_kan_multitask import GraphAttentionKAN

def count_parameters(model):
    """
    Count total parameters, trainable parameters, and estimate model size (MB).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size assuming float32 (4 bytes per parameter)
    model_size_mb = (total_params * 4) / (1024 ** 2)
    
    return total_params, trainable_params, model_size_mb

def create_dummy_ivn_graph(
    num_nodes=64,       # Average number of nodes in an IVN frame
    num_edges=128,      # Average number of edges
    node_feat_dim=16,   # Adjust to match your dataset specifications
    edge_attr_dim=4,    # Adjust to match your dataset specifications
    num_ids=100,        # Matches the JSON config
    num_relations=4,    # Matches the JSON config
    device='cuda'
):
    """
    Generate a dummy graph simulating a real In-Vehicle Network (IVN) structure.
    """
    x = torch.randn(num_nodes, node_feat_dim, device=device)
    
    # Randomly generate edge connections (2, num_edges)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    edge_attr = torch.randn(num_edges, edge_attr_dim, device=device)
    edge_type = torch.randint(0, num_relations, (num_edges,), device=device)
    
    # ID tokens for node embedding
    id_token = torch.randint(0, num_ids, (num_nodes,), device=device)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type, id_token=id_token)
    
    # Wrap in a Batch object as the model's forward pass utilizes data.batch
    batch_data = Batch.from_data_list([data])
    return batch_data

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Initializing inference test on device: {device.type.upper()}")

    # ==========================================
    # 1. MODEL INITIALIZATION (Based on JSON Setup)
    # ==========================================
    NODE_FEAT_DIM = 16 
    EDGE_ATTR_DIM = 4  
    NUM_CLASSES = 2       # Graph-level classification (e.g., Attack vs. Normal)
    NUM_NODE_CLASSES = 2  # Node-level classification (Multitask enabled)
    NUM_IDS = 100         # Total ID tokens

    model = GraphAttentionKAN(
        node_feat_dim=NODE_FEAT_DIM,
        edge_attr_dim=EDGE_ATTR_DIM,
        num_classes=NUM_CLASSES,
        num_ids=NUM_IDS,
        num_node_classes=NUM_NODE_CLASSES,
        
        # --- Hyperparameters from JSON ---
        hidden_dim=128,
        num_layers=3,
        heads=2,
        id_emb_dim=32,
        rel_emb_dim=8,
        num_relations=4,
        dropout=0.2,
        ffn_ratio=2.0,
        
        block_kan_grid_size=3,
        block_kan_spline_order=3,
        block_kan_scale_noise=0.1,
        block_kan_scale_base=1.0,
        block_kan_scale_spline=1.0,
        
        kan_hidden=64,
        kan_grid_size=3,
        kan_spline_order=3,
        kan_scale_noise=0.1,
        kan_scale_base=1.0,
        kan_scale_spline=1.0
    ).to(device)

    # Enforce evaluation mode to disable Dropout and BatchNorm updates
    model.eval() 

    # Retrieve parameter counts
    total_params, trainable_params, model_size_mb = count_parameters(model)

    # ==========================================
    # 2. DATA PREPARATION & GPU WARM-UP
    # ==========================================
    # Testing with batch_size = 1 to measure real-time edge inference latency
    dummy_input = create_dummy_ivn_graph(
        num_nodes=64, num_edges=200, 
        node_feat_dim=NODE_FEAT_DIM, edge_attr_dim=EDGE_ATTR_DIM, 
        num_ids=NUM_IDS, num_relations=4, device=device
    )

    print("[INFO] Executing GPU warm-up sequence (50 iterations)...")
    with torch.no_grad():
        for _ in tqdm(range(50), desc="Warm-up Progress", leave=False, unit="iter"):
            _ = model(dummy_input, return_node_logits=True)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print("[INFO] GPU warm-up completed.")

    # ==========================================
    # 3. LATENCY & THROUGHPUT MEASUREMENT
    # ==========================================
    num_iterations = 1000
    latencies = []

    print(f"\n[INFO] Starting precise latency measurement ({num_iterations} iterations)...")
    with torch.no_grad():
        for _ in tqdm(range(num_iterations), desc="Inference Testing", unit="iter"):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            _ = model(dummy_input, return_node_logits=True)
            
            if device.type == 'cuda':
                torch.cuda.synchronize() # Crucial for accurate GPU timing
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000) # Convert to milliseconds

    # Statistical calculations
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    p99_latency = np.percentile(latencies, 99)
    throughput = 1000.0 / avg_latency # Samples per second

    # ==========================================
    # 4. PRINT COMPLEXITY REPORT
    # ==========================================
    print("\n" + "="*55)
    print(" MODEL COMPLEXITY & INFERENCE REPORT")
    print("="*55)
    print(f"Total Parameters      : {total_params:,}")
    print(f"Trainable Parameters  : {trainable_params:,}")
    print(f"Estimated Storage Size: {model_size_mb:.2f} MB")
    print("-" * 55)
    print("Evaluation Condition  : Batch Size = 1 (Edge Setting)")
    print(f"Average Latency       : {avg_latency:.3f} ms (± {std_latency:.3f} ms)")
    print(f"99th Percentile (P99) : {p99_latency:.3f} ms")
    print(f"Inference Throughput  : {throughput:.2f} FPS (Samples/sec)")
    print("="*55 + "\n")

if __name__ == "__main__":
    main()