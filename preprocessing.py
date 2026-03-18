import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import ast

# --- CẤU HÌNH ---
INPUT_DIR = './data/2017-subaru-forester/merged'
OUTPUT_FILE = './data/2017-subaru-forester/preprocessed/multiclass_graphs_data.pt'
WINDOW_SIZE = 50
STEP_SIZE = 25

ATTACK_MAP = {
    'normal': 0, # Mặc định
    'dos': 1,
    'fuzzing': 2,
    'gear': 3,
    'interval': 4,
    'rpm': 5,
    'speed': 6,
    'standstill': 7,
    'systematic': 8,
    'combined': 9
}

def bin_string_to_bytes(bin_str):
    return [int(bin_str[i:i+8], 2) for i in range(0, 64, 8)]

def create_dataset():
    csv_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    
    # 1. Tạo Global ID Mapping (Giữ nguyên như cũ)
    unique_ids = set()
    print("Scanning for IDs...")
    for f in tqdm(csv_files):
        df_temp = pd.read_csv(f, usecols=['arbitration_id'])
        unique_ids.update(df_temp['arbitration_id'].unique())
    id_map = {can_id: i for i, can_id in enumerate(sorted(unique_ids))}
    num_nodes = len(id_map)
    
    all_graphs = []
    
    # 2. Xử lý từng file
    for file_path in csv_files:
        filename = os.path.basename(file_path).lower() # dos.csv
        
        # Xác định Attack ID dựa trên tên file
        # Ví dụ: file 'DoS.csv' -> chứa attack loại 1
        current_attack_type_id = 0
        for key, val in ATTACK_MAP.items():
            if key in filename:
                current_attack_type_id = val
                break
        
        print(f"Processing {filename} -> Attack Class: {current_attack_type_id} ({list(ATTACK_MAP.keys())[list(ATTACK_MAP.values()).index(current_attack_type_id)]})")
        
        df = pd.read_csv(file_path)
        byte_data = np.array([bin_string_to_bytes(x) for x in df['data_field']])
        timestamps = df['timestamp'].values
        can_ids = df['arbitration_id'].values
        # Cột attack gốc trong CSV chỉ là 0 hoặc 1
        raw_labels = df['attack'].values 
        delta_t = np.diff(timestamps, prepend=timestamps[0])
        
        num_samples = len(df)
        
        for i in tqdm(range(0, num_samples - WINDOW_SIZE + 1, STEP_SIZE), leave=False):
            w_ids = can_ids[i : i+WINDOW_SIZE]
            w_bytes = byte_data[i : i+WINDOW_SIZE]
            w_dt = delta_t[i : i+WINDOW_SIZE]
            w_raw_labels = raw_labels[i : i+WINDOW_SIZE] # 0 hoặc 1
            
            # --- TÍNH TOÁN NODE FEATURES (Giữ nguyên) ---
            x = np.zeros((num_nodes, 19))
            present_ids, counts = np.unique(w_ids, return_counts=True)
            
            for pid, count in zip(present_ids, counts):
                idx = id_map[pid]
                mask = (w_ids == pid)
                
                # Feature 0: Count
                x[idx, 0] = count / WINDOW_SIZE
                
                # Feature 1-2: Time (Mean & Std)
                curr_dt = w_dt[mask]
                x[idx, 1] = np.mean(curr_dt)
                x[idx, 2] = np.std(curr_dt) # <--- QUAN TRỌNG: Bắt được sự bất ổn về thời gian
                
                # Feature 3-18: Bytes (Mean & Std cho 8 bytes)
                curr_bytes = w_bytes[mask] # Shape [k, 8]
                
                # Mean 8 bytes (cũ)
                x[idx, 3:11] = np.mean(curr_bytes, axis=0) / 255.0
                
                # Std 8 bytes (MỚI - Cực quan trọng cho Fuzzing/Gear)
                # Nếu byte nhảy loạn xạ -> Std cao. Nếu byte đứng im -> Std thấp.
                x[idx, 11:19] = np.std(curr_bytes, axis=0) / 255.0

            x_tensor = torch.tensor(x, dtype=torch.float)
            
            # Edges
            node_indices = [id_map[uid] for uid in w_ids]
            edge_index = torch.tensor([node_indices[:-1], node_indices[1:]], dtype=torch.long)
            
            # --- XỬ LÝ NHÃN MULTI-CLASS (QUAN TRỌNG) ---
            # Nếu trong window có gói tin tấn công (raw_label=1) -> Gán nhãn loại tấn công cụ thể (ví dụ DoS=1)
            # Nếu window hoàn toàn sạch (raw_label toàn 0) -> Gán nhãn Normal (0)
            if np.sum(w_raw_labels) > 0:
                y_val = current_attack_type_id # Ví dụ: 1 (DoS)
            else:
                y_val = 0 # Normal
                
            y_tensor = torch.tensor([y_val], dtype=torch.long)
            
            data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
            all_graphs.append(data)

    print(f"Total Graphs: {len(all_graphs)}")
    torch.save(all_graphs, OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    create_dataset()