# python build_graph_sequences.py \
#   --graph_folder ./data/2017-subaru-forester/graphs_sequences_all \
#   --output_folder ./data/2017-subaru-forester/seq_temporal_clean \
#   --seq_len 4 \
#   --seq_stride 1 \
#   --chunk_len_graphs 64 \
#   --min_chunk_graphs 8 \
#   --train_ratio 0.6 \
#   --val_ratio 0.2 \
#   --test_ratio 0.2 \
#   --seed 42

python build_graph_sequences.py \
  --graph_folder ./data/2017-subaru-forester/graphs_sequences_all \
  --output_folder ./data/2017-subaru-forester/seq_temporal_majority \
  --seq_len 3 \
  --seq_stride 1 \
  --chunk_len_graphs 16 \
  --min_chunk_graphs 6 \
  --purge_graphs 6 \
  --train_ratio 0.6 \
  --val_ratio 0.2 \
  --test_ratio 0.2 \
  --seed 42 \
  --label_mode majority