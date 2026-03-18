python build_ivn_graphs.py \
  --input_dir ./data/2017-subaru-forester/preprocessed_v3 \
  --output_dir ./data/2017-subaru-forester/graphs_sequences_all \
  --graphs_per_shard 2000 \
  --temporal_k 1 \
  --same_id_k 1 \
  --payload_topk 1 \
  --timing_topk 1 \
  --splits all