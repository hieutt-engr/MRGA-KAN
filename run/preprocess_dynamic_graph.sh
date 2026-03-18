python preprocess_dynamic_graph.py \
  --input_dir ./data/2017-subaru-forester/merged \
  --output_dir ./data/2017-subaru-forester/preprocessed_v3 \
  --window_size 64 \
  --stride 16 \
  --sampling_stride 64 \
  --min_attack_count 1 \
  --min_attack_ratio 0.0 \
  --seed 42