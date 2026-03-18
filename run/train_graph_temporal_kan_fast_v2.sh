python train_graph_temporal_kan_v2.py \
  --sequence_folder ./data/2017-subaru-forester/seq_temporal_clean \
  --save_folder ./save/graph_temporal_kan_v2_small \
  --model_name graph_temporal_kan_v2_small \
  --batch_size 64 \
  --num_workers 8 \
  --epochs 100 \
  --learning_rate 1e-3 \
  --weight_decay 1e-4 \
  --freeze_graph_encoder_epochs 0\
  --subsample_train_frac 0.2 \
  --subsample_val_frac 0.2 \
  --subsample_test_frac 0.2 \
  --subsample_min_per_class 0 \
  --sequence_model_mode transformer \
  --use_cls_token \
  --device cuda \
  > ./save/log/train_graph_temporal_kan_v2_small_64.log 2>&1 &