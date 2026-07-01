python train_gat_ffn_kan_multitask.py \
  --graph_folder ./data/2016-chevrolet-silverado/graphs_subsample_node_classification_v2 \
  --save_folder ./save/graph_attention_ffn_kan_multitask_2016-chevrolet-silverado \
  --model_name graph_attention_ffn_kan_multitask_2016-chevrolet-silverado \
  --batch_size  128 \
  --num_workers 8 \
  --epochs 100 \
  --print_freq 100 \
  --learning_rate 0.001 \
  --weight_decay 0.0001 \
  --hidden_dim 128 \
  --num_layers 3 \
  --node_head_from_layer -1 \
  --heads 2 \
  --id_emb_dim 32 \
  --rel_emb_dim 8 \
  --dropout 0.2 \
  --kan_hidden 64 \
  --block_kan_grid_size 3 \
  --kan_grid_size 3 \
  --loss_name ce \
  --enable_node_task \
  --node_target node_y \
  --node_loss_weight 1.5 \
  --selection_metric joint \
  --kan_reg_lambda 0.00001 \
  --use_node_class_weights \
  --node_loss_name polyfocal \
  --focal_gamma 3.0 \
  --device cuda \
  --gpu_id 1 \
  --print_val_node_cm_every 5 \
  --epoch_save_every 5 \
  > ./save/log/train_gat_ffn_kan_multitask_2016-chevrolet-silverado.log 2>&1 &


