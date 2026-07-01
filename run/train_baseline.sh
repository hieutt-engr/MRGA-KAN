# python train_gat_baseline_multitask_ablation.py \
#   --model_variant mlpffn_kanhead \
#   --graph_folder ./data/2017-subaru-forester/graphs_subsample_node_classification_v2 \
#   --save_folder ./save/ablation_mlpffn_kanhead \
#   --model_name ablation_mlpffn_kanhead \
#   --batch_size 128 \
#   --num_workers 8 \
#   --epochs 100 \
#   --learning_rate 0.001 \
#   --weight_decay 0.0001 \
#   --hidden_dim 128 \
#   --num_layers 3 \
#   --node_head_from_layer -1 \
#   --heads 2 \
#   --id_emb_dim 32 \
#   --rel_emb_dim 8 \
#   --dropout 0.2 \
#   --kan_hidden 64 \
#   --loss_name ce \
#   --enable_node_task \
#   --node_target node_y \
#   --node_loss_weight 1.0 \
#   --node_loss_name polyfocal \
#   --focal_gamma 2.0 \
#   --selection_metric joint \
#   --kan_reg_lambda 0.00001 \
#   --device cuda \
#   --gpu_id 0 \
#   --save_epoch_checkpoints \
#   --epoch_save_every 1 \
#   > ./save/log/train_ablation_mlpffn_kanhead.log 2>&1 &


python train_gat_baseline_multitask_ablation.py \
  --model_variant mlpffn_mlphead \
  --graph_folder ./data/2017-subaru-forester/graphs_subsample_node_classification_v2 \
  --save_folder ./save/ablation_mlpffn_mlphead \
  --model_name ablation_mlpffn_mlphead \
  --batch_size 128 \
  --num_workers 8 \
  --epochs 100 \
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
  --loss_name ce \
  --enable_node_task \
  --node_target node_y \
  --node_loss_weight 1.0 \
  --node_loss_name polyfocal \
  --focal_gamma 2.0 \
  --selection_metric joint \
  --kan_reg_lambda 0.00001 \
  --device cuda \
  --gpu_id 0 \
  --save_epoch_checkpoints \
  --epoch_save_every 1 \
  > ./save/log/train_ablation_mlpffn_mlphead.log 2>&1 &


# python train_gat_baseline_multitask_ablation.py \
#   --model_variant ffnkan_mlphead \
#   --graph_folder ./data/2017-subaru-forester/graphs_subsample_node_classification_v2 \
#   --save_folder ./save/ablation_ffnkan_mlphead \
#   --model_name ablation_ffnkan_mlphead \
#   --batch_size 128 \
#   --num_workers 8 \
#   --epochs 100 \
#   --learning_rate 0.001 \
#   --weight_decay 0.0001 \
#   --hidden_dim 128 \
#   --num_layers 3 \
#   --node_head_from_layer -1 \
#   --heads 2 \
#   --id_emb_dim 32 \
#   --rel_emb_dim 8 \
#   --dropout 0.2 \
#   --kan_hidden 64 \
#   --loss_name ce \
#   --enable_node_task \
#   --node_target node_y \
#   --node_loss_weight 1.0 \
#   --node_loss_name polyfocal \
#   --focal_gamma 2.0 \
#   --selection_metric joint \
#   --kan_reg_lambda 0.00001 \
#   --device cuda \
#   --gpu_id 0 \
#   --save_epoch_checkpoints \
#   --epoch_save_every 1 \
#   > ./save/log/train_ablation_ffnkan_mlphead.log 2>&1 &