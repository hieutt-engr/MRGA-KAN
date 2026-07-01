# for MODEL in gcn graphsage gin gatv2 edge_gatv2 rgcn transformer
# do
#   python train_graph_baselines.py \
#     --graph_folder ./data/2017-subaru-forester/graphs_subsample_node_classification_v2 \
#     --save_folder ./save/baselines/${MODEL} \
#     --model_type ${MODEL} \
#     --epochs 50 \
#     --batch_size 128 \
#     --device cuda \
#     --gpu_id 0 \ 
#     --print_freq 100 \
#     > ./save/log/train_${MODEL}.log 2>&1 &
# done

python train_graph_baselines.py \
  --model_type rgcn \
  --graph_folder ./data/2017-subaru-forester/graphs_subsample_node_classification_v2 \
  --save_folder ./save/baseline_rgcn\
  --model_name baseline_rgcn \
  --batch_size 128 \
  --num_workers 8 \
  --epochs 40 \
  --learning_rate 0.001 \
  --weight_decay 0.0001 \
  --hidden_dim 128 \
  --num_layers 3 \
  --node_head_from_layer -1 \
  --heads 2 \
  --id_emb_dim 32 \
  --rel_emb_dim 8 \
  --dropout 0.2 \
  --enable_node_task \
  --node_target node_y \
  --node_loss_weight 1.0 \
  --selection_metric joint \
  --device cuda \
  --gpu_id 0 \
  > ./save/log/train_rgcn.log 2>&1 &