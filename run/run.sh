python train.py --data_folder ./data/2017-subaru-forester/preprocessed/ --n_classes 10  --epochs 100 --num_node_features 27

 /home/hieutt/CAN-SupCon-IDS/data/2017-subaru-forester/preprocessed


python train_transconv_kan.py --data_folder ./data/2017-subaru-forester/preprocessed_graphs_time --train_file train_graphs_T10ms_S5ms_downR3.pt --test_file test_graphs_T10ms_S5ms.pt --loss ldam --hidden_channels 64 --num_layers 3 --heads 4 --batch_size 128 --epochs 100
