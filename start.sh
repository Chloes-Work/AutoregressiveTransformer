#!/bin/bash
python3 train.py --batch_size=128 --num_workers=8 --n_mels=80 --max_epochs=1000 --output_dim=252 --embedding_dim=252 --save_dir=/home/safu/BA/MyTransformer/results --modul_name=Conformer --train_csv_path=./train.csv --valid_csv_path=./valid.csv --test_csv_path=./test.csv --learning_rate=0.001 --num_classes=1251 --loss_name=amsoftmax --num_blocks=6 --step_size=4 --gamma=0.5 --weight_decay=0.0000001 --input_layer=conv2d2 --pos_enc_layer_type=rel_pos --top_n_rows=20 --trial_path=./veri_test2.txt


