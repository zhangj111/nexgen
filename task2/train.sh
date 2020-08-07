#!/usr/bin/env bash
if [ ! -d "models" ]; then
        mkdir "models"
fi
#baseline
#dir="only_try"
#python train.py -save_model models/$dir/baseline -data data/$dir/baseline -global_attention mlp -word_vec_size 128 -rnn_size 128 -layers 1 -encoder_type brnn -train_steps 100000 -max_grad_norm 2 -dropout 0. -batch_size 32 -valid_batch_size 32 -optim adam -learning_rate 0.001 -bridge -log_file models/$dir.log -world_size 1 -gpu_ranks 0

#multi_encoder
dir="multi_slicing"
python train.py -save_model models/$dir/multi_encoder \
-mask_attention -mask_path data/$dir/src-train.mask:data/$dir/src-valid.mask  \
-refer -data data/$dir/multi_encoder \
-global_attention mlp -word_vec_size 128 -rnn_size 128 -layers 1 -encoder_type brnn \
-train_steps 100000 -max_grad_norm 2 -dropout 0. -batch_size 32 -valid_batch_size 32 \
-optim adam -learning_rate 0.001 -bridge -log_file models/$dir.log -world_size 1 -gpu_ranks 0


