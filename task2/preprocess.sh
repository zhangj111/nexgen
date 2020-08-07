#!/usr/bin/env bash
# baseline
#python preprocess.py -train_src data/baseline/src-train.txt -train_tgt data/baseline/tgt-train.txt -valid_src data/baseline/src-valid.txt -valid_tgt data/baseline/tgt-valid.txt -save_data data/baseline/baseline -src_seq_length 10000 -tgt_seq_length 10000 -src_seq_length_trunc 400 -tgt_seq_length_trunc 100

#multi_encoder
dir="multi_slicing"
python preprocess.py -train_src data/$dir/src-train.front -train_ref data/$dir/src-train.back \
-train_tgt data/$dir/tgt-train.txt -valid_src data/$dir/src-valid.front -valid_ref data/$dir/src-valid.back \
-valid_tgt data/$dir/tgt-valid.txt -save_data data/$dir/multi_encoder \
-src_seq_length 10000 -ref_seq_length 10000 -tgt_seq_length 10000 -src_seq_length_trunc 400 -ref_seq_length_trunc 400 \
-tgt_seq_length_trunc 100

#only_try
#dir="only_try"
#python preprocess.py -train_src data/$dir/src-train.txt -train_tgt data/$dir/tgt-train.txt -valid_src data/$dir/src-valid.txt -valid_tgt data/$dir/tgt-valid.txt -save_data data/$dir/baseline -src_seq_length 10000 -tgt_seq_length 10000 -src_seq_length_trunc 400 -tgt_seq_length_trunc 100



