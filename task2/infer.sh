#!/usr/bin/env bash
if [ ! -d "testout" ]; then
        mkdir "testout"
fi
#baseline
#dir="only_try"
#srun --gres=gpu:V100:1 python translate.py -gpu 0  -batch_size 32 -beam_size 5 -model models/$dir/baseline_step_100000.pt -src data/$dir/src-test.txt -output testout/$dir.out


#multi_encoder
dir="multi_slicing"
python translate.py -gpu 0  -batch_size 32 -beam_size 5 -min_length 3 \
-mask_attention -mask_path data/$dir/src-test.mask \
-model models/$dir/multi_encoder_step_100000.pt -src data/$dir/src-test.front -refer -ref_path data/$dir/src-test.back \
-output testout/$dir.out
