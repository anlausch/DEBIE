#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
for config_number in 0 1 2 3 4 5; do
    echo $config_number
    python exp_debbie.py \
        ${config_number} ${config_number} |& tee ./output/debbie_${config_number}.out
done
