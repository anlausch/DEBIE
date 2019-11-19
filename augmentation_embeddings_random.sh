#!/usr/bin/env bash
for weat in 8; do
    for k in 4 ; do
        for switch in "True" "False"; do
            python augmentation.py \
                --path_to_weat=/work/anlausch/debbie/data/weat_${weat}.txt \
                --path_to_embeddings=/home/anlausch/post-specialized-embeddings/postspec/ft_postspec.txt \
                --output_path=/work/anlausch/debbie/data/weat_${weat}_aug_postspec_random_${k}_switches=${switch}.txt \
                --k=${k} \
                --random_attributes="True" \
                --switch_targets_and_attributes=${switch}
        done
    done
done