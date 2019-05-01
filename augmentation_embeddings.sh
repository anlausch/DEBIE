#!/usr/bin/env bash
for weat in 1 2 8 9; do
    for k in 2 3 4 5 ; do
        python augmentation.py \
            --path_to_weat=/work/anlausch/debbie/data/weat_${weat}.txt \
            --path_to_embeddings=/home/anlausch/post-specialized-embeddings/postspec/ft_postspec.txt \
            --output_path=/work/anlausch/debbie/data/weat_${weat}_aug_postspec_${k}_new.txt \
            --k=${k}
    done
done