#!/usr/bin/env bash
for weat in 2 8 9; do
    for k in 2 3 4 5 ; do
        python augmentation.py \
            --path_to_weat=./data/weat_${weat}.txt \
            --path_to_embeddings=./data/post-specialized-embeddings/postspec/ft_postspec.txt \
            --output_path=./data/weat_${weat}_aug_postspec_${k}.txt \
            --k=${k}
    done
done