#!/usr/bin/env bash

for test_number in 1 2 8 9; do
    python ./../xweat/weat.py \
        --test_number $test_number \
        --permutation_number 1000000 \
        --output_file /work/anlausch/debbie/output/original_weat_${test_number}_cased.txt \
        --lower False \
        --use_glove False \
        --embedding_vocab \
        /work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab \
        --embedding_vectors \
        /work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vectors \
        --similarity_type cosine |& tee /work/anlausch/debbie/output/original_weat_${test_number}_cased.log
done
