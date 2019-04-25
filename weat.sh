#!/usr/bin/env bash
for model in 0 1 2 3 4 5; do
    for test_number in 1; do
        python ./../xweat/weat.py \
            --test_number $test_number \
            --permutation_number 1000000 \
            --output_file /work/anlausch/debbie/results/debbie_weat_${test_number}_model_${model}.res \
            --lower True \
            --use_glove False \
            --embedding_vocab \
            /work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab \
            --embedding_vectors \
            /work/anlausch/debbie/output_vectors${model}.${model}.vec \
            --similarity_type cosine |& tee /work/anlausch/debbie/results/debbie_weat_${test_number}_model_${model}.out
    done
done