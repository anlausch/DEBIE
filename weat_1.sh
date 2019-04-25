#!/usr/bin/env bash
OUTPUT_PATH=/work/anlausch/debbie/output/reg_factor/drp=0.9_rf=0.0

EMBEDDING_VECTOR_PATH=${OUTPUT_PATH}/drp=0.9_rf=0.0.vec
EMBEDDING_VOCAB_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab


for test_number in 1; do
    python ./../xweat/weat.py \
        --test_number $test_number \
        --permutation_number 1000000 \
        --output_file ${OUTPUT_PATH}/weat_${test_number}.txt \
        --lower False \
        --use_glove False \
        --embedding_vocab \
        ${EMBEDDING_VOCAB_PATH} \
        --embedding_vectors \
        ${EMBEDDING_VECTOR_PATH}\
        --similarity_type cosine |& tee ${OUTPUT_PATH}/weat_${test_number}.log
done