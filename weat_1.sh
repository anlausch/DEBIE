#!/usr/bin/env bash
OUTPUT_PATH=/work/anlausch/debbie/output/data/postspec_2_wo_original_repr/drp=0.9_rf=0.15

EMBEDDING_VECTOR_PATH=${OUTPUT_PATH}/drp=0.9_rf=0.15.vec
EMBEDDING_VOCAB_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab


for test_number in 1; do
    python ./../xweat/weat.py \
        --test_number $test_number \
        --permutation_number 100000 \
        --output_file ${OUTPUT_PATH}/weat_${test_number}.txt \
        --lower True \
        --use_glove False \
        --embedding_vocab \
        ${EMBEDDING_VOCAB_PATH} \
        --embedding_vectors \
        ${EMBEDDING_VECTOR_PATH}\
        --similarity_type cosine |& tee ${OUTPUT_PATH}/weat_${test_number}.log
done