#!/usr/bin/env bash
OUTPUT_PATH=/work/anlausch/debbie/output/reg_factor/drp=0.9_rf=0.25

EMBEDDING_VECTOR_PATH=${OUTPUT_PATH}/drp=0.9_rf=0.25.vec
EMBEDDING_VOCAB_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab
TEST_NUMBER=1

python simlex.py \
    --output_path=${OUTPUT_PATH} \
    --embedding_vector_path=${EMBEDDING_VECTOR_PATH} \
    --embedding_vocab_path=${EMBEDDING_VOCAB_PATH}


python ./../xweat/weat.py \
    --test_number $TEST_NUMBER \
    --permutation_number 1000000 \
    --output_file ${OUTPUT_PATH}/weat_${TEST_NUMBER}.txt \
    --lower False \
    --use_glove False \
    --embedding_vocab \
    ${EMBEDDING_VOCAB_PATH} \
    --embedding_vectors \
    ${EMBEDDING_VECTOR_PATH}\
    --similarity_type cosine |& tee ${OUTPUT_PATH}/weat_${TEST_NUMBER}.log


