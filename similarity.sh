#!/usr/bin/env bash
OUTPUT_PATH=/work/anlausch/debbie/output/data/weat_8_postspec_4_wo_original_adv_weight/drp=0.9_rf=0.25_ef=1.0_if=0.2

EMBEDDING_VECTOR_PATH=${OUTPUT_PATH}/drp=0.9_rf=0.25_ef=1.0_if=0.2.vec
EMBEDDING_VOCAB_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab

python simlex.py \
    --output_path=${OUTPUT_PATH} \
    --embedding_vector_path=${EMBEDDING_VECTOR_PATH} \
    --embedding_vocab_path=${EMBEDDING_VOCAB_PATH}


