#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=
OUTPUT_PATH=/work/anlausch/debbie/output/data/weat_8_postspec_4_wo_original/drp=0.9_rf=0.2

EMBEDDING_VECTOR_PATH=/work/gglavas/data/app-specific/debie/xling/de.vectors
EMBEDDING_VOCAB_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab

python exp_debbie_args_predict_only.py \
    --output_path=${OUTPUT_PATH} \
    --checkpoint="drp=0.9_rf=0.2" \
    --embedding_vector_path=${EMBEDDING_VECTOR_PATH} \
    --embedding_vocab_path=${EMBEDDING_VOCAB_PATH} \
    --lang=de\
    |& tee ${OUTPUT_PATH}/log.out