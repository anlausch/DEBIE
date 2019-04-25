#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
OUTPUT_PATH=/work/anlausch/debbie/output/reg_factor
INPUT_PATH=/work/anlausch/debbie/data/weat_1_prepared_filtered.txt
EMBEDDING_VECTOR_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vectors
EMBEDDING_VOCAB_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab

python exp_debbie_args.py \
    --dropout_keep_probs="[0.9]" \
    --reg_factors="[0.0, 0.25, 0.5, 0.75, 1.0]" \
    --output_path=${OUTPUT_PATH} \
    --input_path=${INPUT_PATH} \
    --embedding_vector_path=${EMBEDDING_VECTOR_PATH} \
    --embedding_vocab_path=${EMBEDDING_VOCAB_PATH} \
    |& tee ${OUTPUT_PATH}/log.out


