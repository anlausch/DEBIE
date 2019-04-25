#!/usr/bin/env bash
OUTPUT_PATH=/work/anlausch/debbie/output

EMBEDDING_VECTOR_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vectors
EMBEDDING_VOCAB_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab

python simlex.py \
    --output_path=${OUTPUT_PATH} \
    --embedding_vector_path=${EMBEDDING_VECTOR_PATH} \
    --embedding_vocab_path=${EMBEDDING_VOCAB_PATH}


