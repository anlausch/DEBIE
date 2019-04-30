#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
OUTPUT_PATH=/work/anlausch/debbie/output/data/cleaned_wo_original
INPUT_PATH=/work/anlausch/debbie/data/weat_1_prepared_filtered_wo_original.txt
EMBEDDING_VECTOR_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vectors
EMBEDDING_VOCAB_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab

python exp_debbie_args.py \
    --dropout_keep_probs="[0.9]" \
    --reg_factors="[0.5]" \
    --output_path=${OUTPUT_PATH} \
    --input_path=${INPUT_PATH} \
    --embedding_vector_path=${EMBEDDING_VECTOR_PATH} \
    --embedding_vocab_path=${EMBEDDING_VOCAB_PATH} \
    |& tee ${OUTPUT_PATH}/log.out

OUTPUT_PATH=${OUTPUT_PATH}/drp=0.9_rf=0.5
EMBEDDING_VECTOR_PATH=${OUTPUT_PATH}/drp=0.9_rf=0.5.vec

python simlex.py \
    --output_path=${OUTPUT_PATH} \
    --embedding_vector_path=${EMBEDDING_VECTOR_PATH} \
    --embedding_vocab_path=${EMBEDDING_VOCAB_PATH}

TEST_NUMBER=1
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


