#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
OUTPUT_PATH=./output/data/postspec_2_wo_original_splitted
INPUT_PATH_TRAIN=./data/weat_1_prepared_filtered_postspec_2_wo_original_train.txt
INPUT_PATH_DEV=./data/weat_1_prepared_filtered_postspec_2_wo_original_dev.txt
EMBEDDING_VECTOR_PATH=./data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vectors
EMBEDDING_VOCAB_PATH=./data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab

python exp_debbie_args.py \
    --dropout_keep_probs="[0.9]" \
    --reg_factors="[0.01, 0.05, 0.1, 0.15, 0.2, 0.25]" \
    --output_path=${OUTPUT_PATH} \
    --input_path_train=${INPUT_PATH_TRAIN} \
    --input_path_dev=${INPUT_PATH_DEV} \
    --embedding_vector_path=${EMBEDDING_VECTOR_PATH} \
    --embedding_vocab_path=${EMBEDDING_VOCAB_PATH} \
    |& tee ${OUTPUT_PATH}/log.out

for config in "drp=0.9_rf=0.01" "drp=0.9_rf=0.05" "drp=0.9_rf=0.1" "drp=0.9_rf=0.15" "drp=0.9_rf=0.2" "drp=0.9_rf=0.25" "drp=0.9_rf=0.5"; do
    OUTPUT_PATH2=${OUTPUT_PATH}/${config}
    EMBEDDING_VECTOR_PATH=${OUTPUT_PATH2}/${config}.vec

    python simlex.py \
        --output_path=${OUTPUT_PATH2} \
        --embedding_vector_path=${EMBEDDING_VECTOR_PATH} \
        --embedding_vocab_path=${EMBEDDING_VOCAB_PATH}

    TEST_NUMBER=1
    python ./../xweat/weat.py \
        --test_number $TEST_NUMBER \
        --permutation_number 1000000 \
        --output_file ${OUTPUT_PATH2}/weat_${TEST_NUMBER}.txt \
        --lower False \
        --use_glove False \
        --embedding_vocab \
        ${EMBEDDING_VOCAB_PATH} \
        --embedding_vectors \
        ${EMBEDDING_VECTOR_PATH}\
        --similarity_type cosine |& tee ${OUTPUT_PATH2}/weat_${TEST_NUMBER}.log
done


