#!/usr/bin/env bash

for output_path_root in "/work/anlausch/debbie/output/data/weat_8_postspec_4_wo_original_adv"; do
    for config in "drp=0.9_rf=0.01_adv=True" "drp=0.9_rf=0.1_adv=True" "drp=0.9_rf=0.15_adv=True" "drp=0.9_rf=0.2_adv=True" "drp=0.9_rf=0.25_adv=True" "drp=0.9_rf=0.3_adv=True" "drp=0.9_rf=0.4_adv=True"; do

        OUTPUT_PATH=${output_path_root}/${config}

        EMBEDDING_VECTOR_PATH=${OUTPUT_PATH}/${config}.vec
        EMBEDDING_VOCAB_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab
        TEST_NUMBER=8

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

    done
done