#!/usr/bin/env bash
for config in "drp=0.9_rf=0.05" "drp=0.9_rf=0.1" "drp=0.9_rf=0.15" "drp=0.9_rf=0.2" "drp=0.9_rf=0.25" "drp=0.9_rf=0.5"; do

    OUTPUT_PATH=/work/anlausch/debbie/output/data/postspec_2_wo_original/${config}

    EMBEDDING_VECTOR_PATH=${OUTPUT_PATH}/${config}.vec
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

done
