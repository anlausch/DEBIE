#!/usr/bin/env bash
for output_path in "/work/anlausch/debbie/output/data/weat_8_postspec_4_wo_original/drp=0.9_rf=0.2"; do
    OUTPUT_PATH=${output_path}

    #EMBEDDING_VECTOR_PATH=${OUTPUT_PATH}/glove_200k.vec
    #EMBEDDING_VOCAB_PATH=${OUTPUT_PATH}/glove_200k.vocab
    EMBEDDING_VOCAB_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab
    EMBEDDING_VECTOR_PATH=${OUTPUT_PATH}/original_fasttext_debiased.vectors
    #EMBEDDING_VECTOR_PATH=${OUTPUT_PATH}/w2v_cbow_200k.vec
    #EMBEDDING_VOCAB_PATH=${OUTPUT_PATH}/w2v_cbow_200k.vocab

    TEST_NUMBER=8

    python simlex.py \
        --output_path=${OUTPUT_PATH}/best \
        --embedding_vector_path=${EMBEDDING_VECTOR_PATH} \
        --embedding_vocab_path=${EMBEDDING_VOCAB_PATH} \
        --specialized_embeddings=False

    # lower True bei Fasttext and Glove
    # lower False for W2v? (first run with lower true) --> should be lower false
    python ./../xweat/weat.py \
        --test_number $TEST_NUMBER \
        --permutation_number 1000000 \
        --output_file ${OUTPUT_PATH}/best/weat_${TEST_NUMBER}.txt \
        --lower True \
        --use_glove False \
        --embedding_vocab \
        ${EMBEDDING_VOCAB_PATH} \
        --embedding_vectors \
        ${EMBEDDING_VECTOR_PATH}\
        --postspec=False \
        --similarity_type cosine |& tee ${OUTPUT_PATH}/best/weat_${TEST_NUMBER}.log
done