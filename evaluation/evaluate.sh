#!/usr/bin/env bash
for output_path_root in "/work/anlausch/debbie/output/data/glove_weat_8_postspec_4_wo_original"; do
#for output_path_root in "/work/anlausch/debbie/output/data/weat_8_postspec_4_wo_original_random_new"; do
    for config in "drp=0.9_rf=0.1_ef=1.0_if=0.0" "drp=0.9_rf=0.2_ef=1.0_if=0.0" "drp=0.9_rf=0.25_ef=1.0_if=0.0"; do

        OUTPUT_PATH=${output_path_root}/${config}

        EMBEDDING_VECTOR_PATH=${OUTPUT_PATH}/${config}.vec
        #EMBEDDING_VOCAB_PATH=/work/anlausch/debbie/output/w2v_cbow_200k.vocab
        EMBEDDING_VOCAB_PATH=/work/anlausch/debbie/output/glove_200k.vocab
        #EMBEDDING_VECTOR_PATH=/work/gglavas/data/word_embs/debie/utah-procrustes.vectors.npy
        #EMBEDDING_VECTOR_PATH=${OUTPUT_PATH}/ft_postspec.vec
        #EMBEDDING_VOCAB_PATH=${OUTPUT_PATH}/ft_postspec.vocab
        #EMBEDDING_VOCAB_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab
        #EMBEDDING_VOCAB_PATH=/work/anlausch/debbie/output/glove_200k.vocab
        #EMBEDDING_VOCAB_PATH=/home/anlausch/post-specialized-embeddings/postspec/ft_postspec.txt
        TEST_NUMBER=8

        python simlex.py \
            --output_path=${OUTPUT_PATH} \
            --embedding_vector_path=${EMBEDDING_VECTOR_PATH} \
            --embedding_vocab_path=${EMBEDDING_VOCAB_PATH} \
            --specialized_embeddings=False


        python ./../xweat/weat.py \
            --test_number $TEST_NUMBER \
            --permutation_number 1000000 \
            --output_file ${OUTPUT_PATH}/weat_${TEST_NUMBER}.txt \
            --lower True \
            --use_glove False \
            --embedding_vocab \
            ${EMBEDDING_VOCAB_PATH} \
            --embedding_vectors \
            ${EMBEDDING_VECTOR_PATH}\
            --postspec=False \
            --similarity_type cosine |& tee ${OUTPUT_PATH}/weat_${TEST_NUMBER}.log

    done
done