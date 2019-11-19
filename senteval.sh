#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="${PYTHONPATH}:/work/anlausch/SentEval"
#for output_path_root in "/work/anlausch/debbie/output/data/weat_8_postspec_4_wo_original_postspec"; do
for output_path_root in "/work/anlausch/debbie/output"; do
#    for config in "drp=0.9_rf=0.2_ef=1.0_if=1.0"; do
        SENTEVAL_DATA_PATH=/home/rlitschk/projects/SentEval/data
        OUTPUT_PATH=${output_path_root}
        #/${config}
        #OUTPUT_PATH=${output_path_root}
        #EMBEDDING_VECTOR_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vectors
        #EMBEDDING_VECTOR_PATH=/work/gglavas/data/word_embs/debie/utah-procrustes.vectors.npy
        #EMBEDDING_VECTOR_PATH=/work/gglavas/data/word_embs/debie/procrustes-utah.vectors.npy
        EMBEDDING_VECTOR_PATH=/work/gglavas/data/word_embs/debie/procrustes.vectors.npy
        #EMBEDDING_VECTOR_PATH=${OUTPUT_PATH}/${config}.vec
        EMBEDDING_VOCAB_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab
        #EMBEDDING_VOCAB_PATH=/home/anlausch/post-specialized-embeddings/postspec/ft_postspec.txt
        python ./../SentEval/examples/bow.py \
            --embedding_vocab_path \
            ${EMBEDDING_VOCAB_PATH} \
            --embedding_vector_path \
            ${EMBEDDING_VECTOR_PATH} \
            --specialized_embeddings \
            False \
            --senteval_data_path \
            ${SENTEVAL_DATA_PATH} |& tee ${OUTPUT_PATH}/senteval_procrustes.log
#    done
done