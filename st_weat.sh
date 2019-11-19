#!/usr/bin/env bash
OUTPUT_PATH=/work/anlausch/debbie/output

#EMBEDDING_VECTOR_PATH=${OUTPUT_PATH}/drp=0.9_rf=0.5.vec
#EMBEDDING_VOCAB_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab

#EMBEDDING_VECTOR_PATH=${OUTPUT_PATH}/${config}.vec
#EMBEDDING_VECTOR_PATH=/work/gglavas/data/word_embs/debie/utah-procrustes.vectors.npy
#EMBEDDING_VECTOR_PATH=${OUTPUT_PATH}/ft_postspec.vec
#EMBEDDING_VECTOR_PATH=/work/gglavas/data/word_embs/debie/utah-procrustes.vectors.npy
echo "FASTTEXT"
EMBEDDING_VECTOR_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vectors
#EMBEDDING_VOCAB_PATH=${OUTPUT_PATH}/ft_postspec.vocab
EMBEDDING_VOCAB_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab
#EMBEDDING_VOCAB_PATH=/home/anlausch/post-specialized-embeddings/postspec/ft_postspec.txt


for test_number in 1 2; do
    python single_target_weat.py \
        --test_number $test_number \
        --permutation_number 100000 \
        --output_file ${OUTPUT_PATH}/st_weat_${test_number}_orginal.txt \
        --lower True \
        --use_glove False \
        --embedding_vocab \
        ${EMBEDDING_VOCAB_PATH} \
        --embedding_vectors \
        ${EMBEDDING_VECTOR_PATH} |& tee ${OUTPUT_PATH}/st_weat_${test_number}_original.log
done;
echo "Glove"
#EMBEDDING_VECTOR_PATH=${OUTPUT_PATH}/drp=0.9_rf=0.5.vec
#EMBEDDING_VOCAB_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab

#EMBEDDING_VECTOR_PATH=${OUTPUT_PATH}/${config}.vec
#EMBEDDING_VECTOR_PATH=/work/gglavas/data/word_embs/debie/utah-procrustes.vectors.npy
EMBEDDING_VECTOR_PATH=${OUTPUT_PATH}/glove_200k.vec
#EMBEDDING_VECTOR_PATH=/work/gglavas/data/word_embs/debie/utah-procrustes.vectors.npy
#EMBEDDING_VECTOR_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vectors
EMBEDDING_VOCAB_PATH=${OUTPUT_PATH}/glove_200k.vocab
#EMBEDDING_VOCAB_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab
#EMBEDDING_VOCAB_PATH=/home/anlausch/post-specialized-embeddings/postspec/ft_postspec.txt


for test_number in 1 2; do
    python single_target_weat.py \
        --test_number $test_number \
        --permutation_number 100000 \
        --output_file ${OUTPUT_PATH}/st_weat_${test_number}_orginal.txt \
        --lower True \
        --use_glove False \
        --embedding_vocab \
        ${EMBEDDING_VOCAB_PATH} \
        --embedding_vectors \
        ${EMBEDDING_VECTOR_PATH} |& tee ${OUTPUT_PATH}/st_weat_${test_number}_original.log
done;
echo "W2V"
#EMBEDDING_VECTOR_PATH=${OUTPUT_PATH}/drp=0.9_rf=0.5.vec
#EMBEDDING_VOCAB_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab

#EMBEDDING_VECTOR_PATH=${OUTPUT_PATH}/${config}.vec
#EMBEDDING_VECTOR_PATH=/work/gglavas/data/word_embs/debie/utah-procrustes.vectors.npy
EMBEDDING_VECTOR_PATH=${OUTPUT_PATH}/w2v_cbow_200k.vec
#EMBEDDING_VECTOR_PATH=/work/gglavas/data/word_embs/debie/utah-procrustes.vectors.npy
#EMBEDDING_VECTOR_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vectors
EMBEDDING_VOCAB_PATH=${OUTPUT_PATH}/w2v_cbow_200k.vocab
#EMBEDDING_VOCAB_PATH=/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab
#EMBEDDING_VOCAB_PATH=/home/anlausch/post-specialized-embeddings/postspec/ft_postspec.txt


for test_number in 1 2; do
    python single_target_weat.py \
        --test_number $test_number \
        --permutation_number 100000 \
        --output_file ${OUTPUT_PATH}/st_weat_${test_number}_orginal.txt \
        --lower True \
        --use_glove False \
        --embedding_vocab \
        ${EMBEDDING_VOCAB_PATH} \
        --embedding_vectors \
        ${EMBEDDING_VECTOR_PATH} |& tee ${OUTPUT_PATH}/st_weat_${test_number}_original.log
done;