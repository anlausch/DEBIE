# DEBIE: A General Framework for Implicit and Explicit Debiasing of Distributional Word Vector Spaces (AAAI2020)

## Paper Abstract

Distributional word vectors have recently been shown to encode many of the human biases, most notably gender and racial biases, and models for attenuating such biases have consequently been proposed. However, existing models and studies (1) operate on under-specified and mutually differing bias definitions, (2) are tailored for a particular bias (e.g., gender bias) and (3) have been evaluated inconsistently and non-rigorously. In this work, we introduce a general framework for debiasing word embeddings. We operationalize the definition of a bias by discerning two types of bias specification: explicit and implicit. We then propose three debiasing models that operate on explicit or implicit bias specifications, and that can be composed towards more robust debiasing. Finally, we devise a full-fledged evaluation framework in which we couple existing bias metrics with newly proposed ones. Experimental findings across three embedding methods suggest that the proposed debiasing models are robust and widely applicable: they often completely remove the bias both implicitly and explicitly, without degradation of semantic information encoded in any of the input distributional spaces. Moreover, we successfully transfer debiasing models, by means of crosslingual embedding spaces, and remove or attenuate biases in distributional word vector spaces of languages that lack readily available bias specifications.

## Citation

```
@inproceedings{lauscher2019general,
  title={A general framework for implicit and explicit debiasing of distributional word vector spaces},
  author={Lauscher, Anne and Glava{\v{s}}, Goran and Ponzetto, Simone Paolo and Vuli{\'c}, Ivan},
  booktitle={Accepted for publication at the Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI 2020)},
  year={2019}
}
```

[ArXive Link](https://arxiv.org/abs/1909.06092)
-----------------------
## Update: Demo Notebooks
We recommend starting by looking at one of our example Colab Notebooks.
- A notebook demonstrating the use of ECT, BAT, and KMeans++ for reproducing our scores on English fastText can be found here: https://colab.research.google.com/drive/1C05uSYIK1Qn02G2f02IGfUF3J-Hxix0I
- A notebook demonstrating the use of GBDD on fastText and the reproduction of our scores for ECT, BAT, KMeans++ can be found here:
https://colab.research.google.com/drive/1ysT5RIODmqwMhtxWCpE_0xr_q3f1W57V

Please don't hesitate in case you have any questions. We will happily extend the examples in case there are any requests.

------------------------
## Repository Description

This code contains all code needed to reproduce the experiments and results reported in our paper.

### Augmentation of the test specifications

Test specifications for English WEAT 1,2,8, and 9 are given in `./data`.
The automatic augmentation using a post-specialized embedding space is implemented in `augmentation.py`. An example call is given in `run_augmentation.sh`.
In order to reproduce the results, you'll need to download a post-specialized embedding space, which we do not provide in this repository.
Example augmentations can be found in `./data`, e.g., `./data/weat_2_aug_postspec_2.txt`.

### Debiasing Models

#### DebiasNet

The model code for DebiasNet is given in `debiasnet.py`. The preparation of the input data using the augmented
test specifications is implemented in `data_handler.py` and the training code can be found in `exp_debiasnet_args.py`.
The bash script `run_debiasnet.sh` demonstrates how to start the training.
Just running the prediction, i.e., the debiasing, with a pretrained debiasing model is demonstrated in
`exp_debiasnet_args_predict_only.py` and an example call can be found in `run_debiasnet_prediction.sh`.

#### BAM
BAM is implemented in `bam.py`. The code for running the experiments with BAM together with GBDD and the final DebiasNet
evaluation can be found in `exp_gbdd_bam.py`.

#### GBDD
The code for GBDD can be found in `gbdd.py`. The code for running the experiments with BAM together with GBDD and the final DebiasNet
evaluation can be found in `exp_gbdd_bam.py`.

### Evaluation Framework

#### Structure

All code related to the evaluation framework can be found in `./evaluation`. The script `./evaluation/eval.py` contains
functions for running the following tests:

- Implicit Bias Tests

        - ... clustering with KMeans++
        - ... classification with SVM

- Explicit Bias Tests

        - Embedding Coherence Test
        - Bias Analogy Test

- Semantic Quality Evaluation

        - SimLex-999 (Test data not provided in this repo)
        - WordSim-543 (Test data not provided in this repo)

XWEAT (explicit bias test): For the full code of XWEAT we refer to [XWEAT Repo Link](https://github.com/umanlp/XWEAT).
However, in `exp_gbdd_bam.py` we have copied the test specifications needed to perform the tests used in the paper.

#### Usage of the Evaluation Framework
(1) Decide on bias specifications to use: Either write them on your own or use predefined bias specifications, e.g., `DEBIE/DATA/weat_8.txt`

(2) All evaluation tests are functions in `eval.py` except for WEAT. The usage is demonstrated in https://github.com/anlausch/DEBIE/blob/master/exp_gbdd_bam.py (but don't be confused as it also demonstrates the debiasing with BAM and GBDD): 

- lines 21-38: load a specification
- lines 45-84: load embedding spaces (we expect vocab to be a dict mapping words to indices and vecs is a numpy matrix containing word embeddings such that for instance `vocab["house"]` returns the index for the vector of house, which can be accessed with `vecs[index]`)
- if you want to load XWEAT test translations (for the predefined tests): lines 106-129 (you have to adapt the path to the translations)
- Tests
    - BAT: lines 223-227
    - WEAT: lines 233-243
    - KMeans: lines 250-254
    - ECT: lines 261-265
    - Simlex: loading the evaluation data set line 94, usage line 273 -276
    -  WordSim: loading the evaluation data set line 99, usage line 296-299
    - SVM: lines 211-214; SVM needs training data, which we produce by automatically augmenting the test with a similarity specialiced word embedding space; this is demonstrated in run_augmentation.sh

Don't hesitate to contact us in case you have questions.
