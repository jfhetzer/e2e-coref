# Coreference Resolution Systems: PyTorch Implementation

Pytorch implementation of the *independent* model with BERT and SpanBERT proposed in
__[BERT for Coreference Resolution: Baselines and Analysis](https://arxiv.org/abs/1908.09091)__ and 
__[SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529)__.

This implementation contains additional scripts and configurations I used in the context of my master's thesis. That 
includes the use of other pre-trained language models, training the model on various German datasets and improving its
performance for low resource languages leveraging transfer learning. For the vanilla versions of the fundamental 
coreference resolution models see the [Repository Overview](#repository-overview).

This implementation is based upon the [original implementation](https://github.com/mandarjoshi90/coref) by the papers 
authors. The __model__ and _misc__ packages are written from scratch, whereas some scripts in the __setup__ package and 
the entire __eval__ package are borrowed with almost no changes from the original implementation. Optimization for 
mention pruning inspired by [Ethan Yang](https://github.com/YangXuanyue/pytorch-e2e-coref). 

## Repository Overview
This repository contains three additional branches of different PyTorch models. These are reimplemantations of models
originally implemented with Tensorflow. The following tables gives an overview over the branches, the corresponding 
papers and original implementations.

|                               Branch                                |                                               Paper                                              |              Implementation           |
|:-------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------:|:----------------------------------------------:|
|  [e2e-coref](https://github.com/jfhetzer/e2e-coref/tree/e2e-coref)  |    [End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045)   | [GitHub](https://github.com/kentonl/e2e-coref/tree/e2e) |
|  [c2f-coref](https://github.com/jfhetzer/e2e-coref/tree/c2f-coref)  |    [Higher-order Coreference Resolution with Coarse-to-fine Inference](https://arxiv.org/abs/1804.05392)   | [GitHub](https://github.com/kentonl/e2e-coref) |
| [bert-coref](https://github.com/jfhetzer/e2e-coref/tree/bert-coref) |    [BERT for Coreference Resolution: Baselines and Analysis](https://arxiv.org/abs/1908.09091)   | [GitHub](https://github.com/mandarjoshi90/coref) |


## Requirements
This project was written with Python 3.8.5 and PyTorch 1.7.1. For installation details regarding PyTorch please visit the 
official [website](https://pytorch.org/). Further requirements are listed in the __requirements.txt__ and can be 
installed via pip: <code>pip install -r requirements.txt</code> 


## Setup
> __Hint:__ Run setup.sh in an environment with Python 2.7 so the CoNLL-2012 scripts are executed by the correct interpreter 

To obtain all necessary data for training, evaluation, and inference in English run __setup.sh__ with the path to the 
[OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) folder (often named ontonotes-release-5.0).

e.g. <code>$ ./setup.sh /path/to/ontonotes-release-5.0</code>

Run <code>python setup/bert_tokenize.py -c \<conf\></code> to tokenize and segment the data before training, evaluating 
or testing that specific configuration. See [coref.cong](coref.conf) for the available configurations.

The __misc__ folder contains scripts to convert the German datasets Tüba-D/Z v10/11, SemEval-2010 and DIRNDL into the 
required CoNLL-2012 format. For the SemEval-2010 make sure to remove the singletons in order to get comparable results. 
Use [minimize.py](setup/minimize.py) and [bert_tokenize.py](setup/bert_tokenize.py) to obtain the desired file to train 
with.


## Training
Run the training with <code>python train.py -c \<config\> -f \<folder\> --cpu --amp --check --split</code>.Select with 
<code>conf</code> one of the four available configurations (bert-base, bert-large, spanbert-base, spanbert-large). The 
parameter <code>folder</code> names the folder the snapshots, taken during the training, are saved to. If the given 
folder already exists and contains at least one snapshot the training is restarted loading the latest snapshot. The 
optional flags <code>cpu</code> and <code>amp</code> can be set to train exclusively on the CPU or to use the automatic 
mixed precision training. Gradient checkpointing can be used with the option <code>check</code> to further reduce the 
GPU memory usage or the model can even be split up onto two GPUs with <code>split</code>.  

### Fine-tuning on German
To fine-tune multilingual models trained on the OntoNotes 5.0 dataset on German datasets, adapt the configuration in 
[coref.conf](coref.conf) and place the latest snapshot into the same folder the fine-tuned model should write its 
snapshots to. Then start training as described above.

For easier parameter tuning use [train_fine.py](train_fine.py) and write a shell script to programmatically pass the 
learning rates and epochs into the training.

### Adversarial Cross-lingual Training
To redo the adversarial training described in the thesis run <code>python train_adv.py</code>. The only configuration 
setup for this training is the *bert-multilingual-base*. Make sure to have created the *adv_data_file* besides the 
English data before training.

To evalute the trained model on German comment in the desired dataset in the [coref.conf](coref.conf). For validate if 
the training brought English and German embeddings closer together as desired use the 
[analyze_emb_similarity.py](misc/analyze_emb_similarity.py) script.


## Evaluation
Run the evaluation with <code>python evaluate.py -c \<conf\> -p \<pattern\> --cpu --amp --split</code>. All snapshots in the 
__data/ckpt__ folder that match the given <code>pattern</code> are evaluated. This works with simple snapshots (pt) as 
well as with snapshots with additional metadata (pt.tar). See [Training](#Training) for details on the remaining 
parameters.

To evaluate previous predictions dumped during training or evaluation use the 
[eval_dumped_preds.py](misc/eval_dumped_preds.py) script.