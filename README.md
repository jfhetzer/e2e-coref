# BERT and SpanBERT for Coreference Resolution: PyTorch Implementation

Pytorch implementation of the *independent* model with BERT and SpanBERT proposed in
__[BERT for Coreference Resolution: Baselines and Analysis](https://arxiv.org/abs/1908.09091)__ and 
__[SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529)__.
 
This implementation is based upon the [original implementation](https://github.com/mandarjoshi90/coref) by the 
papers authors. The __model__ package is written from scratch, whereas some scripts in the __setup__ package and the 
entire __eval__ package are borrowed with almost no changes from the original implementation. Optimization for mention 
pruning inspired by [Ethan Yang](https://github.com/YangXuanyue/pytorch-e2e-coref). 


## Requirements
This project was written with Python 3.8.5 and PyTorch 1.7.1. For installation details regarding PyTorch please visit the 
official [website](https://pytorch.org/). Further requirements are listed in the __requirements.txt__ and can be 
installed via pip: <code>pip install -r requirements.txt</code> 


## Setup
> __Hint:__ Run setup.sh in an environment with Python 2.7 so the CoNLL-2012 scripts are executed by the correct interpreter 

To obtain all necessary data for training and evaluation run __setup.sh__ with the path to the 
[OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) folder (often named ontonotes-release-5.0).

e.g. <code>$ ./setup.sh /path/to/ontonotes-release-5.0</code>

Run <code>python setup/bert_tokenize.py -c \<conf\></code> to tokenize and segment the data before training, evaluating 
or testing that specific configuration. See [Training](#Training) for the available configurations.


## Training
Run the training with <code>python train.py -c \<config\> -f \<folder\> --cpu --amp --check --split</code>.Select with 
<code>conf</code> one of the four available configurations (bert-base, bert-large, spanbert-base, spanbert-large). The 
parameter <code>folder</code> names the folder the snapshots, taken during the training, are saved to. If the given 
folder already exists and contains at least one snapshot the training is restarted loading the latest snapshot. The 
optional flags <code>cpu</code> and <code>amp</code> can be set to train exclusively on the CPU or to use the automatic 
mixed precision training. Gradient checkpointing can be used with the option <code>check</code> to further reduce the 
GPU memory usage or the model can even be split up onto two GPUs with <code>split</code>.  


## Evaluation
Run the evaluation with <code>python evaluate.py -c \<conf\> -p \<pattern\> --cpu --amp --split</code>. All snapshots in the 
__data/ckpt__ folder that match the given <code>pattern</code> are evaluated. This works with simple snapshots (pt) as 
well as with snapshots with additional metadata (pt.tar). See [Training](#Training) for details on the remaining 
parameters.