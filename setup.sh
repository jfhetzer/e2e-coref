#!/bin/bash

###################################### HELPER FUNCTIONS ######################################
function dlx() {
  wget $1/$2
  tar -xvzf $2
  rm $2
}

function compile_partition() {
    rm -f $2.$5.$3$4
    cat conll-2012/$3/data/$1/data/$5/annotations/*/*/*/*.$3$4 >> data/data/$2.$5.$3$4
}

function compile_language() {
    compile_partition development dev v4 _auto_conll $1
    compile_partition train train v4 _auto_conll $1
    compile_partition test test v4 _gold_conll $1
}


############################################ DATA ############################################
mkdir -p data
mkdir -p data/data
mkdir -p data/embs

# download and unzip conll data and scripts
conll_url=http://conll.cemantix.org/2012/download
dlx $conll_url conll-2012-train.v4.tar.gz
dlx $conll_url conll-2012-development.v4.tar.gz
dlx $conll_url/test conll-2012-test-key.tar.gz
dlx $conll_url/test conll-2012-test-official.v9.tar.gz
dlx $conll_url conll-2012-scripts.v3.tar.gz
dlx http://conll.cemantix.org/download reference-coreference-scorers.v8.01.tar.gz
mv reference-coreference-scorers conll-2012/scorer

# create data by combining conll with ontonotes data
bash conll-2012/v3/scripts/skeleton2conll.sh -D $1/data/files/data conll-2012

# move english data to data directory and remove needless information
compile_language english
python setup/minimize.py


######################################## CHECKPOINTS #########################################
mkdir -p data/ckpt