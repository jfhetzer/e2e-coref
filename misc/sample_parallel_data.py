import random

###
#    Used for analyzing embedding similarity leveraging adversarial learning
#    WMT17 dataset has around 1.4 mio 4 sentence documents which is way too large for some analysis
#    Randomly selects 10k documents of the parallel WMT17 training corpus
###

EN_FILE = 'data/data/emb.wmt17_en.jsonlines'
DE_FILE = 'data/data/emb.wmt17_de.jsonlines'

EN_FILE_OUT = 'data/data/emb.wmt17_10k_en.jsonlines'
DE_FILE_OUT = 'data/data/emb.wmt17_10k_de.jsonlines'

num_lines_en = sum(1 for _ in open(EN_FILE))
num_lines_de = sum(1 for _ in open(DE_FILE))

assert num_lines_en == num_lines_de

selection = random.sample(range(0, num_lines_en), 10000)

lines_en = [line for idx, line in enumerate(open(EN_FILE)) if idx in selection]
open(EN_FILE_OUT, 'w').writelines(lines_en)

lines_de = [line for idx, line in enumerate(open(DE_FILE)) if idx in selection]
open(DE_FILE_OUT, 'w').writelines(lines_de)
