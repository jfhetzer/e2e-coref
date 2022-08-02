import os
import sys
import json
import torch
from pathlib import Path
from pyhocon import ConfigFactory

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from model.model import Model

###
#    Calculate cosine similarity of BERT embeddings for two different languages/datasets
#    Used to review the effect of the adversarial cross-lingual training
###

CKPT = 'data/ckpt/bert_adv/ckpt_epoch-019.pt.tar'
CONF = 'bert-multilingual-base'

config = ConfigFactory.parse_file('./coref.conf')[CONF]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Model(config, device, device)

if CKPT:
    print('load from checkpoint')
    ckpt = torch.load(CKPT, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)

model.bert_model.to(device)
model.bert_model.eval()
path = Path(config['data_folder'])

num, sim = 0, 0
print('analyze emb similarity')
with torch.no_grad():
    with open(path / config['emb_src_data_file'], 'r') as f_src:
        with open(path / config['emb_tgt_data_file'], 'r') as f_tgt:
            for src, tgt in zip(f_src.readlines(), f_tgt.readlines()):
                src_sents = json.loads(src)['segments']
                tgt_sents = json.loads(tgt)['segments']
                src_emb = torch.unsqueeze(model.bert_model(src_sents).mean(dim=0), dim=0)
                tgt_emb = torch.unsqueeze(model.bert_model(tgt_sents).mean(dim=0), dim=0)
                cos_sim = torch.nn.functional.cosine_similarity(src_emb, tgt_emb)
                print(f'{num:05d}: {cos_sim[0]}')
                sim += cos_sim[0]
                num += 1

print(f'finished with avg. similarity: {sim/num}')
