import json
import torch
import argparse
from pathlib import Path
from pyhocon import ConfigFactory

from transformers import AutoTokenizer


class Document:

    def __init__(self, doc_key):
        self.doc_key = doc_key
        self.segments, self.subtokens = [], []
        self.speakers, self.speakers_tmp = [], []
        self.token_map, self.token_map_tmp = [], []
        self.sents_map, self.sents_map_tmp = [], []
        self.token_end = []
        self.sents_end = []
        self.clusters = []

    def get_dict(self):
        return {
            'doc_key': self.doc_key,
            'segments': self.segments,
            'sent_map': self.sents_map,
            'token_map': self.token_map,
            'speakers': self.speakers,
            'clusters': self.clusters
        }


class Tokenizer:

    def __init__(self, model, seg_len):
        self.doc = None
        self.seg_len = seg_len
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def tokenize(self, in_path, out_path):
        # create folder
        dir_path = Path(out_path).parent
        dir_path.mkdir(exist_ok=True)

        # read file document by document
        with open(in_path, 'r') as in_file:
            with open(out_path, "w") as out_file:
                for line in in_file:
                    raw_doc = json.loads(line)
                    doc_dict = self.tokenize_doc(raw_doc)
                    out_file.write(json.dumps(doc_dict))
                    out_file.write("\n")

    def tokenize_doc(self, raw_doc):
        self.doc = Document(raw_doc['doc_key'])
        text = raw_doc['sentences']
        idx = 0
        for s, sent in enumerate(text):
            for w, word in enumerate(sent):
                speaker = raw_doc['speakers'][s][w]
                self.add_word(idx, s, word, speaker)
                idx += 1
            self.doc.sents_end[-1] = True
        self.split()
        self.cluster(raw_doc['clusters'])
        self.token_map()
        return self.doc.get_dict()

    def add_word(self, idx, sent_idx, word, speaker):
        tokenized = self.tokenizer.tokenize(word)
        self.doc.subtokens += tokenized
        l = len(tokenized)
        self.doc.token_end += [False] * (l - 1) + [True]
        self.doc.sents_end += [False] * l
        self.doc.sents_map_tmp += [sent_idx] * l
        self.doc.token_map_tmp += [idx] * l
        self.doc.speakers_tmp += [speaker] * l

    def split(self):
        idx = 0
        max_len = len(self.doc.subtokens)
        while idx < max_len:
            # calculate splitting
            end = min(idx + self.seg_len - 2, max_len) - 1
            end_token = -1
            while end >= idx:
                if self.doc.sents_end[end]:
                    break
                if end_token == -1 and self.doc.token_end[end]:
                    end_token = end
                end -= 1
            end = (end_token if end < idx else end) + 1
            if end < idx:
                raise Exception('Token does not fit into segment')
            # apply splitting to text
            cls_tkn = self.tokenizer.cls_token
            sep_tkn = self.tokenizer.sep_token
            segment = [cls_tkn] + self.doc.subtokens[idx:end] + [sep_tkn]
            self.doc.segments.append(segment)
            # apply splitting to speakers
            speaker = ['[SPL]'] + self.doc.speakers_tmp[idx:end] + ['[SPL]']
            self.doc.speakers.append(speaker)
            # apply splitting to subtoken mapping
            mapping = ['[MPL]'] + self.doc.token_map_tmp[idx:end] + ['[MPL]']
            self.doc.token_map.extend(mapping)
            # apply splitting to sentence mapping
            first, last = self.doc.sents_map_tmp[idx], self.doc.sents_map_tmp[end-1]
            sents_map = [first] + self.doc.sents_map_tmp[idx:end] + [last+1]
            self.doc.sents_map.extend(sents_map)
            idx = end

    def cluster(self, clusters):
        # map original token index to start and end index of sub-token
        current = '[MPL]'
        sub_ends = []
        sub_starts = []
        for sub, token in enumerate(self.doc.token_map):
            if token == current:
                continue
            if token != '[MPL]':
                sub_starts.append(sub)
            if current != '[MPL]':
                sub_ends.append(sub - 1)
            current = token

        # adjust clusters to tokenized and split up sentences
        for cluster in clusters:
            for i, ment in enumerate(cluster):
                start = sub_starts[ment[0]]
                end = sub_ends[ment[1]]
                cluster[i] = [start, end]
        self.doc.clusters = clusters

        # sort clusters (only for comparison with tensorflow implementation)
        for cluster in self.doc.clusters:
            cluster.sort()
        self.doc.clusters.sort()

    def token_map(self):
        prev = 0
        for i, token in enumerate(self.doc.token_map):
            if token == '[MPL]':
                self.doc.token_map[i] = prev
            else:
                prev = token


def tokenize_for_width(width, folder, model, lang):
    tokenizer = Tokenizer(model, width)
    tokenizer.tokenize(f'data/data/train.{lang}.jsonlines', f'{folder}/train.{lang}.jsonlines')
    tokenizer.tokenize(f'data/data/dev.{lang}.jsonlines', f'{folder}/dev.{lang}.jsonlines')
    tokenizer.tokenize(f'data/data/test.{lang}.jsonlines', f'{folder}/test.{lang}.jsonlines')


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Tokenize and split documents into segments.')
    parser.add_argument('-c', metavar='CONF', default='bert-base', help='configuration (see coref.conf)')
    parser.add_argument('-l', metavar='LANG', default='english', help='language to tokenize')
    args = parser.parse_args()

    # load configuration
    config = ConfigFactory.parse_file('coref.conf')[args.c]

    # run segmentation and tokenization
    with torch.no_grad():
        tokenize_for_width(config['segm_size'], config['data_folder'], config['bert'], args.l)
