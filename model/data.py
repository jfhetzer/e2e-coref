import torch
import json
import random
import itertools
from torch.utils import data
from model.embeddings import WordEmbedder, CharDictEmbedder


class Dataset(data.Dataset):

    def __init__(self, config, data_path, training=True):
        self.config = config
        self.training = training
        self.data = []
        with open(data_path, encoding='utf8') as file:
            for line in file:
                self.data.append(json.loads(line))
        self.size = len(self.data)
        self.genres = {g: i for i, g in enumerate(self.config['genres'])}
        self.emb_glove = WordEmbedder(300, self.config['glove_emb'])
        self.emb_turian = WordEmbedder(50, self.config['turian_emb'])
        self.emb_char = CharDictEmbedder(self.config['char_vocab'])

    def get_raw_data(self, item):
        return self.data[item]

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        # get requested document
        doc = self.data[item]
        # truncate in training if document is too long
        max_sent_num = self.config['max_sent_num']
        if self.training and len(doc['sentences']) > max_sent_num:
            doc = self.truncate(doc, max_sent_num)

        # calculate some auxiliary variables
        sents = doc['sentences']
        sent_num = len(sents)
        sent_len = [len(s) for s in sents]
        word_len = [[len(w) for w in s] for s in sents]
        max_sent_len = max(sent_len)
        max_word_len = max(itertools.chain.from_iterable(word_len))

        # apply embeddings and mappings on word and character level
        glove_size, turian_size = self.config['glove_size'], self.config['turian_size']
        word_emb = torch.zeros(sent_num, max_sent_len, glove_size + turian_size)
        char_index = torch.zeros(sent_num, max_sent_len, max_word_len, dtype=torch.long)
        for i, sent in enumerate(sents):
            for j, word in enumerate(sent):
                word_emb[i, j, :glove_size] = self.emb_glove[word]
                word_emb[i, j, glove_size:] = self.emb_turian[word]
                char_index[i, j, :len(word)] = self.emb_char[word]

        # genre-id
        genre_id = torch.as_tensor([self.genres[doc['doc_key'][:2]]])

        # speaker-ids
        speakers = list(itertools.chain.from_iterable(doc['speakers']))
        speaker_dict = {s: i for i, s in enumerate(set(speakers))}
        speaker_ids = torch.as_tensor([speaker_dict[s] for s in speakers])

        # read cluster and mentions
        clusters = doc['clusters']
        gold_mentions = sorted(tuple(m) for m in itertools.chain.from_iterable(clusters))
        gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}

        # gold-starts / gold-ends
        gold_starts, gold_ends = map(
            torch.as_tensor,
            zip(*gold_mentions) if gold_mentions else ([], [])
        )

        # cluster-ids
        cluster_ids = torch.zeros(len(gold_mentions))
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id

        # cand-starts / cand-ends
        cand_starts, cand_ends = self.create_candidates(sent_len)

        # return all necessary information for training and evaluation
        return word_emb, char_index, sent_len, genre_id, speaker_ids, gold_starts, gold_ends, cluster_ids, cand_starts, cand_ends

    def truncate(self, doc, max_sent_num):
        sents = doc['sentences']
        sent_num = len(sents)
        sent_len = [len(s) for s in sents]

        # calculated borders for truncation
        sentence_start = random.randint(0, sent_num - max_sent_num)
        sentence_end = sentence_start + max_sent_num
        word_start = sum(sent_len[:sentence_start])
        word_end = sum(sent_len[:sentence_end])

        # update mention indices and remove truncated mentions
        clusters = [[
            (l - word_start, r - word_start)
            for l, r in cluster
            if word_start <= l <= r < word_end
        ] for cluster in doc['clusters']]
        # remove empty clusters
        clusters = [cluster for cluster in clusters if cluster]

        # return truncated document with only the necessary information
        return {
            'sentences': sents[sentence_start:sentence_end],
            'clusters': clusters,
            'speakers': doc['speakers'][sentence_start:sentence_end],
            'doc_key': doc['doc_key']
        }

    def create_candidates(self, sent_len):
        # calculate all possible mentions
        max_ment_width = self.config['max_ment_width']
        cand_starts, cand_ends = [], []
        offset = 0
        for s in sent_len:
            for i in range(s):
                width = min(max_ment_width, s-i)
                start = i + offset
                cand_starts.extend([start] * width)
                cand_ends.extend(range(start, start+width))
            offset += s

        # return candidate boundaries as tensors
        cand_starts = torch.as_tensor(cand_starts)
        cand_ends = torch.as_tensor(cand_ends)
        return cand_starts, cand_ends


class DataLoader(data.DataLoader):

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, collate_fn=DataLoader.collate, batch_size=1, pin_memory=True, **kwargs)

    @staticmethod
    def collate(batch):
        return batch[0]