import torch
import numpy as np
from torch import nn
from collections import defaultdict


class WordEmbedder:

    def __init__(self, dim, path):
        self.embeddings = self.load_embedding(dim, path)
        self.size = len(self.embeddings)
        self.dim = dim

    def __len__(self):
        return self.size

    def __getitem__(self, word):
        return self.embeddings[word]

    @staticmethod
    def normalize(emb):
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def load_embedding(self, dim, path):
        # set zero vector as default embedding
        default_embedding = torch.zeros(dim)
        embeddings = defaultdict(lambda: default_embedding)

        # load embeddings from file
        with open(path, encoding='utf8') as file:
            for line in file.readlines():
                word, *embedding = line.split(' ')
                embedding = torch.as_tensor(list(map(float, embedding)))
                embeddings[word] = self.normalize(embedding)
        return embeddings


class CharDictEmbedder:

    def __init__(self, path):
        self.char_dict = self.load_char_dict(path)

    def __getitem__(self, word):
        return torch.as_tensor([self.char_dict[c] for c in word])

    @staticmethod
    def load_char_dict(path):
        vocab = [u'<unk>']
        with open(path, encoding='utf8') as file:
            vocab.extend(c.strip() for c in file)
        char_dict = defaultdict(int)
        char_dict.update({c: i for i, c in enumerate(vocab)})
        return char_dict


class CharCnnEmbedder(nn.Module):

    def __init__(self, config):
        super().__init__()

        # char vocab size + unk / emb size
        char_num, char_emb = config['char_num'], config['char_emb']
        self.char_emb = nn.Embedding(char_num, char_emb)

        # embedding size / number kernels / kernel width
        kernel_num, kernel_size = config['kernel_num'], config['kernel_size']
        self.cnns = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=char_emb, out_channels=kernel_num, kernel_size=kernel),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1),
        ) for kernel in kernel_size])

    def forward(self, char_index):
        # result dims: [num words, max_word_length]
        num_sents, num_words, num_chars = char_index.shape
        emb = self.char_emb(char_index.view(-1, num_chars)).transpose_(1, 2)

        # each emb_x: [num words, emb (50), 1]
        emb_0 = self.cnns[0](emb)
        emb_1 = self.cnns[1](emb)
        emb_2 = self.cnns[2](emb)

        # concatenate embeddings [num words, 3*emb (150), 1]
        char_emb = torch.cat((emb_0, emb_1, emb_2), 1)

        # restore sentences [num_sents, num_words, 3*emb (150)]
        return char_emb.view(num_sents, num_words, -1)
