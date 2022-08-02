import math
import torch
import torch.nn as nn
from torch.functional import F

from model.modules import CustomLSTM, Scorer
from model.embeddings import CharCnnEmbedder


class Model(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.dropout = self.config['dropout']
        self.max_width = self.config['max_ment_width']

        # word/document embedding
        self.char_emb = CharCnnEmbedder(self.config)
        word_emb_size = self.config['word_emb']
        lstm_hidden_size = self.config['lstm_hidden_size']
        self.lstm = CustomLSTM(word_emb_size, lstm_hidden_size, dropout=self.dropout)

        # feature embeddings
        bin_width = self.config['bin_widths']
        feature_size = self.config['feature_size']
        genre_num = len(self.config['genres'])
        self.speaker_embds = nn.Embedding(2, feature_size)
        self.genre_emb = nn.Embedding(genre_num, feature_size)
        self.ant_dist_emb = nn.Embedding(len(bin_width), feature_size)
        self.mention_width_emb = nn.Embedding(self.max_width, feature_size)

        # scorer for mentions and antecedents
        hidden_size = self.config['hidden_size']
        ment_emb_size = 4 * lstm_hidden_size + feature_size + word_emb_size
        ante_emb_size = 3 * ment_emb_size + 3 * feature_size
        self.mention_scorer = Scorer(input_size=ment_emb_size, hidden_size=hidden_size, dropout=self.dropout)
        self.antecedent_scorer = Scorer(input_size=ante_emb_size, hidden_size=hidden_size, dropout=self.dropout)

        # bi-directional lstm output / 1
        self.span_head = nn.Linear(2 * lstm_hidden_size, 1)

        # bins for antecedent distance embedding
        # [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+]
        self.bins = []
        for i, w in enumerate(bin_width):
            self.bins.extend([i] * w)
        self.bins = torch.as_tensor(self.bins, dtype=torch.long, device=self.device)

    def ment_embedding(self, flat_doc_embs, flat_lstm_out, ment_start, ment_ends):
        # get representation for start and end of mention
        start_embs = flat_lstm_out[ment_start]
        end_embs = flat_lstm_out[ment_ends]

        # get mention width embedding
        width = ment_ends - ment_start + 1
        width_size, = width.shape
        width_embs = self.mention_width_emb(width - 1)
        width_embs = torch.dropout(width_embs, self.dropout, self.training)

        # get head representation
        doc_len = len(flat_doc_embs)
        span_idxs = torch.clamp(
            torch.arange(self.max_width, device=self.device).view(1, -1) + ment_start.view(-1, 1), max=doc_len - 1)
        ment_doc_embs = flat_doc_embs[span_idxs]
        # max mention width 10
        span_head_emb = self.span_head(flat_lstm_out)
        span_head_emb = torch.squeeze(span_head_emb)
        span_head_emb = span_head_emb[span_idxs]
        # create mask for different mention widths
        ment_mask = torch.arange(self.max_width, device=self.device).view(1, -1).repeat(width_size, 1) < width.view(-1, 1)
        # transform mask 0 -> -inf / 1 -> 0 for softmax and add to embeddings
        span_head_emb += torch.log(ment_mask.float())
        attention = F.softmax(span_head_emb, dim=1).view(-1, self.max_width, 1)
        # calculate final head embedding as weighted sum
        head_embs = (attention * ment_doc_embs).sum(dim=1)

        # combine different embeddings to single mention embedding
        # warning: different order than proposed in the paper
        return torch.cat((start_embs, end_embs, width_embs, head_embs), dim=1)

    def prune_mentions(self, ment_starts, ment_ends, ment_scores, k):
        # get mention indices sorted by the mention score
        ment_idx = ment_scores.argsort(descending=True)

        # map current top indices to start and end index with the maximal spanning width
        # optimization from: https://github.com/YangXuanyue/pytorch-e2e-coref
        max_span_start, max_span_end = {}, {}

        # iterate over sorted mentions and build up top mentions
        top_ment_idx = []
        for idx in ment_idx:
            start_idx = ment_starts[idx].item()
            end_idx = ment_ends[idx].item()

            # check if mentions partially overlap with better scoring mention
            for i in range(start_idx, end_idx + 1):
                if i in max_span_end and i > start_idx and end_idx < max_span_end[i]:
                    break
                if i in max_span_start and i < end_idx and start_idx > max_span_start[i]:
                    break
            else:
                # add mention if it does not overlap
                top_ment_idx.append(idx)
                if start_idx not in max_span_end or end_idx > max_span_end[start_idx]:
                    max_span_end[start_idx] = end_idx
                if end_idx not in max_span_start or start_idx < max_span_start[end_idx]:
                    max_span_start[end_idx] = start_idx

                # stop when k indices added
                if k == len(top_ment_idx):
                    break

        # find mention order by start and end indices
        top_ment_idx = torch.as_tensor(top_ment_idx)
        ment_pos = ment_starts[top_ment_idx] * ment_ends[-1] + ment_ends[top_ment_idx]
        order = ment_pos.argsort()
        # sort top indices and create tensor
        return top_ment_idx[order]

    def get_antecedents(self, ment_starts, ment_ends, gold_starts, gold_ends, cluster_ids):
        # map mention index to gold cluster id
        cluster_map = {}
        m, g = 0, 0
        m_len = len(ment_starts)
        g_len = len(gold_starts)
        max_ante = min(m_len, self.config['max_ment_dist'])
        # mention and gold boundaries are sorted
        while m < m_len and g < g_len:
            m_s, m_e = ment_starts[m], ment_ends[m]
            g_s, g_e = gold_starts[g], gold_ends[g]
            if m_s == g_s and m_e == g_e:
                cluster_map[m] = cluster_ids[g]
                m, g = m + 1, g + 1
            elif m_s < g_s or (m_s == g_s and m_e < g_e):
                m += 1
            elif m_s > g_s or (m_s == g_s and m_e > g_e):
                g += 1

        # number of antecedent candidates per mention
        antes_len = torch.arange(m_len).clamp(max=max_ante)

        # map antecedent position to mention index
        antes = torch.arange(max_ante).view(1, -1) - torch.zeros(m_len).view(-1, 1)
        antes = antes.long().tril(diagonal=-1)
        antes[max_ante:] += torch.arange(m_len - max_ante).view(-1, 1)

        # map antecedents to clusters
        ment_clusters = torch.as_tensor([cluster_map.get(i, -1) for i in range(m_len)], device=self.device)
        cluster_block1 = ment_clusters[:max_ante].view(1, -1).repeat(max_ante, 1)
        cluster_block2 = ment_clusters.as_strided(tuple([m_len-max_ante, max_ante]), tuple([1, 1]))
        clusters = torch.cat((cluster_block1, cluster_block2))

        # create labels from mappings
        ment_clusters = torch.as_tensor([cluster_map.get(i, -2) for i in range(m_len)], device=self.device)
        labels = clusters == ment_clusters.view(-1, 1)
        labels = labels.tril(diagonal=-1)

        # add labels for dummy antecedents
        dummy_antes = ~labels.any(dim=1).view(-1, 1)
        labels = torch.cat((dummy_antes, labels), dim=1)

        return antes, antes_len, labels

    def antecedent_embedding(self, ment_emb, antecendets, genre_id, ment_speakers):
        # build up input of shape [num_ment, max_ante, emb]
        num_ment, max_ante = antecendets.shape

        # genre embedding
        genre_emb = self.genre_emb(genre_id)
        genre_emb = genre_emb.view(1, 1, -1).repeat(num_ment, max_ante, 1)

        # same speaker embedding
        ante_speakers = ment_speakers[antecendets]
        same_speaker = torch.eq(ment_speakers.view(-1, 1), ante_speakers)
        speaker_emb = self.speaker_embds(same_speaker.long())

        # antecedent distance embedding
        ante_dist = torch.arange(num_ment).view(-1, 1) - antecendets
        ante_dist_bin = self.bins[torch.clamp(ante_dist, max=64)]
        ante_dist_emb = self.ant_dist_emb(ante_dist_bin)

        # apply dropout to feature embeddings
        feature_emb = torch.cat((speaker_emb, genre_emb, ante_dist_emb), dim=2)
        feature_emb = torch.dropout(feature_emb, self.dropout, self.training)

        # antecedent embedding / mention embedding / similarity
        antecedent_emb = ment_emb[antecendets]
        ment_emb = ment_emb.view(num_ment, 1, -1).repeat(1, max_ante, 1)
        similarity_emb = antecedent_emb * ment_emb

        return torch.cat((ment_emb, antecedent_emb, similarity_emb, feature_emb), dim=2)

    def forward(self, word_emb, char_index, sent_len, genre_id, speaker_ids, gold_starts, gold_ends, cluster_ids, cand_starts, cand_ends):
        # create sentence mask to flatten tensors
        sent_num, max_sent_len, _ = word_emb.shape
        sent_mask = torch.arange(max_sent_len).view(1, -1).repeat(sent_num, 1) < torch.as_tensor(sent_len).view(-1, 1)

        # cnn char embedder
        char_emb = self.char_emb(char_index.to(self.device))

        # concat word embeddings with char embeddings
        doc_embs = torch.cat((word_emb.to(self.device), char_emb), dim=2)
        doc_embs = torch.dropout(doc_embs, self.config['dropout_lexical'], self.training)

        lstm_out = self.lstm(doc_embs, sent_len)
        flat_lstm_out = lstm_out[sent_mask]
        # dropout additional to reccurent dropout in hidden state
        output = torch.dropout(flat_lstm_out, self.dropout, self.training)

        flat_doc_embs = doc_embs[sent_mask]
        cand_embs = self.ment_embedding(flat_doc_embs, output, cand_starts.to(self.device), cand_ends.to(self.device))
        # get mention scores and prune to get top scores
        cand_scores = self.mention_scorer(cand_embs).squeeze()
        k = math.floor(flat_lstm_out.shape[0] * self.config['ment_ratio'])
        top_ment_idx = self.prune_mentions(cand_starts, cand_ends, cand_scores, k)
        # filter mention candidates for top scoring mentions
        ment_starts = cand_starts[top_ment_idx]
        ment_ends = cand_ends[top_ment_idx]
        ment_embs = cand_embs[top_ment_idx]
        ment_scores = cand_scores[top_ment_idx]
        ment_speakers = speaker_ids[ment_starts]

        # get antecedents, distance between mentions and the final labels
        antecedents, antecedents_len, labels \
            = self.get_antecedents(ment_starts, ment_ends, gold_starts, gold_ends, cluster_ids)

        antecedents_emb = self.antecedent_embedding(ment_embs, antecedents, genre_id.to(self.device), ment_speakers.to(self.device))
        antecedents_scores = self.antecedent_scorer(antecedents_emb).squeeze()

        # apply mask to remove padding from antecedent scores
        num_mention = len(ment_starts)
        max_antecedents = min(self.config['max_ment_dist'], num_mention)
        antecedent_mask = torch.arange(max_antecedents, device=self.device).view(1, -1).repeat(num_mention, 1) < \
                          torch.as_tensor(antecedents_len, device=self.device).view(-1, 1)
        antecedents_scores += torch.log(antecedent_mask.float())

        # combine mention and antecedent score to final co-reference score
        coref_score = antecedents_scores + ment_scores.view(-1, 1) + ment_scores[antecedents]
        # append dummy score
        coref_score = torch.cat((torch.zeros(num_mention, 1, device=self.device), coref_score.float()), dim=1)

        return coref_score, labels, antecedents, ment_starts, ment_ends, cand_scores
