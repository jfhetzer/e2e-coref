import math
import torch
import torch.nn as nn
from torch.functional import F

from model.modules import HighwayLSTM, Scorer, ElmoAggregator
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
        self.aggregator = ElmoAggregator()
        word_emb_size = self.config['word_emb']
        lstm_hidden_size = self.config['lstm_hidden_size']
        lstm_dropout = self.config['lstm_dropout']
        self.lstm = HighwayLSTM(word_emb_size, lstm_hidden_size, dropout=lstm_dropout)

        # mention embedding
        self.span_head = nn.Linear(2 * lstm_hidden_size, 1)

        # feature embeddings
        bin_width = self.config['bin_widths']
        feature_size = self.config['feature_size']
        genre_num = len(self.config['genres'])
        self.speaker_embds = nn.Embedding(2, feature_size)
        self.genre_embeds = nn.Embedding(genre_num, feature_size)
        self.ant_dist_embeds = nn.Embedding(len(bin_width), feature_size)
        self.mention_width_embeds = nn.Embedding(self.max_width, feature_size)

        # scorer for mentions and antecedents
        hidden_size, head_size = self.config['hidden_size'], self.config['head_size']
        char_emb_size = len(self.config['kernel_size']) * self.config['kernel_num']
        ment_emb_size = 4 * lstm_hidden_size + feature_size + head_size + char_emb_size
        ante_emb_size = 3 * ment_emb_size + 3 * feature_size
        self.mention_scorer = Scorer(input_size=ment_emb_size, hidden_size=hidden_size, dropout=self.dropout)
        self.slow_antecedent_scorer = Scorer(input_size=ante_emb_size, hidden_size=hidden_size, dropout=self.dropout)
        self.fast_antecedent_scorer = nn.Linear(ment_emb_size, ment_emb_size)
        self.attended_gate = nn.Sequential(
            nn.Linear(2 * ment_emb_size, ment_emb_size),
            nn.Sigmoid()
        )

        # bins for antecedent distance embedding
        # [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+]
        self.bins = []
        for i, w in enumerate(bin_width):
            self.bins.extend([i] * w)
        self.bins = torch.as_tensor(self.bins, dtype=torch.long, device=self.device)

    def ment_embedding(self, head_doc_embs, flat_lstm_out, ment_start, ment_ends):
        # get representation for start and end of mention
        start_embs = flat_lstm_out[ment_start]
        end_embs = flat_lstm_out[ment_ends]

        # get mention width embedding
        width = ment_ends - ment_start + 1
        width_size, = width.shape
        width_embs = self.mention_width_embeds(width - 1)
        width_embs = torch.dropout(width_embs, self.dropout, self.training)

        # get head representation
        doc_len = len(head_doc_embs)
        span_idxs = torch.clamp(
            torch.arange(self.max_width, device=self.device).view(1, -1) + ment_start.view(-1, 1), max=doc_len - 1)
        ment_doc_embs = head_doc_embs[span_idxs]
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

    def get_fast_antecedents(self, ment_emb, ment_scores, c):
        ment_range = torch.arange(ment_emb.shape[0], device=self.device)
        ante_offset = ment_range.view(-1, 1) - ment_range.view(1, -1)
        ante_mask = ante_offset > 0

        # calculate bilinear antecedent scoring (s_c)
        src_ment_emb = torch.dropout(self.fast_antecedent_scorer(ment_emb), 0.5, self.training)
        tgt_ment_emb = torch.dropout(ment_emb, 0.5, self.training)
        fast_ante_scores = src_ment_emb @ tgt_ment_emb.t()

        # calculate three factor antecedent scoring (s_m + s_m + s_c)
        fast_ante_scores += ment_scores.view(-1, 1) + ment_scores.view(1, -1)
        fast_ante_scores += torch.log(ante_mask.float())

        # get top antes and prune
        _, top_antes = torch.topk(fast_ante_scores, c, sorted=True)
        # top_antes = np.load('antes.npy')
        top_ante_mask = ante_mask[ment_range.view(-1, 1), top_antes]
        top_fast_ante_scores = fast_ante_scores[ment_range.view(-1, 1), top_antes]
        top_ante_offset = ante_offset[ment_range.view(-1, 1), top_antes]
        return top_antes, top_ante_mask, top_fast_ante_scores, top_ante_offset

    def antecedent_embedding(self, ment_emb, antecendets, genre_id, ment_speakers):
        # build up input of shape [num_ment, max_ante, emb]
        num_ment, max_ante = antecendets.shape

        # genre embedding
        genre_emb = self.genre_embeds(genre_id)
        genre_emb = genre_emb.view(1, 1, -1).repeat(num_ment, max_ante, 1)

        # same speaker embedding
        ante_speakers = ment_speakers[antecendets]
        same_speaker = torch.eq(ment_speakers.view(-1, 1), ante_speakers)
        speaker_emb = self.speaker_embds(same_speaker.long())

        # antecedent distance embedding
        ante_dist = torch.arange(num_ment, device=self.device).view(-1, 1) - antecendets
        ante_dist_bin = self.bins[torch.clamp(ante_dist, min=0, max=64)]
        ante_dist_emb = self.ant_dist_embeds(ante_dist_bin)

        # apply dropout to feature embeddings
        feature_emb = torch.cat((speaker_emb, genre_emb, ante_dist_emb), dim=2)
        feature_emb = torch.dropout(feature_emb, self.dropout, self.training)

        # antecedent embedding / mention embedding / similarity
        antecedent_emb = ment_emb[antecendets]
        ment_emb = ment_emb.view(num_ment, 1, -1).repeat(1, max_ante, 1)
        similarity_emb = antecedent_emb * ment_emb

        return torch.cat((ment_emb, antecedent_emb, similarity_emb, feature_emb), dim=2)

    def get_labels(self, ment_starts, ment_ends, gold_starts, gold_ends, cluster_ids, antes, ante_mask):
        if len(gold_starts) == 0:
            # dummy for no gold clusters
            ment_clusters = torch.zeros(ment_starts.shape, device=self.device)
        else:
            # set label/cluster-id per mention
            same_starts = gold_starts.view(-1, 1) == ment_starts.view(1, -1)
            same_ends = gold_ends.view(-1, 1) == ment_ends.view(1, -1)
            same_ment = (same_starts & same_ends).to(self.device)
            cluster_ids = cluster_ids.to(self.device).view(1, -1)
            ment_clusters = cluster_ids @ same_ment.type(cluster_ids.dtype)
            ment_clusters = ment_clusters.squeeze()

        # build up labels
        antes_cluster_id = ment_clusters[antes]
        antes_cluster_id += torch.log(ante_mask.float())
        labels = antes_cluster_id == ment_clusters.view(-1, 1)
        not_dummy = (ment_clusters > 0).view(-1, 1)
        labels = labels & not_dummy
        dummy_labels = ~labels.any(dim=1, keepdim=True)
        return torch.cat((dummy_labels, labels), dim=1)

    def forward(self, cont_emb, head_emb, elmo_emb, char_index, sent_len, genre_id, speaker_ids, gold_starts, gold_ends, cluster_ids, cand_starts, cand_ends):
        # create sentence mask to flatten tensors
        sent_num, max_sent_len, _ = cont_emb.shape
        sent_mask = torch.arange(max_sent_len).view(1, -1).repeat(sent_num, 1) < torch.as_tensor(sent_len).view(-1, 1)

        # cnn char embedder
        char_emb = self.char_emb(char_index.to(self.device))

        # get elmo embedding
        elmo_agg = self.aggregator(elmo_emb.to(self.device))

        # concat embeddings
        cont_doc_emb = torch.cat((cont_emb.to(self.device), char_emb, elmo_agg), dim=2)
        head_doc_emb = torch.cat((head_emb.to(self.device), char_emb), dim=2)
        cont_doc_emb = torch.dropout(cont_doc_emb, self.config['dropout_lexical'], self.training)
        head_doc_emb = torch.dropout(head_doc_emb, self.config['dropout_lexical'], self.training)

        lstm_out = self.lstm(cont_doc_emb, sent_len)
        output = lstm_out[sent_mask]

        head_doc_embs = head_doc_emb[sent_mask]
        cand_embs = self.ment_embedding(head_doc_embs, output, cand_starts.to(self.device), cand_ends.to(self.device))
        # get mention scores and prune to get top scores
        cand_scores = self.mention_scorer(cand_embs).squeeze()
        k = math.floor(output.shape[0] * self.config['ment_ratio'])
        top_ment_idx = self.prune_mentions(cand_starts, cand_ends, cand_scores, k)
        # filter mention candidates for top scoring mentions
        ment_starts = cand_starts[top_ment_idx]
        ment_ends = cand_ends[top_ment_idx]
        ment_embs = cand_embs[top_ment_idx]
        ment_scores = cand_scores[top_ment_idx]
        ment_speakers = speaker_ids[ment_starts]

        # get top c antecedents per mention by fast scoring
        c = min(self.config['max_antes'], k)
        antes, ante_mask, fast_scores, ante_offset = self.get_fast_antecedents(ment_embs, ment_scores, c)

        # iterative inference procedure
        dummy_scores = torch.zeros(k, 1, device=self.device, dtype=fast_scores.dtype)
        for i in range(self.config['coref_depth']):
            # get full scores for antecedents
            ante_ment_emb = ment_embs[antes]
            ante_emb = self.antecedent_embedding(ment_embs, antes, genre_id.to(self.device), ment_speakers.to(self.device))
            ante_scores = fast_scores + self.slow_antecedent_scorer(ante_emb).squeeze()
            # refine mention representations
            ante_weights = torch.softmax(torch.cat((dummy_scores, ante_scores), dim=1), dim=-1)
            ante_emb = torch.cat((ment_embs.unsqueeze(1), ante_ment_emb), dim=1)
            attended_emb = torch.sum(ante_weights.unsqueeze(-1) * ante_emb, dim=1)
            f = self.attended_gate(torch.cat((ment_embs, attended_emb), dim=1))
            ment_embs = f * attended_emb + (1 - f) * ment_embs

        # get final coreference score for antecedents and labels
        coref_score = torch.cat((dummy_scores, ante_scores), dim=1)
        labels = self.get_labels(ment_starts, ment_ends, gold_starts, gold_ends, cluster_ids, antes, ante_mask)

        return coref_score, labels, antes, ment_starts, ment_ends, cand_scores
