import json
import torch
import argparse
from pathlib import Path
from pyhocon import ConfigFactory

from model.data import Dataset, DataLoader
from model.model import Model


class Inference:

    def __init__(self, conf):
        # load configuration
        self.config = ConfigFactory.parse_file('./coref.conf')[conf]
        # load dataset with test data
        self.dataset = Dataset(self.config, training=False)
        self.dataloader = DataLoader(self.dataset, shuffle=False)

    def infer(self, ckpt_path, out_path, amp, gpu, split):
        # configure devices (cpu or up to two sequential gpus)
        use_cuda = gpu and torch.cuda.is_available()
        device1 = torch.device('cuda:0' if use_cuda else 'cpu')
        device2 = torch.device('cuda:1' if use_cuda else 'cpu')
        device2 = device2 if split else device1
        print(f'Running on: {device1} {device2}')

        # initialize model and move to gpu if available
        model = Model(self.config, device1, device2)
        model.bert_model.to(device1)
        model.task_model.to(device2)
        model.eval()

        # load checkpoint
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)

        # infer model
        with open(out_path, 'a') as out:
            self.infer_model(model, out, amp)

    def infer_model(self, model, out, amp):
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                # collect data for evaluating batch
                with torch.cuda.amp.autocast(enabled=amp):
                    _, segm_len, _, _, gold_starts, gold_ends, _, cand_starts, cand_ends = batch
                    scores, labels, antes, ment_starts, ment_ends, cand_scores = model(*batch)

                # get predicted clusters on subtoken level
                pred_clusters = self.create_cluster(scores, antes, ment_starts, ment_ends)

                # flatten clusters into coref-data on token level
                raw_data = self.dataset.get_raw_data(i)
                coref_data = self.create_coref(pred_clusters, raw_data['token_map'])

                # write coref_data to out-file
                json_line = json.JSONEncoder().encode(coref_data)
                out.write(json_line + '\n')

    def create_cluster(self, scores, antes, ment_starts, ment_ends):
        # tensor to numpy array
        ment_starts = ment_starts.numpy()
        ment_ends = ment_ends.numpy()

        # get best antecedent per mention (as mention index)
        pred_ante_idx = torch.argmax(scores, dim=1) - 1
        pred_antes = [-1 if ante_idx < 0 else antes[ment_idx, ante_idx] for ment_idx, ante_idx in
                      enumerate(pred_ante_idx)]

        # get predicted clusters and mapping of mentions to them
        # antecedents have to be sorted by mention start
        ment_to_pred_cluster = {}
        pred_clusters = []
        for ment_idx, pred_idx in enumerate(pred_antes):
            # ignore dummy antecedent
            if pred_idx < 0:
                continue

            # search for corresponding cluster or create new one
            pred_ante = (ment_starts[pred_idx], ment_ends[pred_idx])
            if pred_ante in ment_to_pred_cluster:
                cluster_idx = ment_to_pred_cluster[pred_ante]
            else:
                cluster_idx = len(pred_clusters)
                pred_clusters.append([pred_ante])
                ment_to_pred_cluster[pred_ante] = cluster_idx

            # add mention to cluster
            ment = (ment_starts[ment_idx], ment_ends[ment_idx])
            pred_clusters[cluster_idx].append(ment)
            ment_to_pred_cluster[ment] = cluster_idx

        # replace mention indices with mention boundaries
        pred_clusters = [tuple(cluster) for cluster in pred_clusters]
        return pred_clusters

    def create_coref(self, clusters, token_map):
        token_num = token_map[-2] + 1
        coref = [[] for _ in range(token_num)]
        for c_idx, cluster in enumerate(clusters):
            for mention in cluster:
                subtoken_s = mention[0]
                subtoken_e = mention[1]
                token_s = token_e = -1
                while token_s < 0:
                    token_s = token_map[subtoken_s]
                while token_e < 0:
                    token_e = token_map[subtoken_e]
                for t_idx in range(token_s, token_e + 1):
                    coref[t_idx].append(c_idx)
        return coref


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate multiple model checkpoints.')
    parser.add_argument('-c', metavar='CONF', default='bert-base', help='configuration (see coref.conf)')
    parser.add_argument('-p', metavar='PATH', default='ckpt.pt.tar', help='path to snapshot')
    parser.add_argument('-o', metavar='OUT_FILE', default='out.jsonlines', help='jsonlines output file')
    parser.add_argument('--cpu', action='store_true', help='train on CPU even when GPU is available')
    parser.add_argument('--amp', action='store_true', help='use amp optimization')
    parser.add_argument('--split', action='store_true', help='split the model across two GPUs')
    args = parser.parse_args()

    # get path and search for ckpts
    path = Path('./data/ckpt')
    ckpts = sorted(path.glob(args.p))
    print('Checkpoints found: ')
    print('\n'.join([str(ckpt) for ckpt in ckpts]))

    # run evaluation
    inference = Inference(args.c)
    for ckpt in ckpts:
        print(f'\n### {ckpt} ###')
        inference.infer(ckpt, args.o, args.amp, not args.cpu, args.split)
