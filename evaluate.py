import torch
import argparse
import numpy as np
from pathlib import Path
from pyhocon import ConfigFactory

from eval import evaluators, metrics, conll
from model.data import Dataset, DataLoader
from model.model import Model


class Evaluator:

    def __init__(self, conf, gpu):
        # load configuration
        self.config = ConfigFactory.parse_file('./coref.conf')[conf]

        # load dataset with test data
        path, elmo_path = self.config['eval_data_path'], self.config['eval_elmo_path']
        self.dataset = Dataset(self.config, path, elmo_path, training=False)
        self.dataloader = DataLoader(self.dataset, shuffle=False)

        # initialize model and move to gpu if available
        use_cuda = gpu and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        self.model = Model(self.config, device)
        self.model.to(device)
        self.model.eval()

    def eval(self, ckpt_path, amp):
        # load checkpoint
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)

        # initialize mention evaluators
        ment_evaluators = [
            evaluators.OracleEvaluator('oracle'),
            evaluators.ActualEvaluator('actual'),
            evaluators.ExactEvaluator('exact'),
            evaluators.ThresholdEvaluator('threshold')
        ]
        ment_evaluators.extend([evaluators.RelativeEvaluator(f'{i}%', i) for i in [10, 15, 20, 25, 30, 40, 50]])

        # initialize coref evaluator
        coref_preds = {}
        coref_evaluator = metrics.CorefEvaluator()

        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                # collect data for evaluating batch
                with torch.cuda.amp.autocast(enabled=amp):
                    _, _, _, _, sent_len, _, _, gold_starts, gold_ends, _, cand_starts, cand_ends = batch
                    scores, labels, antes, ment_starts, ment_ends, cand_scores = self.model(*batch)

                # update mention evaluators
                for ment_evaluator in ment_evaluators:
                    ment_evaluator.update(cand_starts, cand_ends, gold_starts, gold_ends, ment_starts, ment_ends,
                                          cand_scores, sent_len)

                # update coref evaluators
                raw_data = self.dataset.get_raw_data(i)
                pred_clusters = self.eval_antecedents(scores, antes, ment_starts, ment_ends, raw_data, coref_evaluator)
                coref_preds[raw_data['doc_key']] = pred_clusters

        # print F1, precision and recall of all mention evaluators
        for ment_evaluator in ment_evaluators:
            ment_evaluator.print_result()

        # print average F1 of CoNLL scorer
        conll_results = conll.evaluate_conll(self.config['eval_gold_path'], coref_preds, True)
        conll_f1 = np.mean([result['f'] for result in conll_results.values()])
        print(f'Average F1 (conll): {conll_f1:.2f}%')

        # print  F1, precision and recall of coreference evaluator
        p, r, f = coref_evaluator.get_prf()
        print(f'Average F1 (py): {100 * f:.2f}%')
        print(f'Average precision (py): {100 * p:.2f}%')
        print(f'Average recall (py): {100 * r:.2f}%\n')

    def eval_antecedents(self, scores, antes, ment_starts, ment_ends, raw_data, coref_evaluator):
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
        ment_to_pred_cluster = {m: pred_clusters[i] for m, i in ment_to_pred_cluster.items()}

        # get gold clusters and mapping of mentions to them
        raw_clusters = raw_data['clusters']
        gold_clusters = [tuple(tuple(ment) for ment in cluster) for cluster in raw_clusters]
        ment_to_gold_cluster = {ment: cluster for cluster in gold_clusters for ment in cluster}

        # update coref evaluator
        coref_evaluator.update(pred_clusters, gold_clusters, ment_to_pred_cluster, ment_to_gold_cluster)
        return pred_clusters


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate multiple model checkpoints.')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see coref.conf)')
    parser.add_argument('-p', metavar='PATTERN', default='*.pt', help='pattern for checkpoints (data/ckpt/<PATTERN>)')
    parser.add_argument('--cpu', action='store_true', help='train on CPU even when GPU is available')
    parser.add_argument('--amp', action='store_true', help='use amp optimization')
    args = parser.parse_args()

    # get path and search for ckpts
    path = Path('./data/ckpt')
    ckpts = sorted(path.glob(args.p))
    print('Checkpoints found: ')
    print('\n'.join([str(ckpt) for ckpt in ckpts]) + '\n')

    # run evaluation
    evaluator = Evaluator(args.c, not args.cpu)
    for ckpt in ckpts:
        print(f'\n########## {ckpt} ##########\n')
        evaluator.eval(ckpt, args.amp)
