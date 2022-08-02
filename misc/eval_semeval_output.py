import re
import argparse
import numpy as np

from eval import conll

###
#    Evaluate system outputs of SemEval-2010 shared task.
#    System outputs can be found under:
###

class Evaluator:

    def __init__(self):
        self.preds = {}
        self.subs = {}
        self.init_doc()

    def init_doc(self):
        self.pred = {}
        self.token = 0
        self.ment_start = {}

    def parse_line(self, line):
        boundaries = line.split('|')
        for b in boundaries:
            cid = int(re.sub('\D', '', b))
            if b.startswith('('):
                start = self.ment_start.get(cid, [])
                start.append(self.token)
                self.ment_start[cid] = start
            if b.endswith(')'):
                cluster = self.pred.get(cid, [])
                start = self.ment_start[cid].pop()
                cluster.append([start, self.token])
                self.pred[cid] = cluster

    def read_system_output(self, out_path):
        with open(out_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#begin document'):
                    continue

                if line.startswith('#end document'):
                    name = line.split(' ')[-1] + '_0'
                    self.preds[name] = self.pred.values()
                    self.subs[name] = list(range(self.token + 1))
                    self.init_doc()
                else:
                    line = line.split('\t')[-1]
                    line = line.replace('_', '')
                    if line:
                        self.parse_line(line)
                    self.token += 1

    def evaluate(self, gold_path, singletons):
        if not singletons:
            for doc, clusters in self.preds.items():
                self.preds[doc] = list(filter(lambda x: len(x) > 1, clusters))

        conll_results = conll.evaluate_conll(gold_path, self.preds, self.subs, True)
        conll_f1 = np.mean([result['f'] for result in conll_results.values()])
        print(f'Average F1 (conll): {conll_f1:.2f}%')


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate multiple model checkpoints.')
    parser.add_argument('-g', metavar='GOLD_PATH', default='data/data/test.semeval.v4_gold_conll', help='path to gold file')
    parser.add_argument('-o', metavar='OUT_PATH', default='preprocess/sucre_closed_regular.txt', help='path to system output')
    parser.add_argument('-s', default=False, action='store_true', help='include singletons')
    args = parser.parse_args()

    evaluator = Evaluator()
    evaluator.read_system_output(args.o)
    evaluator.evaluate(args.g, args.s)
