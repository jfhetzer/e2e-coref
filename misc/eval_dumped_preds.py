import pickle
import numpy as np
from pathlib import Path
import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from eval import conll

###
#    Evaluate predictions dumped during training or inference.
#    Handy to reduce space needed for saved models or to evaluate the models output
#    without the hardware necessary for inference.
###

# TODO: Adjust paths to the structure of folders you dump the predictions into
DUMP_PATH = 'local/dump-cl/'
PRED_PATH = 'local/preds-cl/'

# TODO: Set search string for folders containing the predictions you want to evaluate
path = Path(PRED_PATH)
folders = path.glob('mbert_base*')

for folder in folders:
    print(f'Eval preds: {folder.name}')

    # TODO: Choose correct gold data based on the folder you are looking at
    gold_path = 'data/data/dev' if folder.name.endswith('dev') else 'data/data/test'
    if PRED_PATH == 'local/preds-en/' and folder.name.endswith('dev'):
        gold_path += '.english.v4_auto_conll'
    elif PRED_PATH == 'local/preds-en/':
        gold_path += '.english.v4_gold_conll'
    elif 'dirndl' in folder.name:
        gold_path += '.dirndl.v4_gold_conll'
    elif 'semeval' in folder.name:
        gold_path += '.semeval.v4_gold_conll_single'
    else:
        gold_path += '.tuebadz-v11.v4_gold_conll'

    dump_file = Path(DUMP_PATH) / f'{folder.name}.log'
    with open(dump_file, 'w') as out:
        out.write('Checkpoint  muc-p muc-r muc-f1  b3-p b3-r b3-f1  ceafe-p ceafe-r ceafe-f1  f1\n')

        preds = sorted(folder.glob('*'))
        for pred in preds:
            out.write(pred.name + '  ')
            dump = pickle.load(open(pred, 'rb'))
            for (doc, preds) in dump['pred'].items():
                for cluster in preds:
                    if len(cluster) == 1:
                        print('Alert: Singleton!!!')
            conll_results = conll.evaluate_conll(gold_path, dump['pred'], dump['subs'], False)
            for m in conll_results.values():
                out.write(f'{m["p"]} ')
                out.write(f'{m["r"]} ')
                out.write(f'{m["f"]}  ')

            conll_f1 = np.mean([result['f'] for result in list(conll_results.values())[:3]])
            out.write(f'{conll_f1}\n')
