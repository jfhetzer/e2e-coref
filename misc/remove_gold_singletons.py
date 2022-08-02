import re

###
#    Remove singletons from CONLL-2012 file.
#    Used for SemEval since this is the only corpus with singletons.
###

GOLD_PATH = 'data/data/test.semeval.v4_gold_conll'
SINGLETONS_PATH = 'data/data/test.semeval.v4_gold_conll_single'

ids = {}
text = ''

with open(GOLD_PATH, 'r') as f_in:
    with open(SINGLETONS_PATH, 'w') as f_out:
        for line in f_in:
            text += line
            line = line.strip()
            if line.startswith('#end'):
                singletons = [k for k, v in ids.items() if v < 3]
                for s in singletons:
                    text = text.replace(f'({s}|', '')
                    text = text.replace(f'|({s}\n', '\n')
                    text = text.replace(f'({s}\n', '-\n')
                    text = text.replace(f'|{s})', '')
                    text = text.replace(f' {s})|', ' ')
                    text = text.replace(f' {s})', ' -')
                    text = text.replace(f'|({s})', '')
                    text = text.replace(f'({s})|', '')
                    text = text.replace(f' ({s})\n', ' -\n')
                f_out.write(text)
                text = ''
                ids = {}

            if not line.startswith('export_'):
                continue

            boundaries = line.split(' ')[-1]
            boundaries = boundaries.replace('-', '')
            if not boundaries:
                continue

            for b in boundaries.split('|'):
                id = int(re.sub('\D', '', b))
                count = ids.get(id, 0)
                if b.startswith('('):
                    count += 1
                if b.endswith(')'):
                    count += 1
                ids[id] = count
