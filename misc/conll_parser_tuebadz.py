import re
from string import Template

###
#    Parse Tüba-D/Z v10/11 datasets into CONLL-2012 format
###

SRC_PATH = 'data/tuebadz-11.0-conll2011.txt'
OUT_V10_PATH = Template('data/${mode}.tuebadz-v10.v4_gold_conll')
OUT_V11_PATH = Template('data/${mode}.tuebadz-v11.v4_gold_conll')

BEGIN_PATTER = re.compile(r'#begin document (T.*)\.(.*)')


def parse(file, out, split):
    modes = ['test', 'dev', 'train']
    for mode, length in zip(modes, split):
        out_path = out.substitute(mode=mode)
        with open(out_path, 'w') as f_out:
            doc_count = 0
            while doc_count < length:
                line = file.readline().strip()
                doc_count += add_line(line, f_out)


def add_line(line, out):
    if line.startswith('#end'):
        out.write(line + '\n')
        return 1

    if line.startswith('#begin'):
        result = BEGIN_PATTER.match(line)
        doc, part = result.group(1), int(result.group(2))
        out.write(f'#begin document ({doc}); part {part:02d}\n')

    elif line.startswith('T'):
        props = line.split('\t')
        doc = props[0].split('.')[0]
        part_no = props[0].split('.')[1]
        word_no = props[2]
        word = props[3]
        coref = props[-1]
        out.write(f'{doc} {part_no} {word_no} {word} - - - - - - - - {coref}\n')

    else:
        out.write(line + '\n')

    return 0


with open(SRC_PATH, 'r') as f:
    # parse TüBa-D/Z v10
    parse(f, OUT_V10_PATH, (727, 727, 2190))
with open(SRC_PATH, 'r') as f:
    # parse TüBa-D/Z v11
    parse(f, OUT_V11_PATH, (727, 727, 2362))
