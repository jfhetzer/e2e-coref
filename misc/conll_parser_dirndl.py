###
#    Parse DINRDL datasets into CONLL-2012 format
###

DEV_IN = 'data/dirndl_anaphora_prosody.dev'
TEST_IN = 'data/dirndl_anaphora_prosody.test'
TRAIN_IN = 'data/dirndl_anaphora_prosody.train'

DEV_OUT = 'data/dev.dirndl.v4_gold_conll'
TEST_OUT = 'data/test.dirndl.v4_gold_conll'
TRAIN_OUT = 'data/train.dirndl.v4_gold_conll'


def parse(in_path, out_path):
    with open(in_path, 'r') as in_file:
        with open(out_path, 'w') as out_file:
            for line in in_file:
                line = line.strip()
                if line and not line.startswith('#'):
                    line = parse_line(line)
                out_file.write(line + '\n')


def parse_line(line):
    props = line.strip().split('\t')
    doc_key = props[0]
    part_no = props[1]
    word_no = props[2]
    word = props[3]
    coref = props[-1]
    coref = '-' if coref == '_' else coref
    return f'{doc_key} {part_no} {word_no} {word} - - - - - - - - {coref}'


parse(DEV_IN, DEV_OUT)
parse(TEST_IN, TEST_OUT)
parse(TRAIN_IN, TRAIN_OUT)
