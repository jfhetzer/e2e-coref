import sys

###
#    Count number of documents, sentences and tokens of CONLL-2012 file
###

file = sys.argv[1]
with open(file, 'r') as f:
    docs = 0
    sents = 0
    tokns = 0
    for line in f.readlines():
        parts = line.strip().split(None)
        if line.startswith('#begin document'):
            docs += 1
        if len(parts) > 2 and parts[2] == '1':
            sents += 1
        if line and line[0].isalpha():
            tokns += 1

print('### SIZE OF FILE ###')
print(f'File: {file}')
print(f'Documents: {docs}')
print(f'Sentences: {sents}')
print(f'Tokens: {tokns}')
