import json

###
#    Checks the segments size of segmentized data
###

FILE = 'data/data/spanbert_large/train.english.jsonlines'

with open(FILE, 'r') as f:
    lines = f.readlines()
    max_length = 0
    for line in lines:
        data = json.loads(line)
        length = len(data['segments'][0])
        max_length = max(max_length, length)

print(f'segm_size: {max_length}')
