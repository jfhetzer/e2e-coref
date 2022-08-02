import json

###
#    Checks the maximum number of segments for each datum
###

FILE = 'data/data/spanbert_large/train.english.jsonlines'

stats = [0] * 20
with open(FILE, 'r') as f:
    lines = f.readlines()
    max_length = 0
    for line in lines:
        data = json.loads(line)
        length = len(data['segments'])
        stats[length] += 1
        max_length = max(max_length, length)

print(f'segm_size: {max_length}')
print(f'stats: {stats}')
