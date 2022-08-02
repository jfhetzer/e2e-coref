import torch
import h5py
import json
from allennlp.modules.elmo import batch_to_ids, remove_sentence_boundaries, _ElmoBiLm


OPTIONS = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
WEIGHTS = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


class ElmoCacheBuilder:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.elmo = _ElmoBiLm(OPTIONS, WEIGHTS).to(self.device)
        print(f'running on: {self.device}')

    def elmo_cache(self, data_file, cache_file):
        with open(data_file, 'r') as input_file, h5py.File(cache_file, 'w') as output_file:
            for i, line in enumerate(input_file):
                # use batch_to_ids to convert sentences to character ids
                doc = json.loads(line)
                sentences = doc['sentences']
                sent_len = [len(sent) for sent in sentences]
                character_ids = batch_to_ids(sentences)
                output = self.elmo(character_ids.to(self.device))

                # write to cache file
                doc_key = doc['doc_key'].replace('/', ':')
                cache = output_file.create_group(doc_key)

                # get representations and remove boundaries
                elmo_mask = output['mask']
                elmo_repr = output['activations']
                batches = []
                for batch in elmo_repr:
                    rep, _ = remove_sentence_boundaries(batch, elmo_mask)
                    batches.append(rep)
                output = torch.stack(batches, dim=-1)
                for i, s_len in enumerate(sent_len):
                    cache[str(i)] = output[i].cpu()[:s_len, ...]


if __name__ == '__main__':
    with torch.no_grad():
        builder = ElmoCacheBuilder()
        builder.elmo_cache('data/data/train.english.jsonlines', 'data/embs/train.elmo_cache.hdf5')
        builder.elmo_cache('data/data/dev.english.jsonlines', 'data/embs/dev.elmo_cache.hdf5')
        builder.elmo_cache('data/data/test.english.jsonlines', 'data/embs/test.elmo_cache.hdf5')
