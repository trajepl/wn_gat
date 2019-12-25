import os
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm


from mapping import Mapping
from vector_load import ModelLoad
from word_mapping import lemmas_name

ANTONYMS = 'antonyms.txt'
HYPERNYMS = 'hypernyms.txt'
HYPONYMS = 'hyponyms.txt'
SYNONYMS = 'synonyms.txt'
DICTIONARY = './data/dictionary'
# GLOVE_VEC = '/home/trajep/Workspace/pando/data/embedding/glove/glove.6B.300d.txt'
GLOVE_VEC = '/home/jpli/data/glove.6B.100d.txt'
SYNSETS_VEC = './data/wordnet/node/synsets.txt'


def synset_emb(word_emb: str = GLOVE_VEC, oup_path: str = SYNSETS_VEC) -> None:
    from word_mapping import synset_map
    w2v = ModelLoad('glove', word_emb)
    w2v.load()

    with open(oup_path, 'w', encoding='utf-8') as fout:
        for synset in tqdm(synset_map.token2id.keys()):
            try:
                syn_words = lemmas_name(synset)
            except Exception as _:
                continue
            syn_w2v = []
            for word in syn_words:
                if word in w2v.model.keys():
                    syn_w2v.append(w2v.model[word])
            if len(syn_w2v) == 0:
                continue
            syn_w2v = np.array(syn_w2v).transpose(1, 0).mean(1).tolist()
            emb_str = ' '.join([str(it) for it in syn_w2v])
            fout.write(f'{synset} {emb_str}\n')


def row_col(fin, fn: str) -> List[List]:
    row, col = [], []
    for line in tqdm(fin.readlines()):
        line = list(map(int, line.strip().split()))

        if fn is ANTONYMS or SYNONYMS:
            for item_i in line:
                for item_j in line:
                    if item_i == item_j:
                        continue
                    row.append(item_i)
                    col.append(item_j)
        else:
            row.append(line[0])
            col.append(line[1])

    return row, col


def _load_dictionary(dictionary: str):
    wn_dict = Mapping()
    wn_dict.load(dictionary)
    return wn_dict


def load_data(base_folder: str, dictionary: str = DICTIONARY,
              node_emb_fn: str = SYNSETS_VEC, model_name: str = 'gat') -> List[Any]:

    wn_dict = _load_dictionary(dictionary)
    word_vec = [None] * wn_dict.size()
    n_features = 0

    print('Extract node feature vectors')
    with open(node_emb_fn, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin.readlines()):
            line = line.strip().split(' ')
            node = wn_dict.token2id[line[0]]
            emb = [float(val) for val in line[1:]]
            if n_features == 0:
                n_features = len(emb)
            word_vec[node] = emb
    for idx, item in enumerate(word_vec):
        if item is None:
            word_vec[idx] = [0.] * n_features

    print('Extract node relationships')
    row, col = [], []
    for fn in os.listdir(base_folder):
        fn_full = os.path.join(base_folder, fn)
        with open(fn_full, 'r', encoding='utf-8') as fin:
            row_t, col_t = row_col(fin, fn)
            row += row_t
            col += col_t
    return word_vec, row, col


if __name__ == "__main__":
    # word_vec, row, col = load_data('./data/wn_graph')
    synset_emb()
