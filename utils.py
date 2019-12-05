import os
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from mapping import Mapping
from vector_load import ModelLoad

ANTONYMS = 'antonyms.txt'
HYPERNYMS = 'hypernyms.txt'
HYPONYMS = 'hyponyms.txt'
SYNONYMS = 'synonyms.txt'
DICTIONARY = './data/dictionary_1w'
GLOVE_VEC = '/home/trajep/Workspace/pando/data/embedding/glove/glove.6B.300d.txt'


def emb_save(emb: torch.Tensor, path: str) -> None:
    torch.save(emb, path)


def emb_load(path: str) -> None:
    torch.load(path)


def model_save(state_dict: Dict, path: str) -> None:
    torch.save(state_dict, path)


def model_load(model: Any, path: str) -> None:
    model.load_state_dict(torch.load(path))


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


def load_data(base_folder: str) -> List[Any]:
    wn_dict = _load_dictionary(DICTIONARY)
    word_vec = [None] * wn_dict.size()

    vec_model = ModelLoad('glove', GLOVE_VEC)
    vec_model.load()
    n_features = len(vec_model.model['a'])

    print('Extract glove vector')
    for token, idx in tqdm(wn_dict.token2id.items()):
        if token in vec_model.model.keys():
            word_vec[idx] = vec_model.model[token]
        else:
            word_vec[idx] = [0.] * n_features

    print('Extract word rels')
    row, col = [], []
    for fn in os.listdir(base_folder):
        fn_full = os.path.join(base_folder, fn)
        with open(fn_full, 'r', encoding='utf-8') as fin:
            row_t, col_t = row_col(fin, fn)
            row += row_t
            col += col_t
    return word_vec, row, col


if __name__ == "__main__":
    word_vec, row, col = load_data('./data/wn_graph')
