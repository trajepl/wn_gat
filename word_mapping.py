import torch
from typing import List

from nltk.corpus import wordnet as wn

from mapping import Mapping

SYNSETS_PATH = './data/wordnet/dictionary/synsets.txt'
synset_map = Mapping()
synset_map.load(SYNSETS_PATH)


def to_str(synset_object):
    rls = str(synset_object).strip().split('(')[-1][1:-2]
    if not rls:
        print(synset_object)
    return rls


def synsets(w1: str, synsets_path: str = SYNSETS_PATH) -> torch.Tensor:
    rls = []
    synsets_list = wn.synsets(w1)
    for synset in synsets_list:
        synset_key = to_str(synset)
        rls.append(synset_map.token2id[synset_key])
    return torch.tensor(rls, dtype=torch.long)
