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


def lemmas_name(synset: str) -> List[str]:
    rls = []
    raw_rls = [item.name() for item in wn.synset(synset).lemmas()]
    for item in raw_rls:
        if '_' in item:
            rls += [word.lower() for word in item.split('_')]
        else:
            rls.append(item.lower())
    return rls


def synsets(w1: str) -> torch.Tensor:
    rls = []
    synsets_list = wn.synsets(w1)
    for synset in synsets_list:
        synset_key = to_str(synset)
        rls.append(synset_map.token2id[synset_key])
    return torch.tensor(rls, dtype=torch.long)
