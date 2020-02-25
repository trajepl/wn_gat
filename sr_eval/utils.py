import os
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from mapping import Mapping
from sr_eval import metric
from word_mapping import synsets
from vector_load import ModelLoad

DICT_PATH = './data/dictionary'
SR_GOLDEN_DATA = './sr_eval/golden/'
GLOVE_VEC = '/home/jpli/data/glove.6B.100d.txt'
w2v = ModelLoad('glove', GLOVE_VEC)
w2v.load()

def sr_ouput(fn: str, A: List, B: List) -> float:
    rls_spearman = metric.spearman(A, B)
    rls_pearson = metric.pearson(A, B)
    print(fn + ':', end='')
    print('\tspearman:{:.3f}'.format(rls_spearman), end='')
    print('\tpearson:{:.3f}'.format(rls_pearson))
    return rls_spearman, rls_pearson


def read_vec(fn: str, csep: str = ';') -> List:
    rst_word, rst_score = list(), list()
    with open(fn, 'r') as fin:
        for line in fin.readlines():
            line = line.strip().split(sep=csep)
            words = list(map(str.lower, line[:-1]))
            rst_word.append(words)
            rst_score.append(float(line[-1]))
    return rst_word, rst_score


def sr_score(w1: str,
             w2: str,
             device: torch.device,
             func: object,
             strategy: str = 'max',
             **params) -> torch.Tensor:

    w1_synsets, w2_synsets = synsets(w1).to(device), synsets(w2).to(device)
    w1_synsets_vec = func(w1_synsets, **params).to(device)
    w2_synsets_vec = func(w2_synsets, **params).to(device)

    if device.type.startswith('cuda'):
        w1_synsets_vec, w2_synsets_vec = w1_synsets_vec.cpu(), w2_synsets_vec.cpu()
    w1_synsets_vec, w2_synsets_vec = w1_synsets_vec.detach(
    ).numpy(), w2_synsets_vec.detach().numpy()

    rls = [0, 0]
    for i in w1_synsets_vec:
        for j in w2_synsets_vec:
            ij_score = metric.cosine(i, j)
            # if strategy == 'max':
            rls[0] = max(rls[0], ij_score)
            # elif strategy == 'mean':
            rls[1] += ij_score
    if len(w1_synsets_vec) == 0 or len(w2_synsets_vec) == 0:
        rls[1] = 0.862
    else:
        rls[1] = rls[1] / (len(w1_synsets_vec) * len(w2_synsets_vec))
    return rls

def word_sr_score(w1: str, w2: str) -> float:
    if w1 not in w2v.model.keys() or w2 not in w2v.model.keys():
        return 0
    return metric.cosine(w2v.model[w1], w2v.model[w2])


def sr_eval(device: torch.device,
      func: object,
      csep: str = ';',
      **params) -> List[float]:
    
    sr = []
    for fnt in os.listdir(SR_GOLDEN_DATA):
        rls = [[], []]
        if not fnt.endswith('.csv'):
            continue
        fn = os.path.join(SR_GOLDEN_DATA, fnt)
        golden_words, golden_score = read_vec(fn)
        for _lambda in range(0, 11):
            scores = [[], []]
            for word_pair in golden_words:
                w1, w2 = word_pair[0], word_pair[1]
                ent_scores = sr_score(w1, w2, device, func, 'max', **params)
                word_scores = word_sr_score(w1, w2)
                scores[0].append((_lambda * word_scores + (10 - _lambda) * ent_scores[0]) / 10)
                scores[1].append((_lambda * word_scores + (10 - _lambda) * ent_scores[1]) / 10)
            rls[0].append(sr_ouput(fnt, golden_score, scores[0]))
            rls[1].append(sr_ouput(fnt, golden_score, scores[1]))
        sr.append(rls)
    return sr


def sr_test(device: torch.device,
            func: object,
            strategy: str = 'max',
            csep: str = ';',
            **params) -> List[float]:

    rls = [[], []]
    for fnt in os.listdir(SR_GOLDEN_DATA):
        if not fnt.endswith('.csv'):
            continue
        fn = os.path.join(SR_GOLDEN_DATA, fnt)
        golden_words, golden_score = read_vec(fn)
        test_score = [[], []]
        for word_pair in golden_words:
            w1, w2 = word_pair[0], word_pair[1]
            scores = sr_score(w1, w2, device, func, strategy, **params)
            test_score[0].append(scores[0])
            test_score[1].append(scores[1])
        rls[0].append(sr_ouput(fnt, golden_score, test_score[0]))
        rls[1].append(sr_ouput(fnt, golden_score, test_score[1]))
    return rls
