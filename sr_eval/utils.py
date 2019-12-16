import os
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from mapping import Mapping
from sr_eval import metric

DICT_PATH = './data/dictionary'
EMB_KEY = 'oup'


def sr_ouput(fn: str, A: List, B: List) -> float:
    rls_spearman = metric.spearman(A, B)
    rls_pearson = metric.pearson(A, B)
    print(fn + ':')
    print('\tspearman:{:.3f}'.format(rls_spearman))
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
