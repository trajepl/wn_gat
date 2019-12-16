from typing import List
import numpy as np
from scipy.stats import pearsonr, spearmanr


def cosine(A, B) -> float:
    '''
    A: 1d-vector, object of np.array
    B: 1d-vector, object of np.array
    '''
    num = np.dot(A, B)
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    cos = num / denom
    # 归一化
    return 0.5 + 0.5 * cos


def pearson(A: List, B: List) -> float:
    '''
    A: 2d-list, list objects of np.array
    B: 2d-list, list objects of np.array
    '''
    return pearsonr(A, B)[0]


def spearman(A: List, B: List) -> float:
    '''
    A: 2d-list, list objects of np.array
    B: 2d-list, list objects of np.array
    '''
    return spearmanr(A, B)[0]
