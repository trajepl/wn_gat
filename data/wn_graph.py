import os
from pathlib import Path
from typing import List

from tqdm import tqdm

from mapping import Mapping

WN_GRAPH = './data/wn_graph/'
WN_RAW = './data/wordnet/'


def filter(line: List, dictionary: Mapping) -> List:
    rls = []
    for item in line:
        if item in dictionary.token2id.keys():
            rls.append(str(dictionary.token2id[item]))
    return rls


def compress(wn_dict: Mapping) -> None:
    for fn_raw in os.listdir(WN_RAW):
        fn = os.path.join(WN_RAW, fn_raw)
        fn_rls = os.path.join(WN_GRAPH, fn_raw)
        with open(fn, 'r', encoding='utf-8') as fin:
            with open(fn_rls, 'w', encoding='utf-8') as fout:
                print(F'compress {fn_raw}...')
                for line in tqdm(fin.readlines()):
                    line = line.strip().split()
                    line_rls = filter(line, wn_dict)
                    if len(line_rls) < 2:
                        continue
                    fout.write(' '.join(list(line_rls)) + '\n')


if __name__ == "__main__":
    wn_dict = Mapping()
    wn_dict.load('./data/dictionary_1w')
    compress(wn_dict)
