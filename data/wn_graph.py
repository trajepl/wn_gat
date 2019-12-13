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
        else:
            print(item)
    return rls


def compress(wn_dict: Mapping, src_folder_name: str = WN_RAW, tgt_folder_name: str = WN_GRAPH) -> None:
    for fn_raw in os.listdir(src_folder_name):
        fn = os.path.join(src_folder_name, fn_raw)
        fn_rls = os.path.join(tgt_folder_name, fn_raw)
        with open(fn, 'r', encoding='utf-8') as fin:
            with open(fn_rls, 'w', encoding='utf-8') as fout:
                print(f'compress {fn_raw}...')
                for line in tqdm(fin.readlines()):
                    line = line.strip().split()
                    line_rls = filter(line, wn_dict)
                    if len(line_rls) < 2:
                        continue
                    fout.write(' '.join(list(line_rls)) + '\n')


if __name__ == "__main__":
    # wn_dict = Mapping()
    # wn_dict.load('./data/dictionary_1w')
    # compress(wn_dict)
    lemmas_raw = './data/wordnet/raw_edge/lemmas'
    sysets_raw = './data/wordnet/raw_edge/synsets'
    lemmas = './data/wordnet/edge/lemmas'
    synsets = './data/wordnet/edge/synsets'

    synsets_dict = Mapping()
    synsets_dict.load('./data/wordnet/dictionary/synsets.txt')
    compress(synsets_dict, sysets_raw, synsets)

    lemmas_dict = Mapping()
    lemmas_dict.load('./data/wordnet/dictionary/lemmas.txt')
    compress(lemmas_dict, lemmas_raw, lemmas)
