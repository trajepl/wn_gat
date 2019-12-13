import os
import re

from nltk.corpus import wordnet as wn
from tqdm import tqdm

lemma_set = set()
synset_set = set()

# synsets edges
hypernyms = list()
hyponyms = list()
member_holonyms = list()

# lemmas edge
derivationally_related_forms = list()
pertainyms = list()
antonyms = list()


def to_str(synset_object):
    rls = str(synset_object).strip().split('(')[-1][1:-2]
    if not rls:
        print(synset_object)
    return rls


def record_edge(fn, obj):
    dir_fn = os.path.dirname(fn)
    if not os.path.exists(dir_fn):
        os.makedirs(dir_fn)
    with open(fn, 'w') as fout:
        for item in obj:
            fout.write(f'{item[0]} {item[1]}\n')


def record_node(fn, obj):
    dir_fn = os.path.dirname(fn)
    if not os.path.exists(dir_fn):
        os.makedirs(dir_fn)
    with open(fn, 'w') as fout:
        fout.write('0 _end\n')
        for idx, item in enumerate(obj):
            if item == 'flamboyant.s.01.flamboyant':
                print(1)
            fout.write(f'{idx+1} {item}\n')


if __name__ == "__main__":
    # synset edge
    for synset in tqdm(list(wn.all_synsets())):
        synset_set.add(synset)
        for item in synset.hypernyms():
            hypernyms.append((to_str(synset), to_str(item)))
        for item in synset.hyponyms():
            hyponyms.append((to_str(synset), to_str(item)))
        for item in synset.member_holonyms():
            member_holonyms.append((to_str(synset), to_str(item)))

        # lemma_set
        for item in synset.lemmas():
            lemma_set.add(to_str(item))

    # lemma edge
    for lemma in tqdm(lemma_set):
        lemma = wn.lemma(lemma)
        for item in lemma.derivationally_related_forms():
            derivationally_related_forms.append((to_str(lemma), to_str(item)))
        for item in lemma.pertainyms():
            pertainyms.append((to_str(lemma), to_str(item)))
        for item in lemma.antonyms():
            antonyms.append((to_str(lemma), to_str(item)))

    # node
    lemmas = []
    for item in lemma_set:
        lemmas.append(item)
    # lemmas = [to_str(it) for it in lemma_set]
    synsets = [to_str(it) for it in synset_set]
    words = list(wn.all_lemma_names())

    record_node('./data/wordnet/dictionary/words.txt', words)
    record_node('./data/wordnet/dictionary/lemmas.txt', lemmas)
    record_node('./data/wordnet/dictionary/synsets.txt', synsets)

    record_edge('./data/wordnet/raw_edge/synsets/hypernyms.edgelist', hypernyms)
    record_edge('./data/wordnet/raw_edge/synsets/hyponyms.edgelist', hyponyms)
    record_edge(
        './data/wordnet/raw_edge/synsets/member_holonyms.edgelist', member_holonyms)

    record_edge('./data/wordnet/raw_edge/lemmas/derivationally_related_forms.edgelist',
                derivationally_related_forms)
    record_edge('./data/wordnet/raw_edge/lemmas/pertainyms.edgelist', pertainyms)
    record_edge('./data/wordnet/raw_edge/lemmas/antonyms.edgelist', antonyms)
