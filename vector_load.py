from typing import Dict

import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm


class ModelLoad():
    def __init__(self, model_name: str, model_path: str):
        self.model = {}
        self.model_name = model_name
        self.model_path = model_path

    def load(self) -> Dict:
        print(f'Loading {self.model_name} Model')
        if self.model_name == 'word2vec':
            self.model = Word2Vec.load(self.model_path).wv
        else:
            with open(self.model_path, 'r') as fin:
                for line in tqdm(fin.readlines()):
                    line = line.split()
                    word = line[0]
                    embedding = np.array([float(val) for val in line[1:]])
                    self.model[word] = embedding
            print('Done.', len(self.model), 'words loaded!')


if __name__ == '__main__':
    model = ModelLoad('glove', 'glove_model_path')
    model.load()
