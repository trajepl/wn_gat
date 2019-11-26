from typing import Union

from tqdm import tqdm


class Token(object):
    def __init__(self, id: Union[int, str], token: str) -> None:
        self._id = id
        self._token = token

    @property
    def id(self) -> Union[int, str]:
        return self._id

    @property
    def token(self) -> Union[int, str]:
        return self._token


class Mapping(object):

    def __init__(self) -> None:
        self.token2id = dict()
        self.id2token = dict()
    

    def size(self) -> int:
        return len(self.token2id)

    def load(self, fn) -> None:
        with open(fn, 'r', encoding='utf-8') as fin:
            for line in tqdm(fin.readlines()):
                line = line.strip().split(' ')
                self.add_token(Token(int(line[0]), line[1]))

    def add_token(self, token: Token) -> None:
        if token.id in self.id2token.keys():
            return None
        self.token2id[token.token] = token.id
        self.id2token[token.id] = token.token
