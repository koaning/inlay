import pathlib

import numpy as np
from gensim.models import KeyedVectors
from gensim.models import Word2Vec


class GensimEmbedder:
    def __init__(self, tokenizer, size=10, window=3, min_count=1, iter=5, workers=2, prefix=""):
        self.tokenizer = tokenizer
        self.size = size
        self.window = window
        self.min_count = min_count
        self.iter = iter
        self.workers = workers
        self.prefix = prefix

    def fit(self, X, y=None):
        tokens = self.tokenizer.transform(X)
        self.model_ = Word2Vec(sentences=tokens,
                               size=self.size,
                               window=self.window,
                               min_count=self.min_count,
                               iter=self.iter,
                               workers=self.workers)
        return self

    def encode(self, x):
        tokens = self.tokenizer.transform([x])
        vectors = np.zeros((len(tokens[0]), self.model_.wv.vector_size))
        for idx_t, tok in enumerate(tokens[0]):
            try:
                vectors[idx_t] = self.model_.wv[tok]
            except KeyError:
                pass
        return np.array(vectors).sum(axis=0)

    def transform(self, X, y=None):
        result = np.zeros((len(X), self.model_.wv.vector_size))
        for idx_x, x in enumerate(X):
            result[idx_x] = self.encode(x)
        return result

    @classmethod
    def train_file(cls, tokenizer, input_file, size=10, window=3, min_count=1, iter=5, workers=2, mod_name=None):
        text = pathlib.Path(input_file).read_text()
        tokens = tokenizer.transform(text.split("\n"))
        model = Word2Vec(sentences=tokens, size=size, window=window, min_count=min_count, iter=iter, workers=workers)
        if not mod_name:
            mod_name = f"{pathlib.Path(input_file).stem}-{size}-{window}-{iter}.kv"
        model.wv.save(mod_name)
        return mod_name

    @classmethod
    def from_file(cls, kv_file, tokenizer):
        cls.tokenizer = tokenizer
        cls.kv = KeyedVectors.load(kv_file)
        raise NotImplementedError()
