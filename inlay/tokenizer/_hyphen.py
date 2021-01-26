import pyphen
from sklearn.feature_extraction.text import CountVectorizer

from .common import flatten


class HyphenTokenizer:
    def __init__(self, lang="en_GB"):
        self.dic = pyphen.Pyphen(lang=lang)
        self.cv_ = CountVectorizer()
        self.tokenizer = self.cv_.build_tokenizer()

    def fit(self, X, y=None):
        return self

    def encode(self, x):
        return flatten([self.dic.inserted(t).split("-", -1) for t in self.tokenizer(x)])

    def transform(self, X, y=None):
        return [self.encode(x) for x in X]
