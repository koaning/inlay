import jellyfish
import pyphen
from sklearn.feature_extraction.text import CountVectorizer

from .common import flatten


class PhoneticTokenizer:
    def __init__(self, kind="soundex", hyphen=None):
        """
        kind=soundex, metaphone, nysiis
        """
        methods = {"soundex": jellyfish.soundex,
                   "metaphone": jellyfish.metaphone,
                   "nysiis": jellyfish.nysiis}
        self.method = methods[kind]
        self.hyphen = hyphen
        if hyphen:
            self.dic = pyphen.Pyphen(lang=hyphen)
            self.method = lambda d: self.dic.inserted(methods[kind](d)).split("-", -1)

    def fit(self, X, y=None):
        self.cv_ = CountVectorizer()
        self.tokenizer = self.cv_.build_tokenizer()
        return self

    def encode(self, x):
        if self.hyphen:
            return flatten([self.method(t) for t in self.tokenizer(x)])
        return [self.method(t) for t in self.tokenizer(x)]

    def transform(self, X, y=None):
        return [self.encode(x) for x in X]