import pathlib

import pytest

from inlay.tokenizer import SlidingWindowTokenizer, SentencePieceTokenizer, HyphenTokenizer, PhoneticTokenizer
from inlay.tokenizer.common import flatten


all_tokenizers = flatten([
    [SlidingWindowTokenizer(ngram_range=(n, n)) for n in range(3)],
    [HyphenTokenizer(lang=l) for l in ['nl_NL', 'en_GB']],
    [PhoneticTokenizer(kind=k) for k in ["soundex", "metaphone", "nysiis"]],
    [PhoneticTokenizer(kind=k, hyphen="nl_NL") for k in ["soundex", "metaphone", "nysiis"]],
    [SentencePieceTokenizer(model_type=t, vocab_size=500, prefix="testfoo") for t in ["word", "bpe", "unigram", "char"]]
])


@pytest.fixture()
def simpsons_text():
    return [t for t in pathlib.Path("tests/textdata/simpsons.txt").read_text().split("\n") if len(t) > 2]


@pytest.mark.parametrize('tok', all_tokenizers)
def test_single_string_to_tokens(simpsons_text, tok):
    texts = ["inderdaad lettergrepen superduper cool"]
    result = tok.fit(simpsons_text).transform(texts)
    assert isinstance(result, list)
