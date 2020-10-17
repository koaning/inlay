# inlay

a bit of fun embedding in non-standard ways

```
from inlay.tokenizers import WhitespaceTokenizer, HyphenTokenizer, JellyfishTokenizer, BytePairTokenizer, WordpieceTokenizer
from inlay.embedders import Gensim
from inlay.targeted import DeepAverage, Attention
```

## Observation 1

Maybe we don't need a pipeline. Maybe this is enough;

```
InlayTrainer(tokenizer=HyphenTokenizer.from_file()), iterations=100).fit(texts)
```

## Observation 2

This can get ... complex. 

```
(DeepAverage(tokenizers=[TextCleaner(), WhitespaceTokenzier()])
  .train(sentiment=classification(X_s, y_s), 
         autoencode=autoencode(X_e, X_e), 
         classify=similarity(question, answer)))
```

Then again ... maybe Gensim is fine for now ... no need to complicate things. 
