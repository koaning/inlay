# inlay

a bit of fun embedding in non-standard ways

```
from inlay.tokenizers import WhitespaceTokenizer, HyphenTokenizer, JellyfishTokenizer, BytePairTokenizer, WordpieceTokenizer
from inlay.embedders import Gensim
from inlay.targeted import DeepAverage, Attention
```

## Observation 1

```
pipe = Pipeline([
  ('tok1', WhitespaceTokenizer()),
  ('tok2', JellyfishTokenizer()), 
  ('mod', Gensim())
])

pipe.fit(texts)
```

It'd be grand if tokenizers could accept both text *or* a list of tokens. A lost of tokens could be turned into a list of subtokens. 

## Observation 2 

Maybe we don't need a pipeline. Maybe this is enough;

```
InlayTrainer(tokenizer=HyphenTokenizer.from_file()), iterations=100).fit(texts)
```

## Observation 3 

This can get ... complex. 

```
(DeepAverage(tokenizers=[TextCleaner(), WhitespaceTokenzier()])
  .train(sentiment=(X_s, y_s), 
         autoencode=(X_e, X_e), 
         classify=(X_news, y_news)))
```

Then again ... maybe Gensim is fine for now ... no need to complicate things. 
