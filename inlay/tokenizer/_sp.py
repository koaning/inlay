import pathlib

import sentencepiece as spm


class SentencePieceTokenizer:
    def __init__(self, model_file, output_type=str):
        self.mod = spm.SentencePieceProcessor(model_file=model_file)
        self.output_type = output_type

    def transform(self, X, y=None):
        return [self.mod.encode(x, out_type=self.output_type) for x in X]

    @classmethod
    def train_file(cls, input_file, vocab_size=10_000, model_type="bpe", mod_name=None):
        """
        model_type= "word", "bpe", "unigram", "char"
        """
        if not mod_name:
            mod_name = f"{pathlib.Path(input_file).stem}-{model_type}-{vocab_size}"
        spm.SentencePieceTrainer.train(input=input_file, model_prefix=mod_name, vocab_size=vocab_size,
                                       model_type=model_type)
