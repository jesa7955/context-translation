import os
from typing import List, Optional

from overrides import overrides

import sentencepiece as spm
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer


@Tokenizer.register("sentencepiece")
class SentencepieceTokenizer(Tokenizer):
    def __init__(
        self,
        model_path: str,
        subword_regularization_sample_size: int = 0,
        subword_regularization_alpha: float = 0.2,
    ) -> None:
        self._subword_regularization_sample_size = subword_regularization_sample_size
        self._subword_regularization_alpha = subword_regularization_alpha
        self._processor = spm.SentencePieceProcessor()
        self.model_path = model_path
        self.load()

    def load(self):
        if os.path.exists(self.model_path):
            self._processor.load(self.model_path)

    def model_trained(self):
        return os.path.exists(self.model_path)

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        if self._subword_regularization_sample_size == 0:
            tokens = self._processor.EncodeAsPieces(text)
        else:
            tokens = self._processor.SampleEncodeAsPieces(
                text,
                self._subword_regularization_sample_size,
                self._subword_regularization_alpha,
            )
        tokens = [Token(token) for token in tokens]
        return tokens
