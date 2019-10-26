from typing import List, Optional

from overrides import overrides

import sentencepiece as spm
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer


@Tokenizer.register("sentencepiece")
class SentencepieceTokenizer(Tokenizer):
    def __init__(
        self,
        model_path: str,
        start_tokens: Optional[List[str]] = None,
        end_tokens: Optional[List[str]] = None,
        subword_regularization_sample_size: int = 0,
        subword_regularization_alpha: float = 0.2,
    ) -> None:
        self._start_tokens = start_tokens or []
        # We reverse the tokens here because we're going to insert them with `insert(0)` later;
        # this makes sure they show up in the right order.
        self._start_tokens.reverse()
        self._end_tokens = end_tokens or []
        self._subword_regularization_sample_size = subword_regularization_sample_size
        self._subword_regularization_alpha = subword_regularization_alpha
        self._processor = spm.SentencePieceProcessor()
        self._processor.load(model_path)

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
        for start_token in self._start_tokens:
            tokens.insert(0, Token(start_token, 0))
        for end_token in self._end_tokens:
            tokens.append(Token(end_token, -1))
        return tokens
