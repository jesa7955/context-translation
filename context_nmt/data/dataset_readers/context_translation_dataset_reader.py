import logging
import itertools
import tempfile
import os
from typing import List, Dict
from overrides import overrides

import numpy as np
import sentencepiece as spm
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ArrayField
from allennlp.data import Token, Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN

from context_nmt.data.tokenizers.sentencepiece_tokenizer import SentencepieceTokenizer

CONCAT_SYMBOL = "@concat@"
SEP_SYMBOL = "[SEP]"
CLS_SYMBOL = "[CLS]"

SPECIAL_CHARACTER_COVERAGES_LANG = set(["ja", "zh", "kr"])

logger = logging.getLogger(__name__)


class ContextTranslationDatasetReader(DatasetReader):
    """
    Read a bitext file with document boundary and create sentences pairs.
    SentA should be the so-called context sentence and SentB should be the
    sentence we care about

    Parameters
    ----------

    """

    def __init__(
        self,
        window_size: int = 6,
        context_size: int = 3,
        source_only: bool = False,
        quality_aware: bool = False,
        source_lang: str = "en",
        target_lang: str = "fr",
        source_vocabulary_size: int = 16000,
        target_vocabulary_size: int = 16000,
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        normalization_rule_name: str = "nmt_nfkc_cf",
        source_token_indexers: Dict[str, TokenIndexer] = None,
        source_max_sequence_length: int = 512,
        target_max_sequence_length: int = 512,
        concat_source_context: bool = False,
        use_target_context: bool = True,
        source_add_start_token: bool = True,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy=lazy)
        self._window_size = window_size
        self._context_size = context_size
        self._source_only = source_only
        self._quality_aware = quality_aware
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._source_vocabulary_size = source_vocabulary_size
        self._target_vocabulary_size = target_vocabulary_size
        self._source_tokenizer = source_tokenizer or SentencepieceTokenizer()
        self._target_tokenizer = target_tokenizer or SentencepieceTokenizer()
        self._normalization_rule_name = normalization_rule_name
        self._source_max_sequence_length = source_max_sequence_length
        self._target_max_sequence_length = target_max_sequence_length
        self._concat_source_context = concat_source_context
        self._use_target_context = use_target_context
        self._source_add_start_token = source_add_start_token

        self._source_token_indexers = (
            source_token_indexers
            if source_token_indexers
            else {
                "tokens": SingleIdTokenIndexer(
                    namespace="source_tokens",
                    start_tokens=[START_SYMBOL] if source_add_start_token else [],
                    end_tokens=[END_SYMBOL],
                )
            }
        )
        self._target_token_indexers = (
            None
            if source_only
            else {
                "tokens": SingleIdTokenIndexer(
                    namespace="target_tokens",
                    start_tokens=[START_SYMBOL],
                    end_tokens=[END_SYMBOL],
                )
            }
        )

    def _read_documents_from_raw_data(self, file_path):
        raise NotImplementedError()

    def _get_parallel_document(self, document):
        raise NotImplementedError()

    def _train_sentencepiece_from_list(self, sentences) -> None:
        def train_sentencepiece_from_file(train_data, vocab_size, lang, tokenizer):
            if any([lang in target for target in SPECIAL_CHARACTER_COVERAGES_LANG]):
                character_coverage = 0.9995
            else:
                character_coverage = 1.0
            with tempfile.NamedTemporaryFile(
                mode="w", prefix=f"sentencepiece_model_{lang}_", suffix=".vocab"
            ) as vocab_file:
                model_prefix = os.path.splitext(vocab_file.name)[0]
                spm.SentencePieceTrainer.train(
                    f"--bos_id=-1 --eos_id=-1 "
                    f"--user_defined_symbols={START_SYMBOL},{END_SYMBOL},{DEFAULT_OOV_TOKEN} "
                    f"--input={train_data} --model_prefix={model_prefix} "
                    f"--character_coverage={character_coverage} "
                    f"--vocab_size={vocab_size} "
                    f"--normalization_rule_name={self._normalization_rule_name}"
                )
                tokenizer.load(f"{model_prefix}.model")

        for lang, (sents, tokenizer) in sentences.items():
            with tempfile.NamedTemporaryFile(
                mode="w", prefix=f"sentencepiece_train_{lang}_", suffix=".txt"
            ) as tmp_f:
                tmp_f.write("\n".join(sents))
                tmp_f.flush()
                train_sentencepiece_from_file(
                    train_data=tmp_f.name,
                    vocab_size=self._source_vocabulary_size,
                    lang=lang,
                    tokenizer=tokenizer,
                )

    def _generate_instances(self, parallel_document):
        # Generate examples to train context sentence finder
        include_all_sentences = False
        if self._context_size == 0:
            window_size = 0
        elif self._window_size < self._context_size:
            window_size = 2 * self._context_size
            include_all_sentences = True
        else:
            window_size = self._window_size
        for index, (source, target) in enumerate(parallel_document[window_size:]):
            # When context_size if set to 0, then we generate instances for
            # a normal sentence level machine translation system
            if self._context_size == 0:
                instance = self.text_to_instance(0, 0, "", source, "", target)
                if instance:
                    yield instance
            # Otherwise, we start to generate sentence pairs
            else:
                start = window_size + index - 1
                end = start - window_size
                end = None if include_all_sentences or end < 0 else end
                for bias, (source_context, target_context) in enumerate(
                    parallel_document[start:end:-1]
                ):
                    instance = self.text_to_instance(
                        int(bias < self._context_size),
                        bias,
                        source_context,
                        source,
                        target_context,
                        target,
                    )
                    if instance:
                        yield instance

    @overrides
    def _read(self, file_path):
        docs = self._read_documents_from_raw_data(file_path)
        parallel_docs = [self._get_parallel_document(doc) for doc in docs]
        if (
            isinstance(self._source_tokenizer, SentencepieceTokenizer)
            and not self._source_tokenizer.model_path_setted()
        ):
            sentences = {
                self._source_lang: (0, self._source_tokenizer),
                self._target_lang: (1, self._target_tokenizer),
            }
            sentences = {
                lang: (
                    list(
                        itertools.chain.from_iterable(
                            [[pair[index] for pair in doc] for doc in parallel_docs]
                        )
                    ),
                    tokenizer,
                )
                for lang, (index, tokenizer) in sentences.items()
            }
            self._train_sentencepiece_from_list(sentences)
        logger.info(f"There are {len(docs)} documents")
        if self._source_only:
            example_nums = sum(
                [
                    (len(doc) - self._window_size) * self._window_size
                    for doc in parallel_docs
                ]
            )
            logger.info(f"We can construct {example_nums} source only examples")
        else:
            logger.info(
                f"There are {sum([len(doc) for doc in parallel_docs])} parallel sentences"
            )
        iterators = [self._generate_instances(doc) for doc in parallel_docs]
        for instance in itertools.chain(*iterators):
            yield instance

    @overrides
    def text_to_instance(
        self,
        label: int,
        bias: int,
        source_context: str,
        source: str,
        target_context: str,
        target: str,
    ) -> Instance:
        def _truncate_seq_pair(tokens_a: List[str], tokens_b: List[str]):
            while len(tokens_a) + len(tokens_b) > self._source_max_sequence_length - 3:
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()

        # Train a context finder model
        fields = {}
        # fields['bias'] = bias
        source_context_tokens = self._source_tokenizer.tokenize(source_context)
        source_tokens = self._source_tokenizer.tokenize(source)
        if self._source_only:
            # PretrainedTransformerTokenizer adds [CLS] and [SEP]
            source_context_tokens = source_context_tokens[1:-1]
            source_tokens = source_tokens[1:-1]
            _truncate_seq_pair(source_context_tokens, source_tokens)
            # [CLS] source context [SEP] source [SEP]
            tokens = (
                [Token(CLS_SYMBOL)]
                + source_context_tokens
                + [Token(SEP_SYMBOL)]
                + source_tokens
                + [Token(SEP_SYMBOL)]
            )
            token_type_ids = [0] * (len(source_context_tokens) + 2) + [1] * (
                len(source_tokens) + 1
            )
            fields.update(
                {
                    "tokens": TextField(tokens, self._source_token_indexers),
                    "token_type_ids": ArrayField(
                        np.array(token_type_ids), dtype=np.int
                    ),
                    "label": LabelField(str(label)),
                }
            )
            return Instance(fields)
        else:
            target_context_tokens = self._target_tokenizer.tokenize(target_context)
            target_tokens = self._target_tokenizer.tokenize(target)
            # Contextual NMT systems: 2-to-1, 2-to-2
            if self._context_size != 0:
                if self._concat_source_context:
                    source_context_tokens = []
                    source_tokens = (
                        source_context_tokens + [Token(CONCAT_SYMBOL)] + source_tokens
                    )
                if self._use_target_context:
                    target_tokens = (
                        target_context_tokens + [Token(CONCAT_SYMBOL)] + target_tokens
                    )
                fields.update(
                    {
                        "source_context_tokens": TextField(
                            source_context_tokens, self._source_token_indexers
                        ),
                        "source_tokens": TextField(
                            source_tokens, self._source_token_indexers
                        ),
                        "target_tokens": TextField(
                            target_tokens, self._target_token_indexers
                        ),
                    }
                )
            # A normal sentence level machine translation system
            else:
                fields.update(
                    {
                        "source_tokens": TextField(
                            source_tokens, self._source_token_indexers
                        ),
                        "target_tokens": TextField(
                            target_tokens, self._target_token_indexers
                        ),
                    }
                )
            if (
                len(source_tokens) > self._source_max_sequence_length
                or len(source_context_tokens) > self._source_max_sequence_length
                or len(target_tokens) > self._target_max_sequence_length
            ):
                return None
            return Instance(fields)
