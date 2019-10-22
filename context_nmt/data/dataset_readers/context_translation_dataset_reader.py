import glob
import os
import logging
import itertools
from overrides import overrides
from typing import List, Dict, Tuple, Any

import numpy as np
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ArrayField, MetadataField
from allennlp.data import Token, Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.common.util import START_SYMBOL, END_SYMBOL

CONCAT_SYMBOL = '@concat@'
SEP_SYMBOL = '[SEP]'
CLS_SYMBOL = '[CLS]'

logger = logging.getLogger(__name__)


class ContextTranslationDatasetReader(DatasetReader):
    """
    Read a bitext file with document boundary and create sentences pairs.
    SentA should be the so-called context sentence and SentB should be the
    sentence we care about

    Parameters
    ----------

    """
    def __init__(self,
                 window_size: int = 6,
                 context_size: int = 3,
                 source_only: bool = False,
                 quality_aware: bool = False,
                 score_threhold: float = 0.9,
                 sample_proportion: float = 1.0,
                 source_lang: str = 'en',
                 target_lang: str = 'fr',
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 max_sequence_length: int = 512,
                 concat_context: bool = False,
                 use_source_context: bool = True,
                 use_target_context: bool = True,
                 source_add_start_token: bool = True,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self._window_size = window_size
        self._context_size = context_size
        self._source_only = source_only
        self._quality_aware = quality_aware
        self._score_threhold = score_threhold
        self._sample_proportion = sample_proportion
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._source_tokenizer = source_tokenizer or SpacyTokenizer(
            f"{source_lang}_core_web_sm")
        if not source_only:
            self._target_tokenizer = target_tokenizer if target_tokenizer else SpacyTokenizer(
                f"{target_lang}_core_web_sm")
        self._max_sequence_length = max_sequence_length
        self._concat_context = concat_context
        self._use_source_context = use_source_context
        self._use_target_context = use_target_context
        self._source_add_start_token = source_add_start_token

        self._source_token_indexers = source_token_indexers if source_token_indexers else {
            'tokens':
            SingleIdTokenIndexer(
                namespace='source',
                start_tokens=[START_SYMBOL] if source_add_start_token else [],
                end_tokens=[END_SYMBOL])
        }
        self._target_token_indexers = None if source_only else {
            "tokens":
            SingleIdTokenIndexer(namespace='target',
                                 start_tokens=[START_SYMBOL],
                                 end_tokens=[END_SYMBOL])
        }

    def _read_documents_from_raw_data(self, file_path):
        raise NotImplementedError

    def _get_parallel_document(self, document):
        raise NotImplementedError

    def _generate_instances(self, parallel_document):
        # Generate examples to train context sentence finder
        include_all_sentences = False
        if self._window_size < self._context_size:
            window_size = 2 * self._context_size
            include_all_sentences = True
        else:
            window_size = self._window_size
        for index, (source,
                    target) in enumerate(parallel_document[window_size:]):
            start = window_size + index - 1
            end = start - window_size
            end = None if include_all_sentences or end < 0 else end
            for bias, (source_context, target_context) in enumerate(
                    parallel_document[start:end:-1]):
                yield self.text_to_instance(int(bias < self._context_size),
                                            bias, source_context, source,
                                            target_context, target)

    @overrides
    def _read(self, file_path):
        docs = self._read_documents_from_raw_data(file_path)
        parallel_docs = [self._get_parallel_document(doc) for doc in docs]
        iterators = [self._generate_instances(doc) for doc in parallel_docs]
        for instance in itertools.chain(*iterators):
            yield instance

    @overrides
    def text_to_instance(self, label: int, bias: int, source_context: str,
                         source: str, target_context: str,
                         target: str) -> Instance:
        def _truncate_seq_pair(tokens_a: List[str], tokens_b: List[str]):
            while len(tokens_a) + len(
                    tokens_b) > self._max_sequence_length - 3:
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
            # PretrianedTransformerTokenizer adds [CLS] and [SEP]
            source_context_tokens = source_context_tokens[1:-1]
            source_tokens = source_tokens[1:-1]
            _truncate_seq_pair(source_context_tokens, source_tokens)
            # [CLS] source context [SEP] source [SEP]
            tokens = [Token(CLS_SYMBOL)] + source_context_tokens + [
                Token(SEP_SYMBOL)
            ] + source_tokens + [Token(SEP_SYMBOL)]
            token_type_ids = [0] * (len(source_context_tokens) +
                                    2) + [1] * (len(source_tokens) + 1)
            fields['tokens'] = TextField(tokens, self._source_token_indexers)
            fields['token_type_ids'] = ArrayField(np.array(token_type_ids),
                                                  dtype=np.int)
            fields['label'] = LabelField(str(label))
        else:
            target_context_tokens = self._target_tokenizer.tokenize(
                target_context)
            target_tokens = self._target_tokenizer.tokenize(target)
            if self._use_source_context:
                if self._concat_context:
                    fields['source'] = TextField(
                        source_context_tokens + [Token(CONCAT_SYMBOL)] +
                        source_tokens, self._source_tokenizer)
                else:
                    fields['source_context'] = TextField(
                        source_context_tokens, self._source_token_indexers)
                    fields['source'] = TextField(source_tokens,
                                                 self._source_token_indexers)
            else:
                fields['source'] = TextField(source_tokens,
                                             self._source_token_indexers)
            if self._use_target_context:
                fields['target'] = TextField(
                    target_context_tokens + [Token(CONCAT_SYMBOL)] +
                    target_tokens, self._target_token_indexers)
            else:
                fields['target'] = TextField(target_tokens,
                                             self._target_token_indexers)
        return Instance(fields)
