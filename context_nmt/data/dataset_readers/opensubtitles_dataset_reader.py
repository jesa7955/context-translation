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


@DatasetReader.register("opensubtitles_dataset_reader")
class OpensubtitlesDatasetReader(DatasetReader):
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

    @overrides
    def _read(self, file_path):
        def _get_file_paths(pathname: str, source: str,
                            target: str) -> Dict[str, str]:
            paths = {key: None for key in ['ids', source, target]}
            for file_path in glob.glob(pathname + "/*"):
                suffix = os.path.splitext(os.path.basename(file_path))[1][1:]
                paths[suffix] = file_path

            if not all(paths.values()):
                raise ConfigurationError("No dataset files to read")

            return paths

        file_path = cached_path(file_path)
        file_paths = _get_file_paths(file_path, self._source_lang,
                                     self._target_lang)
        ids_path, source_path, target_path = file_paths['ids'], file_paths[
            self._source_lang], file_paths[self._target_lang]
        with open(ids_path) as ids_f, open(source_path) as source_f, open(
                target_path) as target_f:
            document = {
                self._source_lang: [],
                self._target_lang: [],
                'status': []
            }
            documents = []
            for status, source, target in zip(ids_f, source_f, target_f):
                status = status.split('\t')
                if len(status) != 5:
                    if not self._source_only or np.random.uniform(
                    ) < self._sample_proportion:
                        documents.append(document)
                    document = {
                        self._source_lang: [],
                        self._target_lang: [],
                        'status': []
                    }
                else:
                    _, _, source_indexes, target_indexes, score = status
                    document[self._source_lang].append(source)
                    document[self._target_lang].append(target)
                    document['status'].append(
                        (source_indexes, target_indexes, score))

            if self._source_only:
                example_nums = sum([
                    (len(document[self._source_lang]) - self._window_size) *
                    self._window_size for document in documents
                ])
                logger.info(f"We can construct {example_nums} examples", )
            logger.info(f"There are {len(documents)} documents")
            iterators = [
                self._read_document(document) for document in documents
            ]
            for instance in itertools.chain(*iterators):
                yield instance

    def _read_document(self, document: Dict[str, List[Any]]):
        def construct_sents(document, indexes):
            indexes = list(map(int, indexes.split()))
            return ''.join([document[index - 1] for index in indexes])

        # Generate examples to train context sentence finder
        if self._source_only and not self._quality_aware:
            cleaned_pairs = [(source, "")
                             for source in document[self._source_lang]]
        else:
            cleaned_pairs = []
            for source_indexes, target_indexes, score in document['status']:
                if float(score) >= self._score_threhold:
                    source = construct_sents(document[self._source_lang],
                                             source_indexes)
                    target = construct_sents(document[self._target_lang],
                                             target_indexes)
                    cleaned_pairs.append((source, target))
        include_all_sentences = False
        if self._window_size < self._context_size:
            window_size = 2 * self._context_size
            include_all_sentences = True
        else:
            window_size = self._window_size
        for index, (source, target) in enumerate(cleaned_pairs[window_size:]):
            start = window_size + index - 1
            end = start - window_size
            end = None if include_all_sentences or end < 0 else end
            for bias, (source_context, target_context) in enumerate(
                    cleaned_pairs[start:end:-1]):
                yield self.text_to_instance(int(bias < self._context_size),
                                            bias, source_context, source,
                                            target_context, target)

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
