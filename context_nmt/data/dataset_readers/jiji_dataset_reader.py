import logging
from typing import Dict
import json
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer

from context_nmt.data.dataset_readers.context_translation_dataset_reader import (
    ContextTranslationDatasetReader)

logger = logging.getLogger(__name__)


@DatasetReader.register("jiji_dataset_reader")
class JijiDatasetReader(ContextTranslationDatasetReader):
    def __init__(self,
                 window_size: int = 6,
                 context_size: int = 3,
                 source_only: bool = False,
                 quality_aware: bool = False,
                 score_threhold: float = 0.9,
                 source_lang: str = 'en',
                 target_lang: str = 'ja',
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 max_sequence_length: int = 512,
                 concat_context: bool = False,
                 use_source_context: bool = True,
                 use_target_context: bool = True,
                 source_add_start_token: bool = True,
                 lazy: bool = False) -> None:
        super().__init__(window_size=window_size,
                         context_size=context_size,
                         source_only=source_only,
                         quality_aware=quality_aware,
                         score_threhold=score_threhold,
                         source_lang=source_lang,
                         target_lang=target_lang,
                         source_tokenizer=source_tokenizer,
                         target_tokenizer=target_tokenizer,
                         source_token_indexers=source_token_indexers,
                         max_sequence_length=max_sequence_length,
                         concat_context=concat_context,
                         use_source_context=use_source_context,
                         use_target_context=use_target_context,
                         source_add_start_token=source_add_start_token,
                         lazy=lazy)

    @overrides
    def _read_documents_from_raw_data(self, file_path):
        """
        Actually, the "raw" data files were preprocessed in json format with the
        luigi pipeline, so that this method is quite simple
        """
        file_path = cached_path(file_path)
        with open(file_path) as source:
            documents = json.load(source)
        return list(documents.values())

    @overrides
    def _get_parallel_document(self, document):
        parallel_document = []
        for pair in document:
            source = pair[self._source_lang]
            target = pair[self._target_lang]
            score = pair['score']
            if self._source_only and not self._quality_aware and source:
                parallel_document.append((source, ""))
            elif float(score) >= self._score_threhold:
                parallel_document.append((source, target))
        return parallel_document
