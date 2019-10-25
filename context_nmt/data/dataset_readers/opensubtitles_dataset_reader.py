from collections import defaultdict
import glob
import os
import logging
from typing import Dict
from overrides import overrides

import numpy as np
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer

from context_nmt.data.dataset_readers.context_translation_dataset_reader import (
    ContextTranslationDatasetReader,
)

logger = logging.getLogger(__name__)


@DatasetReader.register("opensubtitles_dataset_reader")
class OpensubtitlesDatasetReader(ContextTranslationDatasetReader):
    def __init__(
        self,
        window_size: int = 6,
        context_size: int = 3,
        source_only: bool = False,
        quality_aware: bool = False,
        score_threshold: float = 0.9,
        sample_proportion: float = 1.0,
        source_lang: str = "en",
        target_lang: str = "fr",
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        source_max_sequence_length: int = 512,
        target_max_sequence_length: int = 512,
        concat_source_context: bool = False,
        use_target_context: bool = True,
        source_add_start_token: bool = True,
        lazy: bool = False,
    ) -> None:
        super().__init__(
            window_size=window_size,
            context_size=context_size,
            source_only=source_only,
            quality_aware=quality_aware,
            score_threshold=score_threshold,
            source_lang=source_lang,
            target_lang=target_lang,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            source_token_indexers=source_token_indexers,
            source_max_sequence_length=source_max_sequence_length,
            target_max_sequence_length=target_max_sequence_length,
            concat_source_context=concat_source_context,
            use_target_context=use_target_context,
            source_add_start_token=source_add_start_token,
            lazy=lazy,
        )
        self._sample_proportion = sample_proportion
        self._score_threshold = score_threshold

    @overrides
    def _read_documents_from_raw_data(self, file_path):
        def _get_file_paths(pathname: str, source: str, target: str) -> Dict[str, str]:
            paths = {key: None for key in ["ids", source, target]}
            for file_path in glob.glob(pathname + "/*"):
                suffix = os.path.splitext(os.path.basename(file_path))[1][1:]
                paths[suffix] = file_path

            if not all(paths.values()):
                raise ConfigurationError("No dataset files to read")

            return paths

        file_path = cached_path(file_path)
        file_paths = _get_file_paths(file_path, self._source_lang, self._target_lang)
        ids_path, source_path, target_path = (
            file_paths["ids"],
            file_paths[self._source_lang],
            file_paths[self._target_lang],
        )
        with open(ids_path) as ids_f, open(source_path) as source_f, open(
            target_path
        ) as target_f:
            documents = []
            document = defaultdict(list)
            for status, source, target in zip(ids_f, source_f, target_f):
                status = status.split("\t")
                if len(status) != 5:
                    if (
                        not self._source_only
                        or np.random.uniform() < self._sample_proportion
                    ):
                        documents.append(document)
                    document = defaultdict(list)
                else:
                    _, _, source_indexes, target_indexes, score = status
                    document[self._source_lang].append(source)
                    document[self._target_lang].append(target)
                    document["status"].append((source_indexes, target_indexes, score))
            # There will remain a document
            if not self._source_only or np.random.uniform() < self._sample_proportion:
                documents.append(document)
        return documents

    @overrides
    def _get_parallel_document(self, document):
        def _construct_sents(document, indexes):
            indexes = list(map(int, indexes.split()))
            return "".join([document[index - 1] for index in indexes])

        if self._source_only and not self._quality_aware:
            parallel_document = [(source, "") for source in document[self._source_lang]]
        else:
            parallel_document = []
            for source_indexes, target_indexes, score in document["status"]:
                if float(score) >= self._score_threshold:
                    source = _construct_sents(
                        document[self._source_lang], source_indexes
                    )
                    target = _construct_sents(
                        document[self._target_lang], target_indexes
                    )
                    parallel_document.append((source, target))
        return parallel_document
