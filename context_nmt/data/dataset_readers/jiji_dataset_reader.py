import collections
import glob
import logging
from typing import Dict
import json
from overrides import overrides
import re

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer

from context_nmt.data.dataset_readers.context_translation_dataset_reader import (
    ContextTranslationDatasetReader,
)

logger = logging.getLogger(__name__)


@DatasetReader.register("jiji")
class JijiDatasetReader(ContextTranslationDatasetReader):
    def __init__(
        self,
        window_size: int = 6,
        context_size: int = 3,
        source_only: bool = False,
        quality_aware: bool = False,
        score_threshold: float = 0.42,
        read_from_raw: bool = False,
        source_lang: str = "en",
        target_lang: str = "ja",
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        source_max_sequence_length: int = 50,
        target_max_sequence_length: int = 50,
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
        self._score_threshold = score_threshold
        self._read_from_raw = read_from_raw

    @overrides
    def _read_documents_from_raw_data(self, file_path):
        """
        Actually, the "raw" data files were preprocessed in json format with the
        luigi pipeline, so that this method is quite simple
        """
        file_path = cached_path(file_path)
        if self._read_from_raw:
            documents = collections.defaultdict(list)
            for text_path in glob.glob(file_path + "/*.txt"):
                with open(text_path) as source:
                    bi_texts = re.split(r"^# |\n# ", source.read())[1:]
                for bi_text in bi_texts:
                    lines = bi_text.strip().split("\n")
                    header = lines[0]
                    doc_id, sent_id, score = re.findall(
                        r"(\d*)_BODY-JE-(\d*) score=(\d\.?\d*)", header
                    )[0]
                    en_sent, ja_sent = "", ""
                    for line in lines[1:]:
                        lang, sentence = line.split(": ", maxsplit=1)
                        if lang[:2] == "en":
                            en_sent += " " + sentence
                        else:
                            ja_sent += sentence
                    documents[doc_id].append(
                        {
                            "sent_id": sent_id,
                            "en": en_sent.strip(),
                            "ja": ja_sent,
                            "score": float(score),
                        }
                    )
        else:
            with open(file_path) as source:
                documents = json.load(source)
        return list(documents.values())

    @overrides
    def _get_parallel_document(self, document):
        parallel_document = []
        for pair in document:
            source = pair[self._source_lang]
            target = pair[self._target_lang]
            score = pair["score"]
            if self._source_only and not self._quality_aware and source:
                parallel_document.append((source, ""))
            elif source and float(score) >= self._score_threshold:
                parallel_document.append((source, target))
        return parallel_document
