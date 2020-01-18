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
        score_threshold: float = float("-inf"),
        read_from_raw: bool = False,
        source_lang: str = "en",
        target_lang: str = "ja",
        source_vocabulary_size: int = 16000,
        target_vocabulary_size: int = 16000,
        share_vocabulary: bool = False,
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        target_token_indexers: Dict[str, TokenIndexer] = None,
        source_max_sequence_length: int = 128,
        target_max_sequence_length: int = 128,
        translation_data_mode: str = "2-to-1",
        classification_data_mode: str = "train",
        concat_source_context: bool = True,
        source_add_start_token: bool = False,
        source_add_end_token: bool = False,
        source_add_factors: bool = False,
        source_only: bool = False,
        lazy: bool = False,
        cache_directory: str = None,
    ) -> None:
        super().__init__(
            window_size=window_size,
            context_size=context_size,
            source_lang=source_lang,
            target_lang=target_lang,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            source_vocabulary_size=source_vocabulary_size,
            target_vocabulary_size=target_vocabulary_size,
            source_token_indexers=source_token_indexers,
            target_token_indexers=target_token_indexers,
            source_max_sequence_length=source_max_sequence_length,
            target_max_sequence_length=target_max_sequence_length,
            translation_data_mode=translation_data_mode,
            classification_data_mode=classification_data_mode,
            concat_source_context=concat_source_context,
            source_add_start_token=source_add_start_token,
            source_add_end_token=source_add_end_token,
            source_add_factors=source_add_factors,
            source_only=source_only,
            lazy=lazy,
            cache_directory=cache_directory,
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
            documents = collections.defaultdict(collections.defaultdict(list))
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
                    score, sent_id = float(score), int(sent_id)
                    documents[doc_id]["en"].append(en_sent.strip())
                    documents[doc_id]["ja"].append(ja_sent)
                    documents[doc_id]["pairs"].append((sent_id - 1, score))
        else:
            with open(file_path) as source:
                documents = json.load(source)
        return documents

    @overrides
    def _get_parallel_document(self, doc_id, doc):
        parallel_document = set()
        for sent_id, score in doc["pairs"]:
            if score >= self._score_threshold:
                parallel_document.add(sent_id)
        return parallel_document
