import logging
import itertools
import tempfile
import os
import random
import json
import pickle as pkl
from overrides import overrides
from typing import List, Dict, Iterable
from collections import defaultdict, Counter

import numpy as np
import sentencepiece as spm
import MeCab
import sacrebleu
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, MetadataField
from allennlp.data import Token, Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.common import Tqdm
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN

from context_nmt.data.tokenizers.sentencepiece_tokenizer import SentencepieceTokenizer
from context_nmt.pipelines.conversation_dataset_merger import read_context_index_file

CONCAT_SYMBOL = "@concat@"
SEP_SYMBOL = "[SEP]"
CLS_SYMBOL = "[CLS]"

SPECIAL_CHARACTER_COVERAGES_LANG = set(["ja", "zh", "kr"])

logger = logging.getLogger(__name__)


def read_context_indicators(path: str, key: str):
    with open(path, "rb") as source:
        data = pkl.load(source)
    context_pairs = defaultdict(dict)
    counter = 0
    for doc_id, doc in data.items():
        for sent_id, score_lists in doc.items():
            sorted_list = sorted(
                [[bias, score] for bias, score in score_lists[key].items()],
                key=lambda x: x[1],
            )
            best_context = sorted_list[0]
            if sorted_list[-1][1] != 0.0:
                context_pairs[doc_id][sent_id] = best_context
            else:
                counter += 1
    logger.info(f"scores on all systems are 0 for {counter} sentences")
    return context_pairs


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
        window_size: int = 5,
        source_lang: str = "en",
        target_lang: str = "ja",
        source_vocabulary_size: int = 16000,
        target_vocabulary_size: int = 16000,
        share_vocabulary: bool = False,
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        normalization_rule_name: str = "nmt_nfkc_cf",
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
        context_sentence_index_file: str = None,
        context_indicators_file: str = None,
        context_indicators_key: str = None,
        noisy_context_dataset_path: str = None,
        lazy: bool = False,
        cache_directory: str = None,
    ) -> None:
        super().__init__(lazy=lazy, cache_directory=cache_directory)
        self._window_size = window_size
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._source_vocabulary_size = source_vocabulary_size
        self._target_vocabulary_size = target_vocabulary_size
        self._share_vocabulary = share_vocabulary
        self._source_tokenizer = source_tokenizer or SpacyTokenizer()
        self._target_tokenizer = target_tokenizer or SpacyTokenizer()
        self._normalization_rule_name = normalization_rule_name
        self._source_max_sequence_length = source_max_sequence_length
        self._target_max_sequence_length = target_max_sequence_length
        assert translation_data_mode in [
            "2-to-1",
            "2-to-1-restricted",
            "2-to-2",
            "1-to-1",
        ]
        assert classification_data_mode in ["train", "inference", "none"]
        self._translation_data_mode = translation_data_mode
        self._classification_data_mode = classification_data_mode
        self._concat_source_context = concat_source_context
        self._source_add_start_token = source_add_start_token
        self._source_add_end_token = source_add_end_token
        self._source_add_factors = source_add_factors
        self._source_only = source_only
        if context_indicators_file:
            self._context_pairs = read_context_indicators(
                context_indicators_file, context_indicators_key
            )
            self._context_pairs_bias = True
        else:
            self._context_pairs = read_context_index_file(context_sentence_index_file)
            self._context_pairs_bias = False
        self._noisy_context_dataset = None
        if noisy_context_dataset_path:
            extension = os.path.splitext(noisy_context_dataset_path)[1]
            if extension == ".pkl":
                with open(noisy_context_dataset_path, "rb") as source:
                    self._noisy_context_dataset = pkl.load(source)
        if self._context_pairs:
            logger.info("We have some gold indicators!!!")
        else:
            logger.info("We have to generate exmaples ourselves")

        self._source_token_indexers = (
            source_token_indexers
            if source_token_indexers
            else {"tokens": SingleIdTokenIndexer(namespace="source_tokens")}
        )
        self._target_token_indexers = (
            target_token_indexers
            if target_token_indexers
            else {"tokens": SingleIdTokenIndexer(namespace="target_tokens")}
        )
        self._source_factor_indexers = (
            {"factors": SingleIdTokenIndexer(namespace="source_factors")}
            if source_add_factors
            else None
        )

    @overrides
    def _instances_from_cache_file(self, cache_filename: str) -> Iterable[Instance]:
        with open(cache_filename, "rb") as cache_file:
            for instance in pkl.load(cache_file):
                yield instance
            # for line in cache_file:
            #     yield self.deserialize_instance(line)

    @overrides
    def _instances_to_cache_file(self, cache_filename: str, instances) -> None:
        with open(cache_filename, "wb") as cache:
            pkl.dump(list(instances), cache)
            # for instance in Tqdm.tqdm(instances):
            #     cache.write(self.serialize_instance(instance) + b"\n")

    @overrides
    def serialize_instance(self, instance: Instance) -> str:
        return pkl.dumps(instance)

    @overrides
    def deserialize_instance(self, string: str) -> Instance:
        return pkl.loads(string)

    def _read_documents_from_raw_data(self, file_path):
        raise NotImplementedError()

    def _get_parallel_document(self, doc_id, doc):
        raise NotImplementedError()

    def _train_sentencepiece_from_list(self, sentences) -> None:
        def train_sentencepiece_from_file(train_data, vocab_size, lang, tokenizer):
            if any([lang in target for target in SPECIAL_CHARACTER_COVERAGES_LANG]):
                character_coverage = 0.9995
            else:
                character_coverage = 1.0
            model_prefix = os.path.splitext(tokenizer.model_path)[0]
            spm.SentencePieceTrainer.train(
                f"--bos_id=-1 --eos_id=-1 "
                f"--unk_piece={DEFAULT_OOV_TOKEN} "
                f"--input={train_data} --model_prefix={model_prefix} "
                f"--character_coverage={character_coverage} "
                f"--vocab_size={vocab_size} "
                f"--normalization_rule_name={self._normalization_rule_name}"
            )

        for lang, (sents, tokenizer, vocab_size) in sentences.items():
            with tempfile.NamedTemporaryFile(
                mode="w", prefix=f"sentencepiece_train_{lang}_", suffix=".txt"
            ) as tmp_f:
                tmp_f.write("\n".join(sents))
                tmp_f.flush()
                train_sentencepiece_from_file(
                    train_data=tmp_f.name,
                    vocab_size=vocab_size,
                    lang=lang,
                    tokenizer=tokenizer,
                )

    @overrides
    def _read(self, file_path):
        docs = self._read_documents_from_raw_data(file_path)
        logger.info(f"There are {len(docs)} documents")
        parallel_docs = {
            doc_id: self._get_parallel_document(doc_id, doc)
            for doc_id, doc in docs.items()
        }
        if isinstance(self._source_tokenizer, SentencepieceTokenizer) and (
            not self._source_tokenizer.model_trained()
            or not self._target_tokenizer.model_trained()
        ):
            source_sentence_list, target_sentence_list = [
                list(
                    itertools.chain.from_iterable([doc[lang] for doc in docs.values()])
                )
                for lang in (self._source_lang, self._target_lang)
            ]
            if self._share_vocabulary:
                sentences = {
                    self._source_lang: (
                        source_sentence_list,
                        self._source_tokenizer,
                        self._source_vocabulary_size,
                    ),
                    self._target_lang: (
                        target_sentence_list,
                        self._target_tokenizer,
                        self._target_vocabulary_size,
                    ),
                }
            else:
                sentences = {
                    f"{self._source_lang}_{self._target_lang}": (
                        source_sentence_list + target_sentence_list,
                        self._source_tokenizer,
                        self._source_vocabulary_size + self._target_vocabulary_size,
                    )
                }
            self._train_sentencepiece_from_list(sentences)
            self._source_tokenizer.load()
            self._target_tokenizer.load()
        if self._context_pairs_bias:
            context_pairs = defaultdict(dict)
            for doc_id, doc in self._context_pairs.items():
                if doc_id not in docs:
                    continue
                for sent_id, (context_bias, score) in doc.items():
                    available_indexes = [
                        index
                        for index in range(sent_id)
                        if docs[doc_id][self._source_lang][index].strip()
                    ]
                    if context_bias != -1 and len(available_indexes) < context_bias:
                        context_bias = -1
                    if context_bias != -1:
                        context_index = available_indexes[-context_bias]
                    else:
                        context_index = -1
                    self._context_pairs[doc_id][sent_id] = (context_index, score)
                    context_pairs[doc_id][sent_id] = context_bias
            logger.info(
                str(
                    Counter(
                        list(
                            itertools.chain.from_iterable(
                                [
                                    [index for index in doc.values()]
                                    for doc in context_pairs.values()
                                ]
                            )
                        )
                    )
                )
            )
        iterators = [
            self._generate_instances(doc_id, parallel_doc, docs)
            for doc_id, parallel_doc in parallel_docs.items()
        ]
        for instance in itertools.chain(*iterators):
            if (
                len(instance["source_tokens"]) <= self._source_max_sequence_length
                and len(instance["source_tokens"]) > 0
                and (
                    "target_tokens" not in instance.fields
                    or len(instance["target_tokens"])
                    <= self._target_max_sequence_length
                )
            ):
                yield instance

    def _generate_instances(self, doc_id, parallel_document, raw_documents):
        def tokenize_context(context):
            if self._source_lang == "ja":
                tagger = MeCab.Tagger()
                context = " ".join(
                    map(
                        lambda x: x.split("\t")[0],
                        tagger.parse(context).split("\n")[:-2],
                    )
                )
            return context

        for sent_id in parallel_document:
            source = raw_documents[doc_id][self._source_lang][sent_id]
            target = raw_documents[doc_id][self._target_lang][sent_id]

            # Handle translation related data generation
            if self._translation_data_mode == "1-to-1":
                yield self.text_to_instance(None, source, None, target)
            else:
                # 2-to-1
                context_sent_index = sent_id - 1
                if "2-to-1" in self._translation_data_mode:
                    target_context = None
                    # We have a magical dict which contains the indexes of context sentences !
                    if self._context_pairs:
                        try:
                            context_sent_index = self._context_pairs[doc_id][sent_id][0]
                        except:
                            logger.info(
                                f"No indicator found for doc {doc_id} sent {sent_id}"
                            )
                            continue
                    # Oh, we have to find context sentences ourselves
                    else:
                        # Previous sentence in original document is used
                        if self._translation_data_mode == "2-to-1":
                            while (
                                context_sent_index >= 0
                                and not raw_documents[doc_id][self._source_lang][
                                    context_sent_index
                                ]
                            ):
                                context_sent_index -= 1
                        # We need to find context sentences like how 2-to-2 does
                        else:
                            if context_sent_index not in parallel_document:
                                continue
                # 2-to-2
                elif self._translation_data_mode == "2-to-2":
                    if context_sent_index in parallel_document:
                        target_context = raw_documents[doc_id][self._target_lang][
                            context_sent_index
                        ]
                    else:
                        continue

                source_context = raw_documents[doc_id][self._source_lang][
                    context_sent_index
                ]
                # We have source with pesudo context (one sentence before), now generate
                # labels
                if self._classification_data_mode == "none":
                    yield self.text_to_instance(
                        source_context, source, target_context, target
                    )
                elif self._classification_data_mode == "train":
                    if self._noisy_context_dataset:
                        possible_context_ids = [
                            context_id
                            for context_id in range(sent_id)
                            if raw_documents[doc_id][self._source_lang][context_id]
                        ][-1 : -int(self._window_size + 1) : -1]
                        if not possible_context_ids:
                            continue
                        else:
                            context_sent_index = random.choice(possible_context_ids)
                            negative_source_context = random.choice(
                                self._noisy_context_dataset[self._source_lang]
                            )
                            possible_context = list(
                                map(
                                    lambda x: [
                                        tokenize_context(
                                            raw_documents[doc_id][self._source_lang][x]
                                        )
                                    ],
                                    possible_context_ids,
                                )
                            )
                            possible_context_len = min(
                                map(lambda x: len(x[0].split()), possible_context)
                            )
                            while len(
                                tokenize_context(negative_source_context).split()
                            ) < possible_context_len and any(
                                sacrebleu.corpus_bleu(
                                    tokenize_context(negative_source_context),
                                    possible_context,
                                ).counts
                            ):
                                negative_source_context = random.choice(
                                    self._noisy_context_dataset[self._source_lang]
                                )
                            source_context = raw_documents[doc_id][self._source_lang][
                                random.choice(possible_context_ids)
                            ]
                            negative_index = -1
                    else:
                        all_previous_sentences = set(
                            [
                                p_id
                                for p_id in range(sent_id - 1, -1, -1)
                                if raw_documents[doc_id][self._source_lang][
                                    p_id
                                ].strip()
                                and p_id != context_sent_index
                            ]
                        )
                        # In restricted 2-to-1 mode we can only use previous sentences
                        # that are paired with target
                        if self._translation_data_mode == "2-to-1-restricted":
                            all_previous_sentences = all_previous_sentences.intersection(
                                parallel_document
                            )
                        if all_previous_sentences:
                            negative_index = random.sample(all_previous_sentences, 1)[0]
                            negative_source_context = raw_documents[doc_id][
                                self._source_lang
                            ][negative_index]
                        else:
                            continue
                    translated_lang = f"{self._target_lang}_translated"
                    if translated_lang in raw_documents[doc_id]:
                        negative_target_context = (
                            raw_documents[doc_id][translated_lang][negative_index]
                            if target_context
                            else None
                        )
                        negative_target = raw_documents[doc_id][translated_lang][
                            sent_id
                        ]
                    else:
                        negative_target_context = (
                            raw_documents[doc_id][self._target_lang][negative_index]
                            if target_context
                            else None
                        )
                        negative_target = target
                    yield self.text_to_instance(
                        negative_source_context,
                        source,
                        negative_target_context,
                        negative_target,
                        doc_id,
                        sent_id,
                        negative_index,
                        0,
                    )
                    yield self.text_to_instance(
                        source_context,
                        source,
                        target_context,
                        target,
                        doc_id,
                        sent_id,
                        context_sent_index,
                        1,
                    )
                elif self._classification_data_mode == "inference":
                    index_range = set([-1])
                    if self._window_size > 0:
                        upbound = max(sent_id - self._window_size - 1, -1)
                    else:
                        upbound = -1
                    index_range.update(
                        set(
                            [
                                index
                                for index in range(sent_id - 1, upbound, -1)
                                if raw_documents[doc_id][self._source_lang][
                                    index
                                ].strip()
                            ]
                        )
                    )
                    for index in index_range:
                        if (
                            self._translation_data_mode
                            not in ("2-to-2", "2-to-1-restricted")
                            or index in parallel_document
                        ):
                            yield self.text_to_instance(
                                raw_documents[doc_id][self._source_lang][index],
                                source,
                                None,
                                None,
                                doc_id,
                                sent_id,
                                index,
                            )

    @overrides
    def text_to_instance(
        self,
        source_context: str,
        source: str,
        target_context: str = None,
        target: str = None,
        doc_id: int = None,
        sent_id: int = None,
        context_sent_id: int = None,
        label: int = None,
    ) -> Instance:
        fields = {}
        target_tokens = (
            [] if target is None else self._target_tokenizer.tokenize(target)
        )
        if self._translation_data_mode == "2-to-2":
            target_context_tokens = self._target_tokenizer.tokenize(target_context)
            target_tokens = (
                target_context_tokens + [Token(CONCAT_SYMBOL)] + target_tokens
            )
        if target_tokens:
            target_tokens.insert(0, Token(START_SYMBOL))
            target_tokens.append(Token(END_SYMBOL))
        if not self._source_only:
            fields["target_tokens"] = TextField(
                target_tokens, self._target_token_indexers
            )
        if self._classification_data_mode != "none":
            # PretrainedTransformerTokenizer can add special tokens by self now
            # What we want here is: [CLS] source_context [SEP] source [SEP]
            for key, value in zip(
                ("doc_id", "sent_id", "context_sent_id"),
                (doc_id, sent_id, context_sent_id),
            ):
                if value is not None:
                    fields[key] = MetadataField(value)
            if label is not None:
                fields["label"] = LabelField(str(label))
            source_tokens = self._source_tokenizer.tokenize_sentence_pair(
                source_context, source
            )
            fields["source_tokens"] = TextField(
                source_tokens, self._source_token_indexers
            )
        else:
            source_context_tokens = (
                []
                if source_context is None
                else self._source_tokenizer.tokenize(source_context)
            )
            source_tokens = (
                [] if source is None else self._source_tokenizer.tokenize(source)
            )
            if self._translation_data_mode != "1-to-1":
                if self._concat_source_context:
                    context_factor, source_factor = Token("C"), Token("S")
                    source_factors = [context_factor] * (
                        len(source_context_tokens) + 1
                    ) + [source_factor] * len(source_tokens)
                    source_tokens = (
                        source_context_tokens + [Token(CONCAT_SYMBOL)] + source_tokens
                    )
                    if self._source_add_factors:
                        fields["source_factors"] = TextField(
                            source_factors, self._source_factor_indexers
                        )
                else:
                    fields["source_context_tokens"] = TextField(
                        source_context_tokens, self._source_token_indexers
                    )
            if self._source_add_start_token:
                source_tokens.insert(0, Token(START_SYMBOL))
            if self._source_add_end_token:
                source_tokens.append(Token(END_SYMBOL))
            fields["source_tokens"] = TextField(
                source_tokens, self._source_token_indexers
            )
        return Instance(fields)
