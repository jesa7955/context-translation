from typing import Dict
import collections
import glob
import logging
import json
import os
import itertools
import tempfile

import luigi
import tqdm
import pandas as pd
import gokart
import sentencepiece as spm
from sklearn.model_selection import train_test_split

logger = logging.getLogger("luigi-interface")
SPECIAL_CHARACTER_COVERAGES_LANG = set(["ja", "zh", "kr", "en_ja"])


class MergeConversationFiles(gokart.TaskOnKart):
    task_namespace = "context_nmt"
    dataset_name = luigi.Parameter()
    source_path = luigi.Parameter()

    def output(self):
        return self.make_target(f"{self.dataset_name}_merged.csv")

    def run(self):
        pairs = []
        if os.path.isfile(self.source_path):
            with open(self.source_path) as source:
                data = json.load(source)
            pairs = [pd.DataFrame(doc) for doc in data]
        elif os.path.isdir(self.source_path):
            for file_path in glob.glob(self.source_path + "/*.json"):
                pairs.append(pd.read_json(file_path))
        pairs = pd.concat(pairs)
        self.dump(pairs[["en_sentence", "ja_sentence"]])


class GeneratePlainText(gokart.TaskOnKart):
    task_namespace = "context_nmt"
    split_name = luigi.Parameter()
    dataset_names = luigi.ListParameter()
    source_paths = luigi.ListParameter()

    def requires(self):
        return {
            name: MergeConversationFiles(dataset_name=name, source_path=path)
            for name, path in zip(self.dataset_names, self.source_paths)
        }

    def output(self):
        return self.make_target(f"{self.split_name}.csv")

    def run(self):
        data = pd.concat([self.load(name) for name in self.dataset_names])
        self.dump(data)


class GenerateConversationSplits(gokart.TaskOnKart):
    task_namespace = "context_nmt"
    train_dataset_names = luigi.ListParameter()
    train_source_paths = luigi.ListParameter()
    valid_dataset_names = luigi.ListParameter()
    valid_source_paths = luigi.ListParameter()
    test_dataset_names = luigi.ListParameter()
    test_source_paths = luigi.ListParameter()
    shared_vocab = luigi.BoolParameter()
    sentencepiece_model_path = luigi.Parameter()
    source_vocab_size = luigi.IntParameter(default=16000)
    target_vocab_size = luigi.IntParameter(default=16000)
    normalization_rule_name = luigi.Parameter(default="nmt_nfkc_cf")

    def requires(self):
        requiements = {}
        for split_name, (dataset_names, source_paths) in zip(
            ("train", "valid", "test"),
            (
                (self.train_dataset_names, self.train_source_paths),
                (self.valid_dataset_names, self.valid_source_paths),
                (self.test_dataset_names, self.test_source_paths),
            ),
        ):
            requiements[split_name] = GeneratePlainText(
                split_name=split_name,
                dataset_names=dataset_names,
                source_paths=source_paths,
            )
        return requiements

    def output(self):
        outputs = {}
        for split_name in ("train", "valid", "test"):
            for lang in ("en", "ja"):
                outputs[f"{split_name}_{lang}"] = self.make_target(
                    f"{split_name}.{lang}",
                    processor=gokart.file_processor.TextFileProcessor(),
                )
        return outputs

    def run(self):
        data = {
            split_name: self.load(split_name)
            for split_name in ("train", "valid", "test")
        }
        if not os.path.exists(self.sentencepiece_model_path):
            train_en = [
                str(row[1]).strip() for row in data["train"]["en_sentence"].iteritems()
            ]
            train_ja = [
                str(row[1]).strip() for row in data["train"]["ja_sentence"].iteritems()
            ]
            if self.shared_vocab:
                sentences = {
                    "en_ja": (
                        train_en + train_ja,
                        self.source_vocab_size + self.target_vocab_size,
                    )
                }
            else:
                sentences = {
                    "en": (train_en, self.source_vocab_size),
                    "ja": (train_ja, self.target_vocab_size),
                }
            self.train_sentencepiece_from_list(sentences, self.sentencepiece_model_path)
        processor = spm.SentencePieceProcessor()
        processor.load(self.sentencepiece_model_path)
        for split_name in ("train", "valid", "test"):
            for lang in ("en", "ja"):
                text = []
                for sentence in data[split_name][f"{lang}_sentence"].iteritems():
                    text.append(" ".join(processor.encode_as_pieces(str(sentence[1]))))
                self.dump("\n".join(text), f"{split_name}_{lang}")

    def train_sentencepiece_from_list(self, sentences: Dict, model_path: str):
        def train_sentencepiece_from_file(train_data, vocab_size, model_path, lang):
            model_prefix = os.path.splitext(model_path)[0]
            character_coverage = (
                0.9995 if lang in SPECIAL_CHARACTER_COVERAGES_LANG else 1.0
            )
            spm.SentencePieceTrainer.train(
                f"--input={train_data} --model_prefix={model_prefix} "
                f"--character_coverage={character_coverage} "
                f"--vocab_size={vocab_size} "
                f"--normalization_rule_name={self.normalization_rule_name}"
            )

        for lang, (sents, vocab_size) in sentences.items():
            with tempfile.NamedTemporaryFile(
                mode="w", prefix=f"sentencepiece_train_{lang}_", suffix=".txt"
            ) as tmp_f:
                tmp_f.write("\n".join(sents))
                tmp_f.flush()
                train_sentencepiece_from_file(
                    train_data=tmp_f.name,
                    vocab_size=vocab_size,
                    model_path=model_path,
                    lang=lang,
                )
