import collections
import glob
import logging
import json
import os
import re
import itertools

import luigi
import gokart
import sentencepiece
from sklearn.model_selection import train_test_split

logger = logging.getLogger("luigi-interface")


class MergeJijiFiles(gokart.TaskOnKart):
    task_namespace = "context_nmt"
    source_path = luigi.Parameter()
    score_threhold = luigi.FloatParameter()
    quality_aware = luigi.BoolParameter()
    score_threhold = luigi.FloatParameter()

    def requires(self):
        pass

    def output(self):
        return self.make_target("merged_jiji_splits.pkl")

    def run(self):
        documents = collections.defaultdict(list)
        for file_path in glob.glob(self.source_path + "/*.txt"):
            with open(file_path) as source:
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
        logger.info(f"There are {len(documents)} documents")
        self.dump(documents)


class GenerateJijiDataSplits(gokart.TaskOnKart):
    task_namespace = "context_nmt"
    jiji_source_path = luigi.Parameter()
    target_path = luigi.Parameter()
    dev_proportion = luigi.FloatParameter()
    test_proportion = luigi.FloatParameter()
    quality_aware = luigi.BoolParameter()
    score_threhold = luigi.FloatParameter(default=0.3)

    def requires(self):
        return MergeJijiFiles(
            source_path=self.jiji_source_path,
            quality_aware=self.quality_aware,
            score_threhold=self.score_threhold,
        )

    def output(self):
        return self.input()

    def run(self):
        documents = self.load()
        test_size = self.test_proportion
        dev_size = self.dev_proportion / (1 - test_size)
        train, test = map(
            dict, train_test_split(list(documents.items()), test_size=test_size)
        )
        train, dev = map(
            dict, train_test_split(list(train.items()), test_size=dev_size)
        )
        if not os.path.isdir(self.target_path):
            os.mkdir(self.target_path)
        for name, split in (("train", train), ("dev", dev), ("test", test)):
            with open(f"{self.target_path}/{name}.json", "w") as target:
                json.dump(split, target, ensure_ascii=False)


class TrainSentencepieceModels(gokart.TaskOnKart):
    task_namespace = "context_nmt"
    jiji_source_path = luigi.Parameter()
    target_path = luigi.Parameter()
    quality_aware = luigi.BoolParameter()
    score_threhold = luigi.FloatParameter(default=0.3)
    vocab_sizes = luigi.DictParameter(default={"en": 16000, "ja": 16000})
    character_coverages = luigi.DictParameter(default={"en": 1.0, "ja": 0.9995})
    model_type = luigi.Parameter(default="unigram")

    def requires(self):
        return MergeJijiFiles(
            source_path=self.jiji_source_path,
            quality_aware=self.quality_aware,
            score_threhold=self.score_threhold,
        )

    def output(self):
        return self.input()

    def run(self):
        documents = self.load()
        langs = ("en", "ja")
        if not os.path.isdir(self.target_path):
            os.mkdir(self.target_path)
        pairs = itertools.chain.from_iterable(
            (document for document in documents.values())
        )
        paths = {
            lang: f"{self.target_path}/sentencepiece_train_data_{lang}.txt"
            for lang in langs
        }
        files = {lang: open(path, "w") for lang, path in paths.items()}
        for pair in pairs:
            for lang in langs:
                if pair[lang] != "":
                    files[lang].write(pair[lang] + "\n")
        for target in files.values():
            target.close()
        train_params = {
            lang: (
                f"--input={paths[lang]} --model_prefix={self.target_path}/jiji_sentencepiece_{lang} "
                f"--character_coverage={self.character_coverages[lang]} "
                f"--vocab_size={self.vocab_sizes[lang]} "
                f"--model_type={self.model_type}"
            )
            for lang in langs
        }
        for lang, param in train_params.items():
            sentencepiece.SentencePieceTrainer.train(param)
