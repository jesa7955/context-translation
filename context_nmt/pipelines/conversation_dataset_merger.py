from typing import Dict, List
import collections
import glob
import logging
import json
import itertools
import os
import tempfile
import copy
import subprocess
import itertools

import luigi
import pandas as pd
import gokart
import sentencepiece as spm

from context_nmt.pipelines.jiji_dataset_merger import read_raw_jiji


logger = logging.getLogger("luigi-interface")
SPECIAL_CHARACTER_COVERAGES_LANG = set(["ja", "zh", "kr", "en_ja"])
CONCAT_TOKEN = "@@CONCAT@@"
SPLIT_NAMES = ("train", "valid", "test")


def read_context_index_file(file_path: str, key: str = "logits"):
    context_pairs = None
    if file_path and os.path.exists(file_path):
        context_pairs = collections.defaultdict(dict)
        with open(file_path, "r") as source:
            for line in source:
                model_output = json.loads(line)
                score = model_output[key][1]
                d_id, s_id, cs_id = model_output["data_indexers"]
                if (
                    not s_id in context_pairs[d_id]
                    or score > context_pairs[d_id][s_id][1]
                ):
                    context_pairs[d_id][s_id] = (cs_id, score)
    return context_pairs


class MergeConversationFiles(gokart.TaskOnKart):
    task_namespace = "context_nmt"
    dataset_name = luigi.Parameter()
    source_path = luigi.Parameter()

    def output(self):
        return self.make_target(f"{self.dataset_name}_merged.pkl")

    def run(self):
        def load_doc(doc: List, doc_id: str):
            doc_df = pd.DataFrame(doc)
            for sent_id, row in doc_df.iterrows():
                docs[doc_id]["en"].append(str(row["en_sentence"]).strip())
                docs[doc_id]["ja"].append(str(row["ja_sentence"]).strip())
                docs[doc_id]["pairs"].append((sent_id, 1.0))

        docs = collections.defaultdict(lambda: collections.defaultdict(list))
        if self.dataset_name == "jiji":
            read_raw_jiji(docs, self.source_path)
        else:
            if os.path.isfile(self.source_path):
                with open(self.source_path) as source:
                    data = json.load(source)
                for doc_index, doc in enumerate(data):
                    load_doc(doc, f"{self.dataset_name}_{doc_index}")
            elif os.path.isdir(self.source_path):
                for file_path in glob.glob(self.source_path + "/*.json"):
                    doc_id = os.path.splitext(file_path)[0].split("/")[-1]
                    with open(file_path) as source:
                        load_doc(json.load(source), doc_id)
        for doc_id, doc in docs.items():
            doc["en"].append(" ")
            doc["ja"].append(" ")
        self.dump(dict(docs))


class MergeMultipleDataset(gokart.TaskOnKart):
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
        return self.make_target(f"{self.split_name}.pkl")

    def run(self):
        data = dict()
        for name in self.dataset_names:
            data.update(self.load(name))
        self.dump(data)


class GenerateDataSplits(gokart.TaskOnKart):
    task_namespace = "context_nmt"
    train_dataset_names = luigi.ListParameter()
    train_source_paths = luigi.ListParameter()
    valid_dataset_names = luigi.ListParameter()
    valid_source_paths = luigi.ListParameter()
    test_dataset_names = luigi.ListParameter()
    test_source_paths = luigi.ListParameter()

    def requires(self):
        requirements = {}
        for split_name, (dataset_names, source_paths) in zip(
            SPLIT_NAMES,
            (
                (self.train_dataset_names, self.train_source_paths),
                (self.valid_dataset_names, self.valid_source_paths),
                (self.test_dataset_names, self.test_source_paths),
            ),
        ):
            requirements[split_name] = MergeMultipleDataset(
                split_name=split_name,
                dataset_names=dataset_names,
                source_paths=source_paths,
            )
        return requirements

    def output(self):
        outputs = self.input()
        outputs["all"] = self.make_target(
            "_".join(
                itertools.chain.from_iterable(
                    (
                        self.train_dataset_names,
                        self.valid_dataset_names,
                        self.test_dataset_names,
                    )
                )
            )
            + ".pkl"
        )
        return outputs

    def run(self):
        data = {split_name: self.load(split_name) for split_name in SPLIT_NAMES}
        merged = {}
        for split in data.values():
            merged.update(split)
        data["all"] = merged
        for key, value in data.items():
            self.dump(value, key)


class GenerateFairseqDataSplits(gokart.TaskOnKart):
    task_namespace = "context_nmt"
    context_pairs = None
    processor = spm.SentencePieceProcessor()
    train_dataset_names = luigi.ListParameter()
    train_source_paths = luigi.ListParameter()
    valid_dataset_names = luigi.ListParameter()
    valid_source_paths = luigi.ListParameter()
    test_dataset_names = luigi.ListParameter()
    test_source_paths = luigi.ListParameter()
    sentencepiece_model_path = luigi.Parameter()
    source_lang = luigi.Parameter(default="en")
    target_lang = luigi.Parameter(default="ja")
    data_mode = luigi.Parameter(default="1-to-1")
    context_bias = luigi.IntParameter(default=1)
    context_sentence_index_file = luigi.Parameter(default=None)
    score_threhold = luigi.FloatParameter(default=0.3)
    vocab_size = luigi.IntParameter(default=32000)
    normalization_rule_name = luigi.Parameter(default="nmt_nfkc_cf")

    def requires(self):
        return GenerateDataSplits(
            train_dataset_names=self.train_dataset_names,
            train_source_paths=self.train_source_paths,
            valid_dataset_names=self.valid_dataset_names,
            valid_source_paths=self.valid_source_paths,
            test_dataset_names=self.test_dataset_names,
            test_source_paths=self.test_source_paths,
        )

    def output(self):
        outputs = {}
        for split_name in SPLIT_NAMES:
            for lang in (self.source_lang, self.target_lang):
                outputs[f"{split_name}_{lang}"] = self.make_target(
                    f"{split_name}.{lang}",
                    processor=gokart.file_processor.TextFileProcessor(),
                )
        return outputs

    def run(self):
        self.context_pairs = read_context_index_file(self.context_sentence_index_file)
        if self.context_pairs:
            logger.info("We have some gold indicators!!!")
        else:
            logger.info("We have to generate our own context")
        data = {split_name: self.load(split_name) for split_name in SPLIT_NAMES}
        if not os.path.exists(self.sentencepiece_model_path):
            train_en = list(
                itertools.chain.from_iterable(
                    (doc["en"] for doc in data["train"].values())
                )
            )
            train_ja = list(
                itertools.chain.from_iterable(
                    (doc["ja"] for doc in data["train"].values())
                )
            )
            sentences = {"en_ja": (train_en + train_ja, self.vocab_size)}
            self.train_sentencepiece_from_list(sentences, self.sentencepiece_model_path)
        self.processor.load(self.sentencepiece_model_path)
        for split_name in SPLIT_NAMES:
            pairs = self.get_pairs(data[split_name])
            for lang in (self.source_lang, self.target_lang):
                self.dump("\n".join(pairs[lang]), f"{split_name}_{lang}")

    def get_pairs(self, docs: Dict):
        pairs = {self.source_lang: [], self.target_lang: []}
        for doc_id, doc in docs.items():
            parallel_doc = set(
                [
                    sent_id
                    for sent_id, score in doc["pairs"]
                    if score >= self.score_threhold
                ]
            )
            for sent_id in parallel_doc:
                source, target = [
                    self.processor.encode_as_pieces(doc[lang][sent_id])
                    for lang in (self.source_lang, self.target_lang)
                ]
                source_context = None
                if "2-to-1" in str(self.data_mode):
                    if self.context_pairs is None:
                        context_sent_index = (
                            -1
                            if sent_id < self.context_bias
                            else sent_id - self.context_bias
                        )
                        if self.data_mode == "2-to-1":
                            while (
                                context_sent_index >= 0
                                and not docs[doc_id][self.source_lang][
                                    context_sent_index
                                ]
                            ):
                                context_sent_index -= 1
                        else:
                            if context_sent_index not in parallel_doc:
                                continue
                    else:
                        context_sent_index = self.context_pairs[doc_id][sent_id][0]
                    source_context = docs[doc_id][self.source_lang][context_sent_index]
                if source_context:
                    source_context = self.processor.encode_as_pieces(source_context)
                    source = source_context + [CONCAT_TOKEN] + source
                pairs[self.source_lang].append(" ".join(source))
                pairs[self.target_lang].append(" ".join(target))
        return pairs

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


class RunFairseqTraining(gokart.TaskOnKart):
    task_namespace = "context_nmt"
    # Data related parameters
    train_dataset_names = luigi.ListParameter()
    train_source_paths = luigi.ListParameter()
    valid_dataset_names = luigi.ListParameter()
    valid_source_paths = luigi.ListParameter()
    test_dataset_names = luigi.ListParameter()
    test_source_paths = luigi.ListParameter()
    sentencepiece_model_path = luigi.Parameter()
    source_lang = luigi.Parameter(default="en")
    target_lang = luigi.Parameter(default="ja")
    data_mode = luigi.Parameter(default="1-to-1")
    context_bias = luigi.IntParameter(default=1)
    context_sentence_index_file = luigi.Parameter(default=None)
    score_threhold = luigi.FloatParameter(default=0.3)
    vocab_size = luigi.IntParameter(default=32000)
    normalization_rule_name = luigi.Parameter(default="nmt_nfkc_cf")
    # Model related parameters
    experiment_path = luigi.Parameter()
    cuda_device = luigi.IntParameter()
    preprocess_workers = luigi.IntParameter(default=8)
    batch_size = luigi.IntParameter(default=1024)
    save_interval = luigi.IntParameter(default=8000)
    source_max_sequence_length = luigi.IntParameter(default=128)
    target_max_sequence_length = luigi.IntParameter(default=128)
    add_source_factor = luigi.BoolParameter()
    source_factor_embed_dim = luigi.IntParameter(default=8)
    source_factor_type_num = luigi.IntParameter(default=2)

    def requires(self):
        return GenerateFairseqDataSplits(
            train_dataset_names=self.train_dataset_names,
            train_source_paths=self.train_source_paths,
            valid_dataset_names=self.valid_dataset_names,
            valid_source_paths=self.valid_source_paths,
            test_dataset_names=self.test_dataset_names,
            test_source_paths=self.test_source_paths,
            sentencepiece_model_path=self.sentencepiece_model_path,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            data_mode=self.data_mode,
            context_bias=self.context_bias,
            context_sentence_index_file=self.context_sentence_index_file,
            score_threhold=self.score_threhold,
            vocab_size=self.vocab_size,
            normalization_rule_name=self.normalization_rule_name,
        )

    def get_experiment_name(self):
        name_components = list(self.train_dataset_names) + [
            self.data_mode,
            self.source_lang,
            self.target_lang,
        ]
        if self.add_source_factor:
            name_components.append("factored")
        if self.data_mode != "1-to-1":
            if self.context_sentence_index_file:
                name_components.append(
                    f"filtered_{str(self.context_sentence_index_file).split('/')[-1]}"
                )
            else:
                name_components.append(f"context_bias_{self.context_bias}")
        experiment_name = "_".join(name_components)
        return experiment_name

    def output(self):
        return self.make_target(f"{self.get_experiment_name()}.txt")

    def run(self):
        data_prefixs = {
            split_name: os.path.splitext(self.input()[f"{split_name}_en"].path())[0]
            for split_name in SPLIT_NAMES
        }
        enviroment_variables = copy.deepcopy(os.environ)
        enviroment_variables["CUDA_VISIBLE_DEVICES"] = str(self.cuda_device)
        fairseq_data_path = (
            f"{self.experiment_path}/data-bin/{self.get_experiment_name()}"
        )
        fairseq_checkpoint_path = f"{self.experiment_path}/{self.get_experiment_name()}"
        logger.info(f"fairseq data is put at {fairseq_data_path}")
        logger.info(f"fairseq model is put at {fairseq_checkpoint_path}")

        # Prepare Data
        preprocess_params = ["fairseq-preprocess", "--joined-dictionary"]
        if self.add_source_factor:
            preprocess_params += ["--task", "factored_translation"]
            preprocess_params += ["--user-dir", "context_nmt"]
        preprocess_params += ["--workers", str(self.preprocess_workers)]
        preprocess_params += ["--source-lang", self.source_lang]
        preprocess_params += ["--target-lang", self.target_lang]
        preprocess_params += ["--destdir", fairseq_data_path]
        for split_name in SPLIT_NAMES:
            preprocess_params += [f"--{split_name}pref", data_prefixs[split_name]]
        if not os.path.exists(fairseq_data_path):
            preprocess_return = subprocess.run(
                " ".join(preprocess_params),
                shell=True,
                check=True,
                env=enviroment_variables,
            )

        # Train Model
        train_params = ["fairseq-train", fairseq_data_path]
        train_params += [
            "--eval-bleu",
            "--eval-bleu-args",
            "'{\"beam\": 6}'",
        ]
        train_params += ["--tokenizer", "space"]
        if self.add_source_factor:
            train_params += ["--arch", "factored_transformer"]
            train_params += ["--task", "factored_translation"]
            train_params += ["--user-dir", "context_nmt"]
            train_params += [
                "--source-factor-embed-dim",
                str(self.source_factor_embed_dim),
            ]
            train_params += [
                "--source-factor-type-num",
                str(self.source_factor_type_num),
            ]
        else:
            train_params += ["--arch", "transformer"]
            train_params += ["--fp16"]
        train_params += ["--share-decoder-input-output-embed", "--share-all-embeddings"]
        train_params += ["--optimizer", "adam", "--adam-betas", "'(0.9, 0.98)'"]
        train_params += ["--max-source-positions", str(self.source_max_sequence_length)]
        train_params += ["--max-target-positions", str(self.target_max_sequence_length)]
        train_params += [
            "--lr",
            "1e-4",
            "--lr-scheduler",
            "reduce_lr_on_plateau",
            "--lr-shrink",
            "0.7",
        ]
        train_params += [
            "--dropout",
            "0.2",
            "--attention-dropout",
            "0.2",
            "--activation-dropout",
            "0.2",
        ]
        train_params += [
            "--criterion",
            "label_smoothed_cross_entropy",
            "--label-smoothing",
            "0.1",
        ]
        train_params += ["--encoder-normalize-before", "--decoder-normalize-before"]
        train_params += ["--max-tokens", str(self.batch_size)]
        train_params += ["--save-interval-updates", str(self.save_interval)]
        train_params += ["--no-epoch-checkpoints", "--keep-interval-updates", "5"]
        train_params += ["--save-dir", fairseq_checkpoint_path]
        train_params += [
            "--tensorboard-logdir",
            f"{fairseq_checkpoint_path}/tensorboard-log",
        ]
        train_params += ["--patience", "10", "--reset-optimizer"]
        train_return = subprocess.run(
            " ".join(train_params), shell=True, check=True, env=enviroment_variables,
        )
        self.dump(" ".join(preprocess_params + train_params))
