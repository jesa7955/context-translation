from typing import Dict, List
import collections
import glob
import logging
import json
import itertools
import os
import tempfile
import subprocess
import random

import tqdm
import luigi
import pandas as pd
import gokart
import sentencepiece as spm
import sacrebleu
import MeCab
import torch
from fairseq.models.transformer import TransformerModel
from bs4 import BeautifulSoup

from context_nmt.pipelines.jiji_dataset_merger import read_raw_jiji


logger = logging.getLogger("luigi-interface")
SPECIAL_CHARACTER_COVERAGES_LANG = set(["ja", "zh", "kr", "en_ja"])
CONCAT_TOKEN = "@@CONCAT@@"
SPLIT_NAMES = ("train", "valid", "test")


def read_seperate_files(dataset_name, source_path, sentence_level=False):
    def _read_file(path):
        data = {}
        extension = os.path.splitext(path)[1]
        if extension == ".sgm":
            with open(path) as source:
                soup = BeautifulSoup(source, "xml")
                for doc in soup.find_all("doc"):
                    doc_id = doc["docid"]
                    segs = [seg.text for seg in doc.find_all("seg")]
                    data[doc_id] = segs
        else:
            with open(path) as source:
                for index, doc in enumerate(source.read().split("\n\n")):
                    doc_id = f"{dataset_name}_{index}"
                    segs = [seg for seg in doc.split("\n") if seg]
                    data[doc_id] = segs
        return data

    docs = {lang: _read_file(path) for lang, path in source_path.items()}
    data = collections.defaultdict(dict)
    lang1, lang2 = docs.keys()
    for doc_id, doc in docs[lang1].items():
        assert len(doc) == len(docs[lang2][doc_id])
        data[doc_id][lang1] = doc
        data[doc_id][lang2] = docs[lang2][doc_id]
        data[doc_id]["pairs"] = [(index, 1.0) for index in range(len(doc))]
    if sentence_level:
        merged_data = {lang: [] for lang in docs.keys()}
        for _, doc in data.items():
            for lang in merged_data.keys():
                merged_data[lang] += doc[lang]
        data = merged_data
    return data


def read_tsv_file(dataset_name, source_path, data_langs, sentence_level=False):
    lang1, lang2 = data_langs
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    doc_id, sent_id = 0, 0
    with open(source_path) as source:
        for line in source:
            sents = [sent.strip() for sent in line.split("\t") if sent.strip()]
            if len(sents) == 0:
                doc_id += 1
                sent_id = 0
            elif len(sents) == 2:
                data[f"{dataset_name}_{doc_id}"][lang1].append(sents[0])
                data[f"{dataset_name}_{doc_id}"][lang2].append(sents[1])
                data[f"{dataset_name}_{doc_id}"]["pairs"].append((sent_id, 1.0))
                sent_id += 1
    if sentence_level:
        merged_data = {lang: [] for lang in data_langs}
        for _, doc in data.items():
            for lang in merged_data.keys():
                merged_data[lang] += doc[lang]
        data = merged_data
    return data


def train_sentencepiece_from_list(
    sentences: Dict, model_path: str, normalization_rule_name: str
):
    def train_sentencepiece_from_file(train_data, vocab_size, model_path, lang):
        model_prefix = os.path.splitext(model_path)[0]
        character_coverage = 0.9995 if lang in SPECIAL_CHARACTER_COVERAGES_LANG else 1.0
        spm.SentencePieceTrainer.train(
            f"--input={train_data} --model_prefix={model_prefix} "
            f"--character_coverage={character_coverage} "
            f"--vocab_size={vocab_size} "
            f"--normalization_rule_name={normalization_rule_name}"
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


def read_context_index_file(file_path: str, key: str = "logits", mode: str = "max"):
    context_pairs = None
    if file_path and os.path.exists(file_path):
        logger.info(f"Using {key} to filter context")
        context_pairs = collections.defaultdict(dict)
        with open(file_path, "r") as source:
            for line in source:
                model_output = json.loads(line)
                if mode == "max":
                    score = model_output[key][1]
                else:
                    score = -model_output[key][0]
                d_id, s_id, cs_id = model_output["data_indexers"]
                if (
                    not s_id in context_pairs[d_id]
                    or score > context_pairs[d_id][s_id][1]
                ):
                    context_pairs[d_id][s_id] = (cs_id, score)
    return context_pairs


class ReadDatasetFiles(gokart.TaskOnKart):
    task_namespace = "context_nmt"
    dataset_name = luigi.Parameter()
    source_path = luigi.Parameter()
    translation_model_name = luigi.Parameter(default=None)
    translation_models = luigi.DictParameter(default={})
    sentencepiece_models = luigi.DictParameter(default={})
    data_langs = luigi.ListParameter(default=["en", "ja"])
    sentence_level = luigi.BoolParameter()

    def output(self):
        if self.translation_model_name:
            return self.make_target(
                f"{self.dataset_name}_{self.translation_model_name}_merged.pkl"
            )
        else:
            return self.make_target(f"{self.dataset_name}_merged.pkl")

    def run(self):
        def load_doc(doc: List, doc_id: str):
            doc_df = pd.DataFrame(doc)
            for sent_id, row in doc_df.iterrows():
                docs[doc_id]["en"].append(str(row["en_sentence"]).strip())
                docs[doc_id]["ja"].append(str(row["ja_sentence"]).strip())
                docs[doc_id]["pairs"].append((sent_id, 1.0))

        docs = collections.defaultdict(lambda: collections.defaultdict(list))
        lang1, lang2 = self.data_langs
        if self.dataset_name == "jiji":
            read_raw_jiji(docs, self.source_path)
        elif not isinstance(self.source_path, str):
            docs = read_seperate_files(
                self.dataset_name, self.source_path, self.sentence_level
            )
        elif os.path.splitext(self.source_path)[1] == ".tsv":
            docs = read_tsv_file(
                self.dataset_name,
                self.source_path,
                self.data_langs,
                self.sentence_level,
            )
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
        if not self.sentence_level:
            for _, doc in docs.items():
                doc[lang1].append(" ")
                doc[lang2].append(" ")

        # Add translated source to the data
        if self.translation_model_name:
            langs = list(self.translation_models.keys())
            source_target_dict = {
                lang: langs[1 - index] for index, lang in enumerate(langs)
            }
            translation_models = {}
            for source, path in self.translation_models.items():
                base_path, checkpoint_path = os.path.split(path)
                model = TransformerModel.from_pretrained(
                    base_path, checkpoint_file=checkpoint_path
                )
                model.to("cuda")
                spm_processor = spm.SentencePieceProcessor()
                spm_processor.load(self.sentencepiece_models[source])
                translation_models[source] = (model, spm_processor)
            for doc_id, doc in tqdm.tqdm(docs.items(), total=len(docs)):
                for lang in translation_models.keys():
                    model = translation_models[source_target_dict[lang]][0]
                    tokenizer = translation_models[source_target_dict[lang]][1]
                    detokenizer = translation_models[lang][1]
                    sources = []
                    no_translation = {}
                    for index, sent in enumerate(doc[source_target_dict[lang]]):
                        if not sent or sent == " ":
                            no_translation[index] = sent
                        else:
                            sources.append(sent)
                    targets = [
                        detokenizer.decode_pieces(target.split())
                        for target in model.translate(
                            [
                                " ".join(tokenizer.encode_as_pieces(source))
                                for source in sources
                            ]
                        )
                    ]
                    for sent_id, sent in no_translation.items():
                        targets.insert(sent_id, sent)
                    doc[f"{lang}_translated"] = targets
        self.dump(dict(docs))


class MergeMultipleDataset(gokart.TaskOnKart):
    task_namespace = "context_nmt"
    split_name = luigi.Parameter()
    dataset_names = luigi.ListParameter()
    source_paths = luigi.ListParameter()
    translation_model_name = luigi.Parameter(default=None)
    translation_models = luigi.DictParameter(default={})
    sentencepiece_models = luigi.DictParameter(default={})
    data_langs = luigi.ListParameter(default=["en", "ja"])
    sentence_level = luigi.BoolParameter()

    def requires(self):
        return {
            name: ReadDatasetFiles(
                dataset_name=name,
                source_path=path,
                translation_model_name=self.translation_model_name,
                translation_models=self.translation_models,
                sentencepiece_models=self.sentencepiece_models,
                data_langs=self.data_langs,
                sentence_level=self.sentence_level,
            )
            for name, path in zip(self.dataset_names, self.source_paths)
        }

    def output(self):
        if self.translation_model_name:
            return self.make_target(
                f"{self.split_name}_{self.translation_model_name}.pkl"
            )
        else:
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
    data_langs = luigi.ListParameter(default=["en", "ja"])
    noisy_dataset_names = luigi.ListParameter(default=None)
    noisy_dataset_source_paths = luigi.ListParameter(default=None)
    sentence_level = luigi.BoolParameter()

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
                data_langs=self.data_langs,
                sentence_level=self.sentence_level,
            )
        if self.noisy_dataset_names:
            requirements["noise"] = MergeMultipleDataset(
                split_name="noise",
                dataset_names=self.noisy_dataset_names,
                source_paths=self.noisy_dataset_source_paths,
                data_langs=self.data_langs,
                sentence_level=True,
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
    noisy_data = None
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
    context_bias_maxmium = luigi.IntParameter(default=5)
    context_sentence_index_file = luigi.Parameter(default=None)
    context_metric_key = luigi.Parameter(default="logits")
    context_metric_mode = luigi.Parameter(default="max")
    score_threhold = luigi.FloatParameter(default=0.3)
    vocab_size = luigi.IntParameter(default=32000)
    normalization_rule_name = luigi.Parameter(default="nmt_nfkc_cf")
    sentence_level = luigi.BoolParameter()
    noisy_dataset_names = luigi.ListParameter(default=None)
    noisy_dataset_source_paths = luigi.ListParameter(default=None)
    experiment_name = luigi.Parameter()
    ensure_no_overlap = luigi.BoolParameter()

    def requires(self):
        requirements = GenerateDataSplits(
            train_dataset_names=self.train_dataset_names,
            train_source_paths=self.train_source_paths,
            valid_dataset_names=self.valid_dataset_names,
            valid_source_paths=self.valid_source_paths,
            test_dataset_names=self.test_dataset_names,
            test_source_paths=self.test_source_paths,
            noisy_dataset_names=self.noisy_dataset_names,
            noisy_dataset_source_paths=self.noisy_dataset_source_paths,
            data_langs=sorted([self.source_lang, self.target_lang]),
            sentence_level=self.sentence_level,
        )
        return requirements

    def output(self):
        outputs = {}
        for split_name in SPLIT_NAMES:
            for lang in (self.source_lang, self.target_lang):
                outputs[f"{split_name}_{lang}"] = self.make_target(
                    f"{split_name}_{self.experiment_name}.{lang}",
                    processor=gokart.file_processor.TextFileProcessor(),
                )
        return outputs

    def run(self):
        data = {split_name: self.load(split_name) for split_name in SPLIT_NAMES}
        if self.noisy_dataset_names:
            self.noisy_data = self.load("noise")
        if not os.path.exists(self.sentencepiece_model_path):
            if self.sentence_level:
                train_source = data["train"][self.source_lang]
                train_target = data["train"][self.target_lang]
            else:
                train_source = list(
                    itertools.chain.from_iterable(
                        (doc[self.source_lang] for doc in data["train"].values())
                    )
                )
                train_target = list(
                    itertools.chain.from_iterable(
                        (doc[self.target_lang] for doc in data["train"].values())
                    )
                )
            sentences = {
                "_".join((self.source_lang, self.target_lang)): (
                    train_source + train_target,
                    self.vocab_size,
                )
            }
            train_sentencepiece_from_list(
                sentences, self.sentencepiece_model_path, self.normalization_rule_name,
            )
        self.processor.load(self.sentencepiece_model_path)
        for split_name in SPLIT_NAMES:
            if not self.sentence_level:
                if str(self.data_mode) != "1-to-1":
                    self.context_pairs = read_context_index_file(
                        self.context_sentence_index_file,
                        self.context_metric_key,
                        self.context_metric_mode,
                    )
                    if self.context_pairs:
                        logger.info("We have some gold indicators!!!")
                    else:
                        logger.info("We have to generate our own context")
                pairs = self.get_pairs(data[split_name])
            else:
                pairs = self.get_sentence_level_pairs(data[split_name])
            for lang in (self.source_lang, self.target_lang):
                self.dump("\n".join(pairs[lang]), f"{split_name}_{lang}")

    def get_sentence_level_pairs(self, docs: Dict):
        for lang, doc in docs.items():
            for index, sent in enumerate(doc):
                docs[lang][index] = " ".join(self.processor.encode_as_pieces(sent))
        return docs

    def get_pairs(self, docs: Dict):
        def tokenize_context(context):
            if self.source_lang == "ja":
                tagger = MeCab.Tagger()
                context = " ".join(
                    map(
                        lambda x: x.split("\t")[0],
                        tagger.parse(context).split("\n")[:-2],
                    )
                )
            return context

        pairs = {self.source_lang: [], self.target_lang: []}
        for doc_id, doc in tqdm.tqdm(docs.items(), total=len(docs)):
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
                    if self.noisy_data:
                        noisy_context = random.choice(self.noisy_data[self.source_lang])
                        if self.ensure_no_overlap:
                            possible_context = [
                                context
                                for context in doc[self.source_lang][:sent_id]
                                if context
                            ][-1 : -int(self.context_bias_maxmium + 1) : -1]
                            possible_context = list(
                                map(lambda x: [tokenize_context(x)], possible_context)
                            )
                            while possible_context and any(
                                sacrebleu.corpus_bleu(
                                    tokenize_context(noisy_context), possible_context
                                ).counts
                            ):
                                noisy_context = random.choice(
                                    self.noisy_data[self.source_lang]
                                )
                        source_context = noisy_context
                    else:
                        if self.context_pairs is None:
                            if self.data_mode == "2-to-1":
                                available_index = [
                                    index
                                    for index in range(0, sent_id)
                                    if doc[self.source_lang][index]
                                ]
                                if len(available_index) < self.context_bias:
                                    context_sent_index = -1
                                else:
                                    context_sent_index = available_index[
                                        -int(self.context_bias)
                                    ]
                            else:
                                context_sent_index = min(
                                    sent_id - self.context_bias, -1
                                )
                                if (
                                    context_sent_index not in parallel_doc
                                    and context_sent_index != -1
                                ):
                                    continue
                        else:
                            context_sent_index = self.context_pairs[doc_id][sent_id][0]
                        source_context = docs[doc_id][self.source_lang][
                            context_sent_index
                        ]
                if source_context:
                    source_context = self.processor.encode_as_pieces(source_context)
                    source = source_context + [CONCAT_TOKEN] + source
                pairs[self.source_lang].append(" ".join(source))
                pairs[self.target_lang].append(" ".join(target))
        return pairs


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
    noisy_dataset_names = luigi.ListParameter(default=None)
    noisy_dataset_source_paths = luigi.ListParameter(default=None)
    source_lang = luigi.Parameter(default="en")
    target_lang = luigi.Parameter(default="ja")
    data_mode = luigi.Parameter(default="1-to-1")
    context_bias = luigi.IntParameter(default=1)
    context_sentence_index_file = luigi.Parameter(default=None)
    context_metric_key = luigi.Parameter(default="logits")
    context_metric_mode = luigi.Parameter(default="max")
    score_threhold = luigi.FloatParameter(default=0.3)
    vocab_size = luigi.IntParameter(default=32000)
    normalization_rule_name = luigi.Parameter(default="nmt_nfkc_cf")
    sentence_level = luigi.BoolParameter()
    # Model related parameters
    experiment_path = luigi.Parameter()
    preprocess_workers = luigi.IntParameter(default=8)
    batch_size = luigi.IntParameter(default=1024)
    save_interval = luigi.IntParameter(default=8000)
    source_max_sequence_length = luigi.IntParameter(default=128)
    target_max_sequence_length = luigi.IntParameter(default=128)
    add_source_factor = luigi.BoolParameter()
    source_factor_embed_dim = luigi.IntParameter(default=8)
    source_factor_type_num = luigi.IntParameter(default=2)
    train_random_seed = luigi.Parameter(default="42")
    use_original_lr_scheduler = luigi.BoolParameter()
    ensure_no_overlap = luigi.BoolParameter()

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
            context_metric_key=self.context_metric_key,
            score_threhold=self.score_threhold,
            vocab_size=self.vocab_size,
            normalization_rule_name=self.normalization_rule_name,
            sentence_level=self.sentence_level,
            noisy_dataset_names=self.noisy_dataset_names,
            noisy_dataset_source_paths=self.noisy_dataset_source_paths,
            experiment_name=self.get_experiment_name(),
            ensure_no_overlap=self.ensure_no_overlap,
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
            if self.context_metric_key != "logits":
                name_components.append(self.context_metric_key)
            name_components.append(self.context_metric_mode)
            if self.context_sentence_index_file:
                name_components.append(
                    f"filtered_{str(self.context_sentence_index_file).split('/')[-1]}"
                )
            else:
                name_components.append(f"context_bias_{self.context_bias}")
        if self.noisy_dataset_names:
            name_components.append("noisy")
            name_components += list(self.noisy_dataset_names)
        name_components.append("seed")
        name_components.append(self.train_random_seed)
        experiment_name = "_".join(name_components)
        return experiment_name

    def output(self):
        return self.make_target(f"{self.get_experiment_name()}.txt")

    def run(self):
        data_prefixs = {
            split_name: os.path.splitext(self.input()[f"{split_name}_en"].path())[0]
            for split_name in SPLIT_NAMES
        }
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
                " ".join(preprocess_params), shell=True, check=True,
            )

        # Train Model
        train_params = ["fairseq-train", fairseq_data_path]
        train_params += ["--seed", str(self.train_random_seed)]
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
        if not self.use_original_lr_scheduler:
            train_params += [
                "--lr",
                "1e-4",
                "--lr-scheduler",
                "reduce_lr_on_plateau",
                "--lr-shrink",
                "0.7",
            ]
        else:
            train_params += [
                "--lr",
                "5e-4",
                "--lr-scheduler",
                "inverse_sqrt",
                "--warmup-updates",
                "4000",
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
        if torch.cuda.device_count() > 1:
            self.batch_size = int(self.batch_size / torch.cuda.device_count())
        train_params += ["--max-tokens", str(self.batch_size)]
        train_params += ["--save-interval-updates", str(self.save_interval)]
        train_params += ["--no-epoch-checkpoints", "--keep-interval-updates", "5"]
        train_params += ["--save-dir", fairseq_checkpoint_path]
        train_params += [
            "--tensorboard-logdir",
            f"{fairseq_checkpoint_path}/tensorboard-log",
        ]
        train_params += ["--patience", "10", "--reset-optimizer"]
        train_params += [
            "--skip-invalid-size-inputs-valid-test",
            "--fp16-scale-tolerance",
            "0.25",
        ]
        train_return = subprocess.run(" ".join(train_params), shell=True, check=True)
        self.dump("\n".join(preprocess_params + train_params))
