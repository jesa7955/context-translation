import collections
import logging
import json
import os

import luigi
import gokart
import tqdm
import sentencepiece as spm
import sacrebleu
from fairseq.models.transformer import TransformerModel
from fairseq.data import LanguagePairDataset

from context_nmt.pipelines.conversation_dataset_merger import (
    MergeMultipleDataset,
    CONCAT_TOKEN,
)

logger = logging.getLogger("luigi-interface")


class GenerateContextIndicator(gokart.TaskOnKart):
    task_namespace = "context_nmt"
    split_name = luigi.Parameter()
    dataset_names = luigi.ListParameter()
    source_paths = luigi.ListParameter()
    source_lang = luigi.Parameter()
    target_lang = luigi.Parameter()
    context_aware_translation_models = luigi.DictParameter()
    context_aware_sentencepiece_model = luigi.Parameter()
    max_source_positions = luigi.IntParameter(default=128)
    max_target_positions = luigi.IntParameter(default=128)
    sentence_translation_model_name = luigi.Parameter(default=None)
    sentence_translation_models = luigi.DictParameter(default={})
    sentence_sentencepiece_models = luigi.DictParameter(default={})
    score_threhold = luigi.FloatParameter(default=0.3)

    def requires(self):
        return MergeMultipleDataset(
            split_name=self.split_name,
            dataset_names=self.dataset_names,
            source_paths=self.source_paths,
            translation_model_name=self.sentence_translation_model_name,
            translation_models=self.sentence_translation_models,
            sentencepiece_models=self.sentence_sentencepiece_models,
        )

    def output(self):
        name_components = [
            self.split_name,
            self.source_lang,
            self.target_lang,
            self.sentence_translation_model_name,
        ]
        return self.make_target("_".join(name_components) + "_context_indicators.pkl")

    def run(self):
        docs = self.load()
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(self.context_aware_sentencepiece_model)
        translation_models = {}
        for bias, path in self.context_aware_translation_models.items():
            base_path, checkpoint_path = os.path.split(path)
            model = (
                TransformerModel.from_pretrained(
                    base_path, checkpoint_file=checkpoint_path
                )
                .half()
                .cuda()
            )
            translation_models[int(bias)] = model
        results = collections.defaultdict(dict)
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
                    tokenizer.encode_as_pieces(doc[lang][sent_id])
                    for lang in (self.source_lang, self.target_lang)
                ]
                available_index = [
                    index for index in range(0, sent_id) if doc[self.source_lang][index]
                ]
                bleus, losses, nll_losses = [], [], []
                # context_bias is the parameter which the model is trained with.
                # context_sent_index is the index of the actual used contextual
                # sentence.
                for context_bias, model in translation_models.items():
                    context_sent_index = None
                    if context_bias != -1:
                        if len(available_index) < context_bias:
                            context_sent_index = -1
                        else:
                            context_sent_index = available_index[-context_bias]
                        source_context = tokenizer.encode_as_pieces(
                            docs[doc_id][self.source_lang][context_sent_index]
                        )
                        real_source = source_context + [CONCAT_TOKEN] + source
                    else:
                        real_source = source
                    if (
                        len(real_source) <= self.max_source_positions
                        and len(target) < self.max_target_positions
                    ):
                        args = model.args
                        task = model.task
                        # Compute BLEU score
                        # Make the BLEU negative to easy the results computaion
                        bleus.append(
                            (
                                context_sent_index,
                                context_bias,
                                -sacrebleu.corpus_bleu(
                                    tokenizer.decode_pieces(
                                        model.translate(" ".join(real_source)).split()
                                    ),
                                    tokenizer.decode_pieces(target),
                                ).score,
                            )
                        )
                        # Compute loss
                        src_tokens = (
                            model.src_dict.encode_line(
                                " ".join(real_source), lambda x: x.split()
                            )
                            .long()
                            .unsqueeze(0)
                        )
                        src_lengths = [token.numel() for token in src_tokens]
                        tgt_tokens = (
                            model.tgt_dict.encode_line(
                                " ".join(target), lambda x: x.split()
                            )
                            .long()
                            .unsqueeze(0)
                        )
                        tgt_lengths = [token.numel() for token in tgt_tokens]
                        temp_dataset = LanguagePairDataset(
                            src_tokens,
                            src_lengths,
                            model.src_dict,
                            tgt_tokens,
                            tgt_lengths,
                            left_pad_source=args.left_pad_source,
                            left_pad_target=args.left_pad_target,
                            max_source_positions=self.max_source_positions,
                            max_target_positions=self.max_target_positions,
                        )
                        sample = temp_dataset.collater(list(temp_dataset))
                        sample["net_input"]["src_tokens"] = sample["net_input"][
                            "src_tokens"
                        ].cuda()
                        sample["net_input"]["src_lengths"] = sample["net_input"][
                            "src_lengths"
                        ].cuda()
                        sample["net_input"]["prev_output_tokens"] = sample["net_input"][
                            "prev_output_tokens"
                        ].cuda()
                        sample["target"] = sample["target"].cuda()
                        _, _, report = task.valid_step(
                            sample, model.models[0], task.build_criterion(args)
                        )
                        losses.append(
                            (context_sent_index, context_bias, report["loss"])
                        )
                        nll_losses.append(
                            (context_sent_index, context_bias, report["nll_loss"])
                        )
                result = {}
                for score_name, score_list in zip(
                    ("bleu", "loss", "nll_loss"), (bleus, losses, nll_losses)
                ):
                    if score_list:
                        result[score_name] = score_list
                results[doc_id][sent_id] = result
        self.dump(dict(results))
