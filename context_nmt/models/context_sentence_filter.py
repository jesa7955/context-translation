import collections
from typing import Dict, Union, Optional, Any
import logging
import heapq
import itertools

from overrides import overrides
import torch

from transformers.modeling_auto import AutoModel
from transformers.configuration_auto import AutoConfig
from transformers.modeling_bert import BertForNextSentencePrediction

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.seq2seq_decoders import SeqDecoder
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn import RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy

logger = logging.getLogger(__name__)


@Model.register("context_sentence_filter")
class ContextSentenceFilter(Model):
    """
    Customized version of BERT NSP. Add the capability for using seq2seq
    as an auxiliary task

    Parameters
    ----------
    """

    def __init__(
        self,
        vocab: Vocabulary,
        model_name: str,
        num_labels: int,
        translation_factor: float = 0.5,
        seq_decoder: SeqDecoder = None,
        decoding_dim: int = 512,
        target_embedding_dim: int = 512,
        load_classifier: bool = False,
        transformer_trainable: bool = True,
        classifier_traninable: bool = True,
        dropout: float = 0.0,
        index: str = "transformer",
        label_namespace: str = "label",
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super().__init__(vocab, regularizer)

        if not num_labels:
            num_labels = vocab.get_vocab_size(namespace=label_namespace)
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        for param in self.transformer.parameters():
            param.requires_grad = transformer_trainable
        # Only BERT supports loading classifier layer currently
        if load_classifier:
            self.classifier = BertForNextSentencePrediction.from_pretrained(
                model_name, config=config
            ).cls
            for param in self.classifier.parameters():
                param.requires_grad = classifier_traninable
        else:
            classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
            initializer(classifier)
            self.classifier = torch.nn.ModuleList(
                (torch.nn.Dropout(dropout), classifier)
            )

        # Add a LSTMCell for translation is specified such
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        self._index = index
        self._label_namespace = label_namespace
        self._translation_factor = translation_factor
        self._seq_decoder = seq_decoder

    def forward(  # type: ignore
        self,
        source_tokens: Dict[str, torch.LongTensor],
        target_tokens: Dict[str, torch.LongTensor] = None,
        label: torch.IntTensor = None,
        doc_id: Any = None,
        sent_id: Any = None,
        context_sent_id: Any = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        source_tokens : Dict[str, torch.LongTensor]
            From a ``TextField`` (that has a pretrained transformer (customized) token indexer)
        target_tokens : Dict[str, torch.LongTensor]
            From a ``TextField`` (that has a single id indexer)
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``

        Returns
        -------
        An output dictionary consisting of:

        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        inputs = source_tokens[self._index]
        input_ids = inputs["token_ids"]
        token_type_ids = inputs["token_type_ids"]
        input_mask = (input_ids != 0).long()

        _, pooled = self.transformer(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
        )

        logits = self.classifier(pooled)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output_dict = {"logits": logits, "probs": probs}
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            self._accuracy(logits, label)

            if self.training and target_tokens and self._seq_decoder:
                batch_size, bert_dim = pooled.shape
                state = {
                    "source_mask": torch.ones(batch_size, 1, device=pooled.device),
                    "encoder_outputs": pooled.view(batch_size, 1, bert_dim),
                }
                decoder_loss = self._seq_decoder(state, target_tokens)["loss"]
                output_dict["dec_loss"] = decoder_loss
                loss = (
                    1 - self._translation_factor
                ) * loss + self._translation_factor * decoder_loss
            output_dict["loss"] = loss
        else:
            output_dict["data_indexers"] = [
                (d_id, s_id, cs_id)
                for d_id, s_id, cs_id in zip(doc_id, sent_id, context_sent_id)
            ]

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add ``"label"`` key to the dictionary with the result.
        """
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(
                self._label_namespace
            ).get(label_idx, str(label_idx))
            classes.append(label_str)
        output_dict["label"] = classes
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        return metrics
