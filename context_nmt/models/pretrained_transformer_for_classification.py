from typing import Dict, Union, Optional
import logging

from overrides import overrides
import torch

from transformers.modeling_auto import AutoModelForSequenceClassification
from transformers.configuration_auto import AutoConfig

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn import RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy

logger = logging.getLogger(__name__)


@Model.register("pretrained_transformer_for_classification")
class PretrainedTransformerForClassification(Model):
    """
    An AllenNLP Model that runs pretrained BERT,
    takes the pooled output, and adds a Linear layer on top.
    If you want an easy way to use BERT for classification, this is it.
    Note that this is a somewhat non-AllenNLP-ish model architecture,
    in that it essentially requires you to use the "bert-pretrained"
    token indexer, rather than configuring whatever indexing scheme you like.

    See `allennlp/tests/fixtures/bert/bert_for_classification.jsonnet`
    for an example of what your config might look like.

    Parameters
    ----------
    """

    def __init__(
        self,
        vocab: Vocabulary,
        model_name: str,
        dropout: float = 0.0,
        num_labels: int = 2,
        index: str = "bert",
        label_namespace: str = "label",
        trainable: bool = True,
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super().__init__(vocab, regularizer)

        if not num_labels:
            num_labels = vocab.get_vocab_size(namespace=label_namespace)
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        config.hidden_dropout_prob = dropout
        self.transformer_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config
        )
        for param in self.transformer_model.parameters():
            param.requires_grad = trainable

        self._accuracy = CategoricalAccuracy()
        self._index = index
        self._label_namespace = label_namespace

    def forward(
        self,
        tokens: Dict[str, torch.LongTensor],
        token_type_ids: torch.LongTensor,
        label: torch.IntTensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField`` (that has a bert-pretrained token indexer)
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
        input_ids = tokens[self._index]
        input_mask = (input_ids != 0).long()

        results = self.transformer_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            labels=label,
        )
        loss, logits = None, None
        results_len = len(results)
        if results_len in (1, 3):
            logits = results[0]
        else:
            loss, logits = results[0], results[1]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output_dict = {"logits": logits, "probs": probs}
        if loss:
            output_dict["loss"] = loss
            self._accuracy(logits, label)

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
