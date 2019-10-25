from typing import Dict, Union, Optional
import logging

from overrides import overrides
import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn import RegularizerApplicator
from allennlp.training.metrics import BLEU, Perplexity

logger = logging.getLogger(__name__)


@Model.register("contextual_seq2seq")
class ContextualSeq2seq(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        encoder=None,
        source_encoder=None,
        trainable: bool = True,
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super().__init__(vocab, regularizer)
        self._bleu = BLEU()
        self._perplexity = Perplexity()

    def forward(
        self,
        source_context_tokens: Dict[str, torch.LongTensor],
        source_tokens: Dict[str, torch.LongTensor],
        target_tokens: Dict[str, torch.LongTensor],
    ) -> Dict[str, torch.Tensor]:
        pass

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "bleu": self._bleu.get_metric(reset),
            "perplexity": self._perplexity.get_metric(reset),
        }
        return metrics
