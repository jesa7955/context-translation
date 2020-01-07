from overrides import overrides

from allennlp.modules import Attention
import torch


@Attention.register("fixed")
class FixedAttention(Attention):
    """
    A special "attention" used for BERT
    """

    def __init__(self, normalize: bool = True):
        super().__init__(normalize)

    @overrides
    def _forward_internal(
        self, vector: torch.Tensor, matrix: torch.Tensor = None
    ) -> torch.Tensor:
        return torch.ones(matrix.shape()[:-1])
