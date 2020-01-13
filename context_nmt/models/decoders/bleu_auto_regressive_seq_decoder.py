from typing import Dict, List, Tuple, Optional
from overrides import overrides
import logging

import numpy
import torch
import torch.nn.functional as F
from torch.nn import Linear

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.modules.seq2seq_decoders import AutoRegressiveSeqDecoder, SeqDecoder
from allennlp.data import Vocabulary
from allennlp.modules import Embedding
from allennlp.modules.seq2seq_decoders.decoder_net import DecoderNet
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import Metric, BLEU

logger = logging.getLogger(__name__)


@SeqDecoder.register("bleu_auto_regressive_seq_decoder")
class BleuAutoRegressiveSeqDecoder(AutoRegressiveSeqDecoder):
    """
    Dirty hack to used AutoRegressiveSeqDecoder with BLEU properly
    Parameters
    ----------
    vocab : Vocabulary, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (Please ask your administrator.) or the target tokens can have a different namespace, in which case it needs to
        be specified as .
    decoder_net : DecoderNet, required
        Module that contains implementation of neural network for decoding output elements
    max_decoding_steps : int, required
        Maximum length of decoded sequences.
    target_embedder : Embedding, required
        Embedder for target tokens.
    target_namespace : str, optional (default = 'tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    beam_size : int, optional (default = 4)
        Width of the beam for beam search.
    tensor_based_metric : Metric, optional (default = None)
        A metric to track on validation data that takes raw tensors when its called.
        This metric must accept two arguments when called: a batched tensor
        of predicted token indices, and a batched tensor of gold token indices.
    token_based_metric : Metric, optional (default = None)
        A metric to track on validation data that takes lists of lists of tokens
        as input. This metric must accept two arguments when called, both
        of type . The first is a predicted sequence for each item
        in the batch and the second is a gold sequence for each item in the batch.
    scheduled_sampling_ratio : float optional (default = 0)
        Defines ratio between teacher forced training and real output usage. If its zero
        (teacher forcing only) and supports parallel decoding, we get the output
        predictions in a single forward pass of the .
    bleu_exclude_tokens : List, optional (defalut = [])
        Token list to ignore when computing BLEU
    """

    def __init__(
        self,
        vocab: Vocabulary,
        decoder_net: DecoderNet,
        max_decoding_steps: int,
        target_embedder: Embedding,
        target_namespace: str = "tokens",
        tie_output_embedding: bool = False,
        scheduled_sampling_ratio: float = 0,
        label_smoothing_ratio: Optional[float] = None,
        beam_size: int = 4,
        tensor_based_metric: Metric = None,
        token_based_metric: Metric = None,
        bleu_exclude_tokens: List = [],
    ) -> None:
        super().__init__(
            vocab=vocab,
            decoder_net=decoder_net,
            max_decoding_steps=max_decoding_steps,
            target_embedder=target_embedder,
            target_namespace=target_namespace,
            tie_output_embedding=tie_output_embedding,
            scheduled_sampling_ratio=scheduled_sampling_ratio,
            label_smoothing_ratio=label_smoothing_ratio,
            beam_size=beam_size,
            tensor_based_metric=tensor_based_metric,
            token_based_metric=token_based_metric,
        )
        if isinstance(self._tensor_based_metric, BLEU):
            pad_index = self._vocab.get_token_index(
                self._vocab._padding_token, self._target_namespace
            )
            new_exclude_indices = set([pad_index])
            for token in bleu_exclude_tokens:
                new_exclude_indices.add(
                    self._vocab.get_token_index(token, self._target_namespace)
                )
            old_bleu = self._tensor_based_metric
            new_exclude_indices.update(old_bleu._exclude_indices)
            logger.info(
                f"Reconstruct BLEU to exclude {' '.join(map(str, new_exclude_indices))}"
            )
            self._tensor_based_metric = BLEU(
                old_bleu._ngram_weights, new_exclude_indices
            )
