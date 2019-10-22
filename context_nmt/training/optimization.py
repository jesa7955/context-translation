from transformers import AdamW
from allennlp.common import Registrable
from allennlp.training.optimizers import Optimizer

Registrable._registry[Optimizer]["adamw"] = AdamW
