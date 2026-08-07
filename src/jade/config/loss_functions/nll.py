"""
Negative log-likelihood loss configuration.

:class:`~jade.config.loss_functions.nll.nll_config` mirrors the parameters of :class:`torch.nn.NLLLoss`.
This loss expects log-probabilities as input (i.e. the model should end with a ``LogSoftmax`` layer).
Use it with a model whose classification head applies a ``LogSoftmax`` (e.g. :class:`~jade.model.EEGNet.EEGNet_Classifier` built with ``use_softmax = True``, which is also what its ``classify`` method assumes).
If the model outputs raw logits use :class:`~jade.config.loss_functions.cross_entropy.cross_entropy_config` instead.
The loss module is built from this config by :func:`~jade.training.support_training.get_loss_function`.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from dataclasses import dataclass
from typing import List, Optional

# Internal imports
from jade.config.loss_functions.generic import loss_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module constants

# Reduction modes accepted by torch.
REDUCTIONS = ("none", "mean", "sum")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# NLL config class

@dataclass
class nll_config(loss_config) :
    """
    Configuration for :class:`torch.nn.NLLLoss`.

    Parameters
    ----------
    weight : list of float or None, default None
        Manual rescaling weight given to each class (a list of ``n_classes`` values).
        Stored as a plain list in the config; :func:`~jade.training.support_training.get_loss_function` turns it into a tensor.
        ``None`` means every class is weighted equally.
    ignore_index : int, default -100
        Target value that is ignored and does not contribute to the gradient.
    reduction : str, default 'mean'
        One of ``REDUCTIONS``.
    """

    name : str = "nll"

    weight       : Optional[List[float]] = None
    ignore_index : int = -100
    reduction    : str = "mean"


    def __post_init__(self) -> None :
        """
        Validate the NLL parameters and freeze the loss name.
        """

        # Freeze the discriminator so the config always serializes as an NLL config.
        self.name = "nll"

        # Class weights (must be a list/tuple of numbers when provided)
        if self.weight is not None :
            if not isinstance(self.weight, (list, tuple)) :
                raise ValueError("weight must be a list of floats or None, got {}".format(type(self.weight)))
            self.weight = [float(single_weight) for single_weight in self.weight]

        # Reduction mode
        self.reduction = str(self.reduction).lower()
        if self.reduction not in REDUCTIONS :
            raise ValueError("reduction must be one of {}, got {}".format(REDUCTIONS, repr(self.reduction)))
