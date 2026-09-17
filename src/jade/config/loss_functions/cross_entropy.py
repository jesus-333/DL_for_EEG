"""
Cross-entropy loss configuration.

:class:`~jade.config.loss_functions.cross_entropy.cross_entropy_config` mirrors the parameters of :class:`torch.nn.CrossEntropyLoss`.
This loss expects raw, un-normalized scores (logits): it applies ``log_softmax`` internally.
Use it with a model whose classification head does NOT apply a softmax (e.g. :class:`~jade.model.EEGNet.EEGNet_Classifier` built with ``use_softmax = False``).
If the model already outputs log-probabilities (``use_softmax = True``) use :class:`~jade.config.loss_functions.nll.nll_config` instead, to avoid applying ``log_softmax`` twice.
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
# Cross-entropy config class

@dataclass
class cross_entropy_config(loss_config) :
    """
    Configuration for :class:`torch.nn.CrossEntropyLoss`.

    Parameters
    ----------
    class_weight : list of float or None, default None
        Manual rescaling weight given to each class (a list of ``n_classes`` values), i.e. the ``weight`` argument of the torch loss.
        It is named ``class_weight`` (and not ``weight``, as in torch) to make clear that these are *per-class* coefficients, used to counter a class imbalance, and not the trainable weights of the model.
        Stored as a plain list in the config; :func:`~jade.training.support_training.get_loss_function` turns it into a tensor.
        ``None`` means every class is weighted equally.
    ignore_index : int, default -100
        Target value that is ignored and does not contribute to the gradient.
    reduction : str, default 'mean'
        One of ``REDUCTIONS``.
    label_smoothing : float, default 0.0
        Amount of label smoothing in ``[0, 1)``.
    """

    name : str = "cross_entropy"

    class_weight    : Optional[List[float]] = None
    ignore_index    : int = -100
    reduction       : str = "mean"
    label_smoothing : float = 0.0


    def __post_init__(self) -> None :
        """
        Validate the cross-entropy parameters and freeze the loss name.
        """

        # Freeze the discriminator so the config always serializes as a cross-entropy config.
        self.name = "cross_entropy"

        # Class weights (must be a list/tuple of numbers when provided)
        if self.class_weight is not None :
            if not isinstance(self.class_weight, (list, tuple)) :
                raise ValueError("class_weight must be a list of floats or None, got {}".format(type(self.class_weight)))
            self.class_weight = [float(single_weight) for single_weight in self.class_weight]

        # Reduction mode
        self.reduction = str(self.reduction).lower()
        if self.reduction not in REDUCTIONS :
            raise ValueError("reduction must be one of {}, got {}".format(REDUCTIONS, repr(self.reduction)))

        # Label smoothing
        if not 0.0 <= self.label_smoothing < 1.0 :
            raise ValueError("label_smoothing must be in [0, 1), got {}".format(repr(self.label_smoothing)))
