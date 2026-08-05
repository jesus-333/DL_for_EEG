"""
AdamW optimizer configuration.

:class:`~dl_eeg.config.optimizers.adamw.adamw_config` mirrors the parameters of :class:`torch.optim.AdamW`.
It is almost identical to :class:`~dl_eeg.config.optimizers.adam.adam_config`, the only meaningful difference being the default ``weight_decay`` (``1e-2`` for AdamW, as in torch) and the fact that the weight decay is always decoupled.
The optimizer is built from this config by :func:`~dl_eeg.training.support_training.get_optimizer`.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from dataclasses import dataclass
from typing import Tuple

# Internal imports
from dl_eeg.config.optimizers.generic import optimizer_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# AdamW config class

@dataclass
class adamw_config(optimizer_config) :
    """
    Configuration for :class:`torch.optim.AdamW`.

    Parameters
    ----------
    lr : float, default 1e-3
        Learning rate. Must be positive.
    betas : tuple of float, default (0.9, 0.999)
        Coefficients used for computing running averages of gradient and its square.
    eps : float, default 1e-8
        Term added to the denominator to improve numerical stability. Must be positive.
    weight_decay : float, default 1e-2
        Weight decay coefficient (decoupled). Must be non-negative.
    amsgrad : bool, default False
        Whether to use the AMSGrad variant.
    maximize : bool, default False
        Maximize the objective with respect to the params, instead of minimizing.
    """

    name : str = "adamw"

    lr           : float = 1e-3
    betas        : Tuple[float, float] = (0.9, 0.999)
    eps          : float = 1e-8
    weight_decay : float = 1e-2
    amsgrad      : bool = False
    maximize     : bool = False


    def __post_init__(self) -> None :
        """
        Validate the AdamW parameters and freeze the optimizer name.
        """

        # Freeze the discriminator so the config always serializes as an AdamW config.
        self.name = "adamw"

        # Learning rate
        if self.lr <= 0 :
            raise ValueError("lr must be positive, got {}".format(repr(self.lr)))

        # Betas (normalize a list coming from a file back into a tuple)
        self.betas = tuple(self.betas)
        if len(self.betas) != 2 :
            raise ValueError("betas must be a pair (beta_1, beta_2), got {}".format(repr(self.betas)))
        for beta in self.betas :
            if not 0.0 <= beta < 1.0 :
                raise ValueError("every beta must be in [0, 1), got {}".format(repr(self.betas)))

        # Epsilon
        if self.eps <= 0 :
            raise ValueError("eps must be positive, got {}".format(repr(self.eps)))

        # Weight decay
        if self.weight_decay < 0 :
            raise ValueError("weight_decay must be non-negative, got {}".format(repr(self.weight_decay)))
