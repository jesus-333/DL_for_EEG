"""
RMSprop optimizer configuration.

:class:`~dl_eeg.config.optimizers.rmsprop.rmsprop_config` mirrors the parameters of :class:`torch.optim.RMSprop`.
Every parameter defaults to torch's value; the optimizer is built from this config by :func:`~dl_eeg.training.support_training.get_optimizer`.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from dataclasses import dataclass

# Internal imports
from dl_eeg.config.optimizers.generic import optimizer_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# RMSprop config class

@dataclass
class rmsprop_config(optimizer_config) :
    """
    Configuration for :class:`torch.optim.RMSprop`.

    Parameters
    ----------
    lr : float, default 1e-2
        Learning rate. Must be positive.
    alpha : float, default 0.99
        Smoothing constant. Must be non-negative.
    eps : float, default 1e-8
        Term added to the denominator to improve numerical stability. Must be positive.
    weight_decay : float, default 0.0
        Weight decay (L2 penalty). Must be non-negative.
    momentum : float, default 0.0
        Momentum factor. Must be non-negative.
    centered : bool, default False
        If True, compute the centered RMSprop (gradient normalized by an estimate of its variance).
    """

    name : str = "rmsprop"

    lr           : float = 1e-2
    alpha        : float = 0.99
    eps          : float = 1e-8
    weight_decay : float = 0.0
    momentum     : float = 0.0
    centered     : bool = False


    def __post_init__(self) -> None :
        """
        Validate the RMSprop parameters and freeze the optimizer name.
        """

        # Freeze the discriminator so the config always serializes as an RMSprop config.
        self.name = "rmsprop"

        # Learning rate
        if self.lr <= 0 :
            raise ValueError("lr must be positive, got {}".format(repr(self.lr)))

        # Alpha, epsilon, weight decay and momentum
        if self.alpha < 0 :
            raise ValueError("alpha must be non-negative, got {}".format(repr(self.alpha)))
        if self.eps <= 0 :
            raise ValueError("eps must be positive, got {}".format(repr(self.eps)))
        if self.weight_decay < 0 :
            raise ValueError("weight_decay must be non-negative, got {}".format(repr(self.weight_decay)))
        if self.momentum < 0 :
            raise ValueError("momentum must be non-negative, got {}".format(repr(self.momentum)))
