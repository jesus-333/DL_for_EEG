"""
SGD optimizer configuration.

:class:`~jade.config.optimizers.sgd.sgd_config` mirrors the parameters of :class:`torch.optim.SGD`.
Every parameter defaults to torch's value; the optimizer is built from this config by :func:`~jade.training.support_training.get_optimizer`.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from dataclasses import dataclass

# Internal imports
from jade.config.optimizers.generic import optimizer_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# SGD config class

@dataclass
class sgd_config(optimizer_config) :
    """
    Configuration for :class:`torch.optim.SGD`.

    Parameters
    ----------
    lr : float, default 1e-3
        Learning rate. Must be positive.
    momentum : float, default 0.0
        Momentum factor. Must be non-negative.
    dampening : float, default 0.0
        Dampening for momentum.
    weight_decay : float, default 0.0
        Weight decay (L2 penalty). Must be non-negative.
    nesterov : bool, default False
        Enable Nesterov momentum. Requires a positive momentum and zero dampening.
    maximize : bool, default False
        Maximize the objective with respect to the params, instead of minimizing.
    """

    name : str = "sgd"

    lr           : float = 1e-3
    momentum     : float = 0.0
    dampening    : float = 0.0
    weight_decay : float = 0.0
    nesterov     : bool = False
    maximize     : bool = False


    def __post_init__(self) -> None :
        """
        Validate the SGD parameters and freeze the optimizer name.
        """

        # Freeze the discriminator so the config always serializes as an SGD config.
        self.name = "sgd"

        # Learning rate
        if self.lr <= 0 :
            raise ValueError("lr must be positive, got {}".format(repr(self.lr)))

        # Momentum and weight decay
        if self.momentum < 0 :
            raise ValueError("momentum must be non-negative, got {}".format(repr(self.momentum)))
        if self.weight_decay < 0 :
            raise ValueError("weight_decay must be non-negative, got {}".format(repr(self.weight_decay)))

        # Nesterov momentum requires a positive momentum and zero dampening (torch raises otherwise).
        if self.nesterov and (self.momentum <= 0 or self.dampening != 0) :
            raise ValueError("nesterov momentum requires a positive momentum and zero dampening, got momentum = {}, dampening = {}".format(repr(self.momentum), repr(self.dampening)))
