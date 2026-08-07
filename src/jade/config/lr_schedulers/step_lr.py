"""
Step learning-rate scheduler configuration.

:class:`~jade.config.lr_schedulers.step_lr.step_lr_config` mirrors the parameters of :class:`torch.optim.lr_scheduler.StepLR`.
Every ``step_size`` steps the learning rate is multiplied by ``gamma``.
The scheduler is built from this config by :func:`~jade.training.support_training.get_lr_scheduler`.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from dataclasses import dataclass

# Internal imports
from jade.config.lr_schedulers.generic import lr_scheduler_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# StepLR config class

@dataclass
class step_lr_config(lr_scheduler_config) :
    """
    Configuration for :class:`torch.optim.lr_scheduler.StepLR`.

    Parameters
    ----------
    step_size : int, default 10
        Period of learning-rate decay (in scheduler steps). Must be positive.
    gamma : float, default 0.1
        Multiplicative factor of the learning-rate decay. Must be in ``(0, 1]``.
    """

    name : str = "step_lr"

    step_size : int = 10
    gamma     : float = 0.1


    def __post_init__(self) -> None :
        """
        Validate the StepLR parameters and freeze the scheduler name.
        """

        # Freeze the discriminator so the config always serializes as a StepLR config.
        self.name = "step_lr"

        # Step size (bool is rejected, being a subclass of int).
        if not isinstance(self.step_size, int) or isinstance(self.step_size, bool) or self.step_size <= 0 :
            raise ValueError("step_size must be a positive int, got {}".format(repr(self.step_size)))

        # Decay factor
        if not 0.0 < self.gamma <= 1.0 :
            raise ValueError("gamma must be in (0, 1], got {}".format(repr(self.gamma)))
