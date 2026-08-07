"""
Exponential-decay learning-rate scheduler configuration.

:class:`~dl_eeg.config.lr_schedulers.exponential_lr.exponential_lr_config` mirrors the parameters of :class:`torch.optim.lr_scheduler.ExponentialLR`.
At every ``step`` the learning rate is multiplied by ``gamma`` (i.e. ``lr = lr * gamma``), giving a geometric/exponential decay.
The scheduler is built from this config by :func:`~dl_eeg.training.support_training.get_lr_scheduler`.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from dataclasses import dataclass

# Internal imports
from dl_eeg.config.lr_schedulers.generic import lr_scheduler_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# ExponentialLR config class

@dataclass
class exponential_lr_config(lr_scheduler_config) :
    """
    Configuration for :class:`torch.optim.lr_scheduler.ExponentialLR`.

    Parameters
    ----------
    gamma : float, default 0.99
        Multiplicative factor of the learning-rate decay, applied at every step.
        Must be in ``(0, 1]`` (``1`` means no decay).
    """

    name : str = "exponential_lr"

    gamma : float = 0.99


    def __post_init__(self) -> None :
        """
        Validate the ExponentialLR parameters and freeze the scheduler name.
        """

        # Freeze the discriminator so the config always serializes as an ExponentialLR config.
        self.name = "exponential_lr"

        # Decay factor
        if not 0.0 < self.gamma <= 1.0 :
            raise ValueError("gamma must be in (0, 1], got {}".format(repr(self.gamma)))
