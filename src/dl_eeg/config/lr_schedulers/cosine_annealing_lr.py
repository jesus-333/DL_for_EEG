"""
Cosine-annealing learning-rate scheduler configuration.

:class:`~dl_eeg.config.lr_schedulers.cosine_annealing_lr.cosine_annealing_lr_config` mirrors the parameters of :class:`torch.optim.lr_scheduler.CosineAnnealingLR`.
The learning rate follows a cosine curve from its initial value down to ``eta_min`` over ``T_max`` steps.
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
# CosineAnnealingLR config class

@dataclass
class cosine_annealing_lr_config(lr_scheduler_config) :
    """
    Configuration for :class:`torch.optim.lr_scheduler.CosineAnnealingLR`.

    Parameters
    ----------
    T_max : int, default 10
        Maximum number of iterations (half a cosine period). Must be positive.
    eta_min : float, default 0.0
        Minimum learning rate. Must be non-negative.
    """

    name : str = "cosine_annealing_lr"

    T_max   : int = 10
    eta_min : float = 0.0


    def __post_init__(self) -> None :
        """
        Validate the CosineAnnealingLR parameters and freeze the scheduler name.
        """

        # Freeze the discriminator so the config always serializes as a CosineAnnealingLR config.
        self.name = "cosine_annealing_lr"

        # Period (bool is rejected, being a subclass of int).
        if not isinstance(self.T_max, int) or isinstance(self.T_max, bool) or self.T_max <= 0 :
            raise ValueError("T_max must be a positive int, got {}".format(repr(self.T_max)))

        # Minimum learning rate
        if self.eta_min < 0 :
            raise ValueError("eta_min must be non-negative, got {}".format(repr(self.eta_min)))
