"""
Cosine-annealing-with-warm-restarts learning-rate scheduler configuration.

:class:`~dl_eeg.config.lr_schedulers.cosine_annealing_warm_restarts.cosine_annealing_warm_restarts_config` mirrors the parameters of :class:`torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`.
The learning rate follows a cosine curve that restarts every ``T_0`` steps, with each restart period multiplied by ``T_mult``.
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
# CosineAnnealingWarmRestarts config class

@dataclass
class cosine_annealing_warm_restarts_config(lr_scheduler_config) :
    """
    Configuration for :class:`torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`.

    Parameters
    ----------
    T_0 : int, default 10
        Number of iterations before the first restart. Must be positive.
    T_mult : int, default 1
        Factor by which the restart period is multiplied after each restart. Must be an int >= 1.
    eta_min : float, default 0.0
        Minimum learning rate. Must be non-negative.
    """

    name : str = "cosine_annealing_warm_restarts"

    T_0     : int = 10
    T_mult  : int = 1
    eta_min : float = 0.0


    def __post_init__(self) -> None :
        """
        Validate the CosineAnnealingWarmRestarts parameters and freeze the scheduler name.
        """

        # Freeze the discriminator so the config always serializes as a CosineAnnealingWarmRestarts config.
        self.name = "cosine_annealing_warm_restarts"

        # First restart period (bool is rejected, being a subclass of int).
        if not isinstance(self.T_0, int) or isinstance(self.T_0, bool) or self.T_0 <= 0 :
            raise ValueError("T_0 must be a positive int, got {}".format(repr(self.T_0)))

        # Restart period multiplier
        if not isinstance(self.T_mult, int) or isinstance(self.T_mult, bool) or self.T_mult < 1 :
            raise ValueError("T_mult must be an int >= 1, got {}".format(repr(self.T_mult)))

        # Minimum learning rate
        if self.eta_min < 0 :
            raise ValueError("eta_min must be non-negative, got {}".format(repr(self.eta_min)))
