"""
Cyclic learning-rate scheduler configuration.

:class:`~dl_eeg.config.lr_schedulers.cyclic_lr.cyclic_lr_config` mirrors the parameters of :class:`torch.optim.lr_scheduler.CyclicLR`.
The learning rate cycles between ``base_lr`` and ``max_lr`` following the selected ``mode``.
The scheduler is built from this config by :func:`~dl_eeg.training.support_training.get_lr_scheduler`.

Note that ``cycle_momentum`` (True by default in torch) only works with optimizers that expose a ``momentum`` (e.g. SGD, RMSprop); set it to False when cycling the learning rate of Adam/AdamW.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from dataclasses import dataclass
from typing import Optional

# Internal imports
from dl_eeg.config.lr_schedulers.generic import lr_scheduler_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module constants

# Cycling policies accepted by torch.
CYCLIC_MODES = ("triangular", "triangular2", "exp_range")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# CyclicLR config class

@dataclass
class cyclic_lr_config(lr_scheduler_config) :
    """
    Configuration for :class:`torch.optim.lr_scheduler.CyclicLR`.

    Parameters
    ----------
    base_lr : float, default 1e-4
        Lower learning-rate boundary of the cycle. Must be positive.
    max_lr : float, default 1e-2
        Upper learning-rate boundary of the cycle. Must be greater than ``base_lr``.
    step_size_up : int, default 2000
        Number of training iterations in the increasing half of a cycle. Must be positive.
    step_size_down : int or None, default None
        Number of training iterations in the decreasing half. ``None`` makes it equal to ``step_size_up``.
    mode : str, default 'triangular'
        One of ``CYCLIC_MODES``.
    gamma : float, default 1.0
        Constant used in the ``'exp_range'`` scaling function. Must be positive.
    cycle_momentum : bool, default True
        Whether to cycle the momentum inversely to the learning rate. Only valid for optimizers exposing a momentum.
    """

    name : str = "cyclic_lr"

    base_lr        : float = 1e-4
    max_lr         : float = 1e-2
    step_size_up   : int = 2000
    step_size_down : Optional[int] = None
    mode           : str = "triangular"
    gamma          : float = 1.0
    cycle_momentum : bool = True


    def __post_init__(self) -> None :
        """
        Validate the CyclicLR parameters and freeze the scheduler name.
        """

        # Freeze the discriminator so the config always serializes as a CyclicLR config.
        self.name = "cyclic_lr"

        # Learning-rate boundaries
        if self.base_lr <= 0 :
            raise ValueError("base_lr must be positive, got {}".format(repr(self.base_lr)))
        if self.max_lr <= self.base_lr :
            raise ValueError("max_lr must be greater than base_lr, got base_lr = {}, max_lr = {}".format(repr(self.base_lr), repr(self.max_lr)))

        # Step sizes (bool is rejected, being a subclass of int).
        if not isinstance(self.step_size_up, int) or isinstance(self.step_size_up, bool) or self.step_size_up <= 0 :
            raise ValueError("step_size_up must be a positive int, got {}".format(repr(self.step_size_up)))
        if self.step_size_down is not None :
            if not isinstance(self.step_size_down, int) or isinstance(self.step_size_down, bool) or self.step_size_down <= 0 :
                raise ValueError("step_size_down must be a positive int or None, got {}".format(repr(self.step_size_down)))

        # Cycling policy
        self.mode = str(self.mode).lower()
        if self.mode not in CYCLIC_MODES :
            raise ValueError("mode must be one of {}, got {}".format(CYCLIC_MODES, repr(self.mode)))

        # Scaling constant
        if self.gamma <= 0 :
            raise ValueError("gamma must be positive, got {}".format(repr(self.gamma)))
