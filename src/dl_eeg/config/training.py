"""
Training configuration.

:class:`~dl_eeg.config.training.training_config` holds the (currently minimal) set of hyper-parameters for a training run.
It is expected to grow over time (schedulers, early stopping, logging, ...); for now it covers the basics: number of epochs, batch size, optimizer and learning rate.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from dataclasses import dataclass
from typing import Optional

# Internal imports
from dl_eeg.config.generic import generic_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module constants

# Optimizers recognized by the training config.
OPTIMIZERS = ("adam", "adamw", "sgd", "rmsprop")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Training config class

@dataclass
class training_config(generic_config) :
    """
    Basic training hyper-parameters.

    Parameters
    ----------
    epochs : int, default 100
        Number of training epochs. Must be positive.
    batch_size : int, default 32
        Mini-batch size. Must be positive.
    learning_rate : float, default 1e-3
        Optimizer learning rate. Must be positive.
    optimizer : str, default 'adam'
        Optimizer name (one of ``OPTIMIZERS``).
    weight_decay : float, default 0.0
        L2 weight-decay coefficient. Must be non-negative.
    seed : int or None, default None
        Optional random seed for reproducibility. If given, must be non-negative.
    """

    epochs        : int = 100
    batch_size    : int = 32
    learning_rate : float = 1e-3
    optimizer     : str = "adam"
    weight_decay  : float = 0.0
    seed          : Optional[int] = None


    def __post_init__(self) -> None :
        """
        Validate the training parameters after initialization.
        """

        # Number of epochs (bool is rejected, being a subclass of int).
        if not isinstance(self.epochs, int) or isinstance(self.epochs, bool) or self.epochs <= 0 :
            raise ValueError("epochs must be a positive int, got {}".format(repr(self.epochs)))

        # Batch size
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size <= 0 :
            raise ValueError("batch_size must be a positive int, got {}".format(repr(self.batch_size)))

        # Learning rate
        if self.learning_rate <= 0 :
            raise ValueError("learning_rate must be positive, got {}".format(repr(self.learning_rate)))

        # Optimizer
        self.optimizer = str(self.optimizer).lower()
        if self.optimizer not in OPTIMIZERS :
            raise ValueError("optimizer must be one of {}, got {}".format(OPTIMIZERS, repr(self.optimizer)))

        # Weight decay
        if self.weight_decay < 0 :
            raise ValueError("weight_decay must be non-negative, got {}".format(repr(self.weight_decay)))

        # Seed
        if self.seed is not None and self.seed < 0 :
            raise ValueError("seed must be non-negative, got {}".format(repr(self.seed)))
