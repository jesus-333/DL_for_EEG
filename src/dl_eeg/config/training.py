"""
Training configuration.
=======================

``TrainingConfig`` holds the (currently minimal) set of hyper-parameters for a training
run. It is expected to grow over time (schedulers, early stopping, logging, ...); for
now it covers the basics: number of epochs, batch size, optimizer and learning rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from dl_eeg.config.generic import GenericConfig

#: Optimizers recognized by the training config.
OPTIMIZERS = ("adam", "adamw", "sgd", "rmsprop")


@dataclass
class TrainingConfig(GenericConfig):
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

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 0.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.epochs, int) or isinstance(self.epochs, bool) or self.epochs <= 0:
            raise ValueError(f"epochs must be a positive int, got {self.epochs!r}")
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size <= 0:
            raise ValueError(f"batch_size must be a positive int, got {self.batch_size!r}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate!r}")
        if str(self.optimizer).lower() not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}")
        self.optimizer = str(self.optimizer).lower()
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay!r}")
        if self.seed is not None and self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed!r}")
