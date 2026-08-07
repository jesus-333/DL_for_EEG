"""
Training subpackage.

This package contains everything related to the training of a model (the federated-learning part, if any, will live in a dedicated submodule).
For now only the classification (finetuning) objective is implemented; a reconstruction/pretraining objective may be added later.

Modules:

* :mod:`jade.training.classifier` - the full training cycle (:func:`~jade.training.classifier.train` and the wandb wrapper :func:`~jade.training.classifier.wandb_train`).
* :mod:`jade.training.classifier_epoch` - the single-epoch functions (:func:`~jade.training.classifier_epoch.train_epoch_function`, :func:`~jade.training.classifier_epoch.validation_epoch_function`).
* :mod:`jade.training.test_functions` - the generic evaluation function :func:`~jade.training.test_functions.test`.
* :mod:`jade.training.support_training` - shared helpers and the config -> torch object factories.
* :mod:`jade.training.metrics` - the (torchmetrics-based) classification metrics.

Following the library logging best practice, a :class:`logging.NullHandler` is attached to the package logger so that importing the package never forces any logging configuration on the user; each module logs through ``logging.getLogger(__name__)``.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Full module imports
import logging

# Internal imports (leaf modules first, so the modules that depend on them find them already loaded)
from jade.training import support_training
from jade.training import metrics
from jade.training.classifier_epoch import train_epoch_function, validation_epoch_function
from jade.training.classifier import train, wandb_train
from jade.training.test_functions import test

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Package logger

# Attach a NullHandler so the library never emits "No handlers could be found" warnings and never configures logging on its own.
logging.getLogger(__name__).addHandler(logging.NullHandler())

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Public API

__all__ = [
    "train",
    "wandb_train",
    "train_epoch_function",
    "validation_epoch_function",
    "test",
    "support_training",
    "metrics",
]
