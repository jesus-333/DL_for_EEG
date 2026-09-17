"""
Training subpackage.

This package contains everything related to the training of a model (the federated-learning part, if any, will live in a dedicated submodule).
Two objectives are implemented, each with its own training cycle and its own single-epoch functions: the **classification** (finetuning) one and the **generative** (variational autoencoder) one.

Modules:

* :mod:`jade.training.classifier` - the full training cycle of a classifier: :func:`~jade.training.classifier.train` (builds the model from a model config and delegates), :func:`~jade.training.classifier.train_model` (trains an already-built model) and their wandb counterparts :func:`~jade.training.classifier.wandb_train` (builds the model) and :func:`~jade.training.classifier.wandb_train_model` (already-built model).
* :mod:`jade.training.classifier_epoch` - the single-epoch functions of a classifier (:func:`~jade.training.classifier_epoch.train_epoch_function`, :func:`~jade.training.classifier_epoch.validation_epoch_function`).
* :mod:`jade.training.vae` - the same four entry points for a generative model (e.g. :class:`~jade.model.vEEGNet.vEEGNet`), with the same signatures.
* :mod:`jade.training.vae_epoch` - the single-epoch functions of a generative model.
* :mod:`jade.training.custom_loss_function` - the losses with no :mod:`torch.nn` counterpart, i.e. the ELBO.
* :mod:`jade.training.wandb_utils` - everything about a wandb run, shared by the two training cycles.
* :mod:`jade.training.test_functions` - the generic evaluation function :func:`~jade.training.test_functions.test`.
* :mod:`jade.training.support_training` - shared helpers and the config -> torch object factories.
* :mod:`jade.training.metrics` - the (torchmetrics-based) classification metrics and the reconstruction metrics.
* :mod:`jade.training.logging_utils` - logging helpers, in particular :func:`~jade.training.logging_utils.setup_logging` (an opt-in convenience for people who do not want to set up their own logger).

Note on the names exported here. The classification entry points keep the plain names they always had (``train``, ``train_model``, ...), and the generative ones are exported with a ``vae_`` prefix, so nothing that already imported the former changes meaning.
For new code the preferred (and unambiguous) style is to import the module and call through it, e.g. ``from jade.training import vae`` followed by ``vae.train_model(...)``.

Following the library logging best practice, a :class:`logging.NullHandler` is attached to the package logger so that importing the package never forces any logging configuration on the user; each module logs through ``logging.getLogger(__name__)``.
Users who do not want to configure logging themselves can call :func:`~jade.training.logging_utils.setup_logging` to wire up a sensible root configuration in one call.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Full module imports
import logging

# Internal imports (leaf modules first, so the modules that depend on them find them already loaded)
from jade.training.logging_utils import setup_logging
from jade.training import wandb_utils
from jade.training import custom_loss_function
from jade.training import support_training
from jade.training import metrics
from jade.training import classifier_epoch
from jade.training import vae_epoch
from jade.training import classifier
from jade.training import vae

# Classification training (the plain names are kept for backward compatibility).
from jade.training.classifier_epoch import train_epoch_function, validation_epoch_function
from jade.training.classifier import train, train_model, wandb_train, wandb_train_model

# Generative (VAE) training. The names are explicitly prefixed, so the ones above keep their meaning.
from jade.training.vae_epoch import (
    train_epoch_function      as vae_train_epoch_function,
    validation_epoch_function as vae_validation_epoch_function,
)
from jade.training.vae import (
    train             as vae_train,
    train_model       as vae_train_model,
    wandb_train       as vae_wandb_train,
    wandb_train_model as vae_wandb_train_model,
)

from jade.training.test_functions import test

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Package logger

# Attach a NullHandler so the library never emits "No handlers could be found" warnings and never configures logging on its own.
logging.getLogger(__name__).addHandler(logging.NullHandler())

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Public API

__all__ = [
    # Classification training
    "train",
    "train_model",
    "wandb_train",
    "wandb_train_model",
    "train_epoch_function",
    "validation_epoch_function",
    # Generative (VAE) training
    "vae_train",
    "vae_train_model",
    "vae_wandb_train",
    "vae_wandb_train_model",
    "vae_train_epoch_function",
    "vae_validation_epoch_function",
    # Shared
    "test",
    "setup_logging",
    # Modules
    "classifier",
    "classifier_epoch",
    "vae",
    "vae_epoch",
    "custom_loss_function",
    "wandb_utils",
    "support_training",
    "metrics",
]
