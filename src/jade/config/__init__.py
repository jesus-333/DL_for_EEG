"""
Configuration subpackage.

Dataclass-based configuration objects for the package, one file (or subpackage) per concern:

* :class:`~jade.config.generic.generic_config` - base class with the ``to_dict`` / ``from_dict`` / ``to_toml`` / ``from_toml`` / ``to_json`` / ``from_json`` (de)serialization helpers that every config inherits.
* :class:`~jade.config.filter.filter_config` - moabb/MNE filtering parameters.
* :class:`~jade.config.dataset.dataset_config` - dataset download + preprocessing (holds a nested :class:`~jade.config.filter.filter_config`).
* :class:`~jade.config.dataloader.dataloader_config` - :class:`torch.utils.data.DataLoader` parameters.
* :class:`~jade.config.wandb_training.wandb_training_config` - wandb logging parameters.
* :class:`~jade.config.training.training_config` - training hyper-parameters (holds the nested optimizer/scheduler/loss/dataloader/wandb configs).

The optimizer, learning-rate scheduler and loss configurations live in dedicated subpackages, each with one file per implementation plus a name -> class registry and a ``get_*_config`` factory:

* :mod:`jade.config.optimizers` (Adam, AdamW, SGD, RMSprop, LBFGS)
* :mod:`jade.config.lr_schedulers` (ExponentialLR, StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, CyclicLR)
* :mod:`jade.config.loss_functions` (CrossEntropy, NLL)

Model configuration lives in the :mod:`jade.config.model` subpackage, which follows the same registry + ``get_*_config`` pattern (one ``{model_name}_cfg.py`` file per model, plus a single file for the reusable 1D pooling configs).
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Internal imports
from jade.config.generic import generic_config
from jade.config.filter import filter_config
from jade.config.dataset import dataset_config
from jade.config.dataloader import dataloader_config
from jade.config.wandb_training import wandb_training_config
from jade.config.training import training_config

# Optimizer / scheduler / loss subpackages (base classes, concrete configs, registries and factories).
from jade.config.optimizers import (
    optimizer_config,
    adam_config,
    adamw_config,
    sgd_config,
    rmsprop_config,
    lbfgs_config,
    OPTIMIZER_CONFIG_REGISTRY,
    get_optimizer_config,
)
from jade.config.lr_schedulers import (
    lr_scheduler_config,
    exponential_lr_config,
    step_lr_config,
    cosine_annealing_lr_config,
    cosine_annealing_warm_restarts_config,
    cyclic_lr_config,
    LR_SCHEDULER_CONFIG_REGISTRY,
    get_lr_scheduler_config,
)
from jade.config.loss_functions import (
    loss_config,
    cross_entropy_config,
    nll_config,
    elbo_config,
    LOSS_CONFIG_REGISTRY,
    get_loss_config,
)

# Model subpackage (base class, concrete model configs, the 1D pooling config family, registries and factories).
from jade.config.model import (
    model_config,
    EEGNet_config,
    eegnet_pooling_config,
    EEGConformer_config,
    conformer_pooling_config,
    ShallowFBCSPNet_config,
    shallowfbcspnet_pooling_config,
    Deep4Net_config,
    deep4net_pooling_config,
    TIDNet_config,
    tidnet_pooling_config,
    braindecode_wrapper_config,
    MODEL_CONFIG_REGISTRY,
    get_model_config,
    pooling_config,
    avg_pooling_config,
    blur_pooling_config,
    sinc_pooling_config,
    dwt_pooling_config,
    max_pooling_config,
    POOLING_CONFIG_REGISTRY,
    get_pooling_config,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Public API

__all__ = [
    # Base and dataset-related configs
    "generic_config",
    "filter_config",
    "dataset_config",
    "dataloader_config",
    "wandb_training_config",
    "training_config",
    # Optimizer configs
    "optimizer_config",
    "adam_config",
    "adamw_config",
    "sgd_config",
    "rmsprop_config",
    "lbfgs_config",
    "OPTIMIZER_CONFIG_REGISTRY",
    "get_optimizer_config",
    # Learning-rate scheduler configs
    "lr_scheduler_config",
    "exponential_lr_config",
    "step_lr_config",
    "cosine_annealing_lr_config",
    "cosine_annealing_warm_restarts_config",
    "cyclic_lr_config",
    "LR_SCHEDULER_CONFIG_REGISTRY",
    "get_lr_scheduler_config",
    # Loss-function configs
    "loss_config",
    "cross_entropy_config",
    "nll_config",
    "elbo_config",
    "LOSS_CONFIG_REGISTRY",
    "get_loss_config",
    # Model configs
    "model_config",
    "EEGNet_config",
    "eegnet_pooling_config",
    "EEGConformer_config",
    "conformer_pooling_config",
    "ShallowFBCSPNet_config",
    "shallowfbcspnet_pooling_config",
    "Deep4Net_config",
    "deep4net_pooling_config",
    "TIDNet_config",
    "tidnet_pooling_config",
    "braindecode_wrapper_config",
    "MODEL_CONFIG_REGISTRY",
    "get_model_config",
    # Pooling configs
    "pooling_config",
    "avg_pooling_config",
    "blur_pooling_config",
    "sinc_pooling_config",
    "dwt_pooling_config",
    "max_pooling_config",
    "POOLING_CONFIG_REGISTRY",
    "get_pooling_config",
]
