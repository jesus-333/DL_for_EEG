"""
Learning-rate scheduler configuration subpackage.

One file per scheduler, each exposing a :func:`dataclasses.dataclass` config that mirrors the parameters of the matching :mod:`torch.optim.lr_scheduler` class:

* :class:`~jade.config.lr_schedulers.exponential_lr.exponential_lr_config`                              - :class:`torch.optim.lr_scheduler.ExponentialLR`
* :class:`~jade.config.lr_schedulers.step_lr.step_lr_config`                                            - :class:`torch.optim.lr_scheduler.StepLR`
* :class:`~jade.config.lr_schedulers.cosine_annealing_lr.cosine_annealing_lr_config`                    - :class:`torch.optim.lr_scheduler.CosineAnnealingLR`
* :class:`~jade.config.lr_schedulers.cosine_annealing_warm_restarts.cosine_annealing_warm_restarts_config` - :class:`torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`
* :class:`~jade.config.lr_schedulers.cyclic_lr.cyclic_lr_config`                                        - :class:`torch.optim.lr_scheduler.CyclicLR`

The :data:`LR_SCHEDULER_CONFIG_REGISTRY` maps the lowercase scheduler name to its config class, and :func:`get_lr_scheduler_config` promotes a plain dict into the proper config instance.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Internal imports
from jade.config.lr_schedulers.generic import lr_scheduler_config
from jade.config.lr_schedulers.exponential_lr import exponential_lr_config
from jade.config.lr_schedulers.step_lr import step_lr_config
from jade.config.lr_schedulers.cosine_annealing_lr import cosine_annealing_lr_config
from jade.config.lr_schedulers.cosine_annealing_warm_restarts import cosine_annealing_warm_restarts_config
from jade.config.lr_schedulers.cyclic_lr import cyclic_lr_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Registry and factory

# Map the lowercase scheduler name to its config class.
LR_SCHEDULER_CONFIG_REGISTRY = {
    "exponential_lr"                 : exponential_lr_config,
    "step_lr"                        : step_lr_config,
    "cosine_annealing_lr"            : cosine_annealing_lr_config,
    "cosine_annealing_warm_restarts" : cosine_annealing_warm_restarts_config,
    "cyclic_lr"                      : cyclic_lr_config,
}


def get_lr_scheduler_config(config) -> lr_scheduler_config :
    """
    Return a scheduler config instance from a dict or an already-built config.

    Parameters
    ----------
    config : lr_scheduler_config or dict
        Either an already-built scheduler config (returned untouched) or a dict with at least the key ``name`` (promoted to the proper subclass through the registry).

    Returns
    -------
    lr_scheduler_config
        The scheduler configuration instance.
    """

    # Already a config: nothing to do.
    if isinstance(config, lr_scheduler_config) :
        return config

    # A dict: promote it through the polymorphic from_dict of the base class.
    if isinstance(config, dict) :
        return lr_scheduler_config.from_dict(config)

    raise TypeError("get_lr_scheduler_config expects an lr_scheduler_config or a dict, got {}".format(type(config)))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Public API

__all__ = [
    "lr_scheduler_config",
    "exponential_lr_config",
    "step_lr_config",
    "cosine_annealing_lr_config",
    "cosine_annealing_warm_restarts_config",
    "cyclic_lr_config",
    "LR_SCHEDULER_CONFIG_REGISTRY",
    "get_lr_scheduler_config",
]
