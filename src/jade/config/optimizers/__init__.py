"""
Optimizer configuration subpackage.

One file per optimizer, each exposing a :func:`dataclasses.dataclass` config that mirrors the parameters of the matching :mod:`torch.optim` class:

* :class:`~jade.config.optimizers.adam.adam_config`       - :class:`torch.optim.Adam`
* :class:`~jade.config.optimizers.adamw.adamw_config`     - :class:`torch.optim.AdamW`
* :class:`~jade.config.optimizers.sgd.sgd_config`         - :class:`torch.optim.SGD`
* :class:`~jade.config.optimizers.rmsprop.rmsprop_config` - :class:`torch.optim.RMSprop`
* :class:`~jade.config.optimizers.lbfgs.lbfgs_config`     - :class:`torch.optim.LBFGS`

The :data:`OPTIMIZER_CONFIG_REGISTRY` maps the lowercase optimizer name to its config class, and :func:`get_optimizer_config` promotes a plain dict (e.g. coming from a TOML/JSON file) into the proper config instance.
This mirrors the ``MODEL_REGISTRY`` / :func:`~jade.model.get_model` convention used elsewhere in the package.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Internal imports
from jade.config.optimizers.generic import optimizer_config
from jade.config.optimizers.adam import adam_config
from jade.config.optimizers.adamw import adamw_config
from jade.config.optimizers.sgd import sgd_config
from jade.config.optimizers.rmsprop import rmsprop_config
from jade.config.optimizers.lbfgs import lbfgs_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Registry and factory

# Map the lowercase optimizer name to its config class.
OPTIMIZER_CONFIG_REGISTRY = {
    "adam"    : adam_config,
    "adamw"   : adamw_config,
    "sgd"     : sgd_config,
    "rmsprop" : rmsprop_config,
    "lbfgs"   : lbfgs_config,
}


def get_optimizer_config(config) -> optimizer_config :
    """
    Return an optimizer config instance from a dict or an already-built config.

    Parameters
    ----------
    config : optimizer_config or dict
        Either an already-built optimizer config (returned untouched) or a dict with at least the key ``name`` (promoted to the proper subclass through the registry).

    Returns
    -------
    optimizer_config
        The optimizer configuration instance.
    """

    # Already a config: nothing to do.
    if isinstance(config, optimizer_config) :
        return config

    # A dict: promote it through the polymorphic from_dict of the base class.
    if isinstance(config, dict) :
        return optimizer_config.from_dict(config)

    raise TypeError("get_optimizer_config expects an optimizer_config or a dict, got {}".format(type(config)))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Public API

__all__ = [
    "optimizer_config",
    "adam_config",
    "adamw_config",
    "sgd_config",
    "rmsprop_config",
    "lbfgs_config",
    "OPTIMIZER_CONFIG_REGISTRY",
    "get_optimizer_config",
]
