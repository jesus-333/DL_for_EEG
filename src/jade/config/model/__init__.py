"""
Model configuration subpackage.

Configuration objects that describe a :mod:`jade.model` network, following the ``{model_name}_cfg.py`` naming convention (the ``cfg`` suffix keeps the config files apart from the model implementation files):

* :class:`~jade.config.model.EEGNet_cfg.EEGNet_config` - :class:`~jade.model.EEGNet.EEGNet` / :class:`~jade.model.EEGNet.EEGNet_Classifier`.

Every model config carries its ``name`` inside; that value is what :func:`~jade.model.get_model` uses to pick the model class.
The :data:`MODEL_CONFIG_REGISTRY` maps the lowercase model name to its config class, and :func:`get_model_config` promotes a plain dict (e.g. coming from a TOML/JSON file) into the proper config instance. This mirrors the optimizer/loss config families and the ``MODEL_REGISTRY`` / :func:`~jade.model.get_model` convention.

The 1D pooling configuration family lives in a single file, :mod:`jade.config.model.pooling_1d_cfg`, because pooling is meant to be reused several times inside a model and its operators share most parameters. Its registry (:data:`POOLING_CONFIG_REGISTRY`) and factory (:func:`get_pooling_config`) are re-exported here for convenience.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Internal imports
from jade.config.model.generic import model_config
from jade.config.model.EEGNet_cfg import EEGNet_config
from jade.config.model.pooling_1d_cfg import (
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
# Registry and factory

# Map the lowercase model name to its config class (the aliases mirror jade.model.MODEL_REGISTRY).
MODEL_CONFIG_REGISTRY = {
    "eegnet"            : EEGNet_config,
    "eegnet_classifier" : EEGNet_config,
    "eegnetclassifier"  : EEGNet_config,
}


def get_model_config(config) -> model_config :
    """
    Return a model config instance from a dict or an already-built config.

    Parameters
    ----------
    config : model_config or dict
        Either an already-built model config (returned untouched) or a dict with at least the key ``name`` (promoted to the proper subclass through the registry).

    Returns
    -------
    model_config
        The model configuration instance.
    """

    # Already a config: nothing to do.
    if isinstance(config, model_config) :
        return config

    # A dict: promote it through the polymorphic from_dict of the base class.
    if isinstance(config, dict) :
        return model_config.from_dict(config)

    raise TypeError("get_model_config expects a model_config or a dict, got {}".format(type(config)))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Public API

__all__ = [
    # Model configs
    "model_config",
    "EEGNet_config",
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
