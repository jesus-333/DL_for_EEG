"""
Model configuration subpackage.

Configuration objects that describe a :mod:`jade.model` network, following the ``{model_name}_cfg.py`` naming convention (the ``cfg`` suffix keeps the config files apart from the model implementation files):

* :class:`~jade.config.model.EEGNet_cfg.EEGNet_config` - :class:`~jade.model.EEGNet.EEGNet` / :class:`~jade.model.EEGNet.EEGNet_Classifier`.
* :class:`~jade.config.model.EEGConformer_cfg.EEGConformer_config` - :class:`~jade.model.EEGConformer.EEGConformer` / :class:`~jade.model.EEGConformer.EEGConformer_Classifier`.
* :class:`~jade.config.model.ShallowFBCSPNet_cfg.ShallowFBCSPNet_config` - :class:`~jade.model.ShallowFBCSPNet.ShallowFBCSPNet` / :class:`~jade.model.ShallowFBCSPNet.ShallowFBCSPNet_Classifier`.
* :class:`~jade.config.model.Deep4Net_cfg.Deep4Net_config` - :class:`~jade.model.Deep4Net.Deep4Net` / :class:`~jade.model.Deep4Net.Deep4Net_Classifier`.
* :class:`~jade.config.model.TIDNet_cfg.TIDNet_config` - :class:`~jade.model.TIDNet.TIDNet` / :class:`~jade.model.TIDNet.TIDNet_Classifier`.
* :class:`~jade.config.model.vEEGNet_cfg.vEEGNet_config` - :class:`~jade.model.vEEGNet.vEEGNet` (a variational autoencoder, so it has no classifier wrapper).
* :class:`~jade.config.model.hvEEGNet_cfg.hvEEGNet_config` - :class:`~jade.model.hvEEGNet.hvEEGNet` (the hierarchical variational autoencoder, no classifier wrapper either).
* :class:`~jade.config.model.braindecode_pooling_extension_cfg.braindecode_wrapper_config` - :class:`~jade.model.braindecode_pooling_extension.braindecode_pooling_wrapper` (any ``braindecode`` model, with its pooling layers swapped).

Every model config carries its ``name`` inside; that value is what :func:`~jade.model.get_model` uses to pick the model class.
The :data:`MODEL_CONFIG_REGISTRY` maps the lowercase model name to its config class, and :func:`get_model_config` promotes a plain dict (e.g. coming from a TOML/JSON file) into the proper config instance. This mirrors the optimizer/loss config families and the ``MODEL_REGISTRY`` / :func:`~jade.model.get_model` convention.

The 1D pooling configuration family lives in a single file, :mod:`jade.config.model.pooling_1d_cfg`, because pooling is meant to be reused several times inside a model and its operators share most parameters. Its registry (:data:`POOLING_CONFIG_REGISTRY`) and factory (:func:`get_pooling_config`) are re-exported here for convenience.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Internal imports
from jade.config.model.generic import model_config
from jade.config.model.EEGNet_cfg import EEGNet_config, eegnet_pooling_config
from jade.config.model.EEGConformer_cfg import EEGConformer_config, conformer_pooling_config
from jade.config.model.ShallowFBCSPNet_cfg import ShallowFBCSPNet_config, shallowfbcspnet_pooling_config
from jade.config.model.Deep4Net_cfg import Deep4Net_config, deep4net_pooling_config
from jade.config.model.TIDNet_cfg import TIDNet_config, tidnet_pooling_config
from jade.config.model.vEEGNet_cfg import vEEGNet_config
from jade.config.model.hvEEGNet_cfg import hvEEGNet_config
from jade.config.model.braindecode_pooling_extension_cfg import braindecode_wrapper_config
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
    "eegnet"                     : EEGNet_config,
    "eegnet_classifier"          : EEGNet_config,
    "eegnetclassifier"           : EEGNet_config,
    "eegconformer"               : EEGConformer_config,
    "eegconformer_classifier"    : EEGConformer_config,
    "eegconformerclassifier"     : EEGConformer_config,
    "shallowfbcspnet"            : ShallowFBCSPNet_config,
    "shallowfbcspnet_classifier" : ShallowFBCSPNet_config,
    "shallowfbcspnetclassifier"  : ShallowFBCSPNet_config,
    "deep4net"                   : Deep4Net_config,
    "deep4net_classifier"        : Deep4Net_config,
    "deep4netclassifier"         : Deep4Net_config,
    "tidnet"                     : TIDNet_config,
    "tidnet_classifier"          : TIDNet_config,
    "tidnetclassifier"           : TIDNet_config,
    # vEEGNet is a variational autoencoder: the classification branch of the original paper is not implemented, so it has no *_classifier alias.
    # Those two names are deliberately left unknown (instead of pointing here) so they stay available for the classifier variant, and so asking for one today fails with a clear error instead of silently building a generative model.
    "veegnet"                    : vEEGNet_config,
    "vae_eegnet"                 : vEEGNet_config,
    "hveegnet"                   : hvEEGNet_config,
    "hierarchical_veegnet"       : hvEEGNet_config,
    "braindecode_wrapper"        : braindecode_wrapper_config,
    "braindecode"                : braindecode_wrapper_config,
    "braindecode_pooling"        : braindecode_wrapper_config,
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
    "eegnet_pooling_config",
    "EEGConformer_config",
    "conformer_pooling_config",
    "ShallowFBCSPNet_config",
    "shallowfbcspnet_pooling_config",
    "Deep4Net_config",
    "deep4net_pooling_config",
    "TIDNet_config",
    "tidnet_pooling_config",
    "vEEGNet_config",
    "hvEEGNet_config",
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
