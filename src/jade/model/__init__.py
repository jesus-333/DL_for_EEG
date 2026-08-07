"""
Model subpackage: EEG deep-learning models and a name -> class registry.

This module hosts a small factory (``get_model`` / ``MODEL_REGISTRY``) that builds a model from a config dict, mirroring the ``get_pooling_layer`` / ``POOLING_REGISTRY`` pattern in :mod:`jade.model.pooling_1d` but at the top level.

The 1D pooling layers are intentionally NOT part of this registry: they are building blocks living in :mod:`jade.model.pooling_1d` with their own ``get_pooling_layer`` factory, and are not full models.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Third party imports
from torch import nn

# Internal imports
from jade.model.EEGNet import EEGNet, EEGNet_Classifier

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Public API

__all__ = [
    "EEGNet",
    "EEGNet_Classifier",
    "MODEL_REGISTRY",
    "get_model",
]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Registry

# Maps lowercase string aliases -> model class.
# Every model class takes a single ``config : dict`` argument, so the factory can build any of them uniformly.
MODEL_REGISTRY = {
    "eegnet"            : EEGNet,
    "eegnet_classifier" : EEGNet_Classifier,
    "eegnetclassifier"  : EEGNet_Classifier,
}


def get_model(config : dict) -> nn.Module :
    """
    Build a model from a config dict - the single entry point for model creation.

    The dict must contain a ``model_name`` key naming the model, plus whatever that model's constructor requires (every model here takes the whole ``config`` dict).

    Parameters
    ----------
    config : dict
        Must contain ``model_name`` (one of ``MODEL_REGISTRY``, case-insensitive), plus the model-specific keys documented on the corresponding class.

    Returns
    -------
    torch.nn.Module
        An instance of the requested model.

    Raises
    ------
    KeyError
        If ``config`` does not contain a ``model_name`` key.
    ValueError
        If ``model_name`` is not a known model.

    Examples
    --------
    >>> get_model({'model_name' : 'eegnet', 'use_bias' : False, 'D' : 2})
    >>> get_model({'model_name' : 'eegnet_classifier', 'n_classes' : 4})
    """

    # Input check
    if "model_name" not in config :
        raise KeyError("config must contain 'model_name'; one of {}".format(sorted(set(MODEL_REGISTRY))))
    name = str(config["model_name"]).lower()

    if name not in MODEL_REGISTRY :
        raise ValueError("unknown model_name {}; expected one of {}".format(repr(name), sorted(set(MODEL_REGISTRY))))

    # Copy so the caller's dict is never mutated by a model constructor.
    return MODEL_REGISTRY[name](dict(config))
