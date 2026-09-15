"""
Model subpackage: EEG deep-learning models and a name -> class registry.

This module hosts a small factory (``get_model`` / ``MODEL_REGISTRY``) that builds a model from a model config, mirroring the ``get_pooling_layer`` / ``POOLING_REGISTRY`` pattern in :mod:`jade.model.pooling_1d` but at the top level.

``get_model`` takes a model config (an instance of a :class:`~jade.config.model.generic.model_config` subclass, e.g. :class:`~jade.config.model.EEGNet_cfg.EEGNet_config`, or a plain dict that is promoted to one) and selects the model class through the config ``name`` field.

The 1D pooling layers are intentionally NOT part of this registry: they are building blocks living in :mod:`jade.model.pooling_1d` with their own ``get_pooling_layer`` factory, and are not full models.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Third party imports
from torch import nn

# Internal imports
from jade.config.model import get_model_config
from jade.model.EEGNet import EEGNet, EEGNet_Classifier
from jade.model.EEGConformer import EEGConformer, EEGConformer_Classifier
from jade.model.ShallowFBCSPNet import ShallowFBCSPNet, ShallowFBCSPNet_Classifier
from jade.model.Deep4Net import Deep4Net, Deep4Net_Classifier
from jade.model.TIDNet import TIDNet, TIDNet_Classifier
from jade.model.braindecode_pooling_extension import braindecode_pooling_wrapper

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Public API

__all__ = [
    "EEGNet",
    "EEGNet_Classifier",
    "EEGConformer",
    "EEGConformer_Classifier",
    "ShallowFBCSPNet",
    "ShallowFBCSPNet_Classifier",
    "Deep4Net",
    "Deep4Net_Classifier",
    "TIDNet",
    "TIDNet_Classifier",
    "braindecode_pooling_wrapper",
    "MODEL_REGISTRY",
    "get_model",
]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Registry

# Maps lowercase string aliases -> model class.
# Every model class takes a single model config argument, so the factory can build any of them uniformly.
MODEL_REGISTRY = {
    "eegnet"                     : EEGNet,
    "eegnet_classifier"          : EEGNet_Classifier,
    "eegnetclassifier"           : EEGNet_Classifier,
    "eegconformer"               : EEGConformer,
    "eegconformer_classifier"    : EEGConformer_Classifier,
    "eegconformerclassifier"     : EEGConformer_Classifier,
    "shallowfbcspnet"            : ShallowFBCSPNet,
    "shallowfbcspnet_classifier" : ShallowFBCSPNet_Classifier,
    "shallowfbcspnetclassifier"  : ShallowFBCSPNet_Classifier,
    "deep4net"                   : Deep4Net,
    "deep4net_classifier"        : Deep4Net_Classifier,
    "deep4netclassifier"         : Deep4Net_Classifier,
    "tidnet"                     : TIDNet,
    "tidnet_classifier"          : TIDNet_Classifier,
    "tidnetclassifier"           : TIDNet_Classifier,
    "braindecode_wrapper"        : braindecode_pooling_wrapper,
    "braindecode"                : braindecode_pooling_wrapper,
    "braindecode_pooling"        : braindecode_pooling_wrapper,
}


def get_model(config) -> nn.Module :
    """
    Build a model from its config - the single entry point for model creation.

    The model class is selected through the config ``name`` field (e.g. ``'eegnet_classifier'``); every model then receives the whole config, which carries all its hyper-parameters.

    Parameters
    ----------
    config : model_config or dict
        The model configuration, an instance of a :class:`~jade.config.model.generic.model_config` subclass (e.g. :class:`~jade.config.model.EEGNet_cfg.EEGNet_config`) or a plain dict with at least the key ``name`` (promoted to the proper config through :func:`~jade.config.model.get_model_config`).

    Returns
    -------
    torch.nn.Module
        An instance of the requested model.

    Raises
    ------
    ValueError
        If the config ``name`` is not a known model.

    Examples
    --------
    >>> from jade.config.model import EEGNet_config
    >>> get_model(EEGNet_config(name = 'eegnet_classifier', n_classes = 4))
    >>> get_model({'name' : 'eegnet', 'use_bias' : False, 'D' : 2})
    """

    # Accept a plain dict and promote it to the proper model config (an already-built config is returned untouched).
    config = get_model_config(config)

    # The name of the model is the discriminator carried by the config.
    name = str(config.name).lower()
    if name not in MODEL_REGISTRY :
        raise ValueError("unknown model name {}; expected one of {}".format(repr(name), sorted(set(MODEL_REGISTRY))))

    return MODEL_REGISTRY[name](config)
