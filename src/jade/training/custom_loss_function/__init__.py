"""
Custom loss functions.

This subpackage holds the losses that have no :mod:`torch.nn` counterpart and must therefore be implemented inside ``jade``.
Currently there is a single member, the ELBO used to train a variational autoencoder:

* :class:`~jade.training.custom_loss_function.elbo.elbo_loss` - described by :class:`~jade.config.loss_functions.elbo.elbo_config`

Note the split between the two halves of a custom loss: the *implementation* lives here (it needs torch), while its *config* lives in :mod:`jade.config.loss_functions` together with every other loss config.
The config has to be there because it is the ``[loss]`` section of a :class:`~jade.config.training.training_config`, so it must be reachable through :data:`~jade.config.loss_functions.LOSS_CONFIG_REGISTRY` to be loadable from a file, and because :mod:`jade.config` is deliberately torch-free.

The :data:`CUSTOM_LOSS_REGISTRY` maps the lowercase loss name to its class and :func:`get_custom_loss_function` builds it, mirroring the registry plus factory pattern used everywhere else in the package (see :func:`jade.model.get_model` and :func:`jade.model.pooling_1d.get_pooling_layer`).
The losses of this subpackage are not built directly by the user: :func:`~jade.training.support_training.get_loss_function` dispatches here when the loss name is not a classification one.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Third party imports
from torch import nn

# Internal imports
from jade.config.loss_functions import get_loss_config
from jade.training.custom_loss_function import support_loss
from jade.training.custom_loss_function.support_loss import (
    RECONSTRUCTION_LOSS_REGISTRY,
    get_reconstruction_loss_function,
    kl_divergence_normal,
    compute_kl_weight,
)
from jade.training.custom_loss_function.elbo import elbo_loss, elbo_terms

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Registry and factory

# Map the lowercase loss name to its class. The keys must match the ones of jade.config.loss_functions.LOSS_CONFIG_REGISTRY.
CUSTOM_LOSS_REGISTRY = {
    "elbo" : elbo_loss,
}


def get_custom_loss_function(config) -> nn.Module :
    """
    Create the custom loss module described by a loss config.

    Parameters
    ----------
    config : loss_config or dict
        The loss configuration (or a dict promoted through :func:`~jade.config.loss_functions.get_loss_config`).
        The concrete loss is chosen from the ``name`` field.

    Returns
    -------
    torch.nn.Module
        The loss module. Note that a custom loss does not necessarily return a single scalar: :class:`~jade.training.custom_loss_function.elbo.elbo_loss` returns an :class:`~jade.training.custom_loss_function.elbo.elbo_terms` named tuple.

    Raises
    ------
    ValueError
        If ``config.name`` is not a known custom loss.

    Examples
    --------
    >>> from jade.config import elbo_config
    >>> from jade.training.custom_loss_function import get_custom_loss_function
    >>> loss_function = get_custom_loss_function(elbo_config(beta = 0.5))
    """

    # Accept both a config instance and a plain dict.
    config = get_loss_config(config)

    name = str(config.name).lower()
    if name not in CUSTOM_LOSS_REGISTRY :
        raise ValueError("unknown custom loss name {}; expected one of {}".format(repr(name), sorted(CUSTOM_LOSS_REGISTRY)))

    return CUSTOM_LOSS_REGISTRY[name](config)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Public API

__all__ = [
    # Losses
    "elbo_loss",
    "elbo_terms",
    # Registry and factory
    "CUSTOM_LOSS_REGISTRY",
    "get_custom_loss_function",
    # Support functions
    "support_loss",
    "RECONSTRUCTION_LOSS_REGISTRY",
    "get_reconstruction_loss_function",
    "kl_divergence_normal",
    "compute_kl_weight",
]
