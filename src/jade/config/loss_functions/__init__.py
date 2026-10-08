"""
Loss-function configuration subpackage.

One file per loss, each exposing a :func:`dataclasses.dataclass` config.

Most of them are classification losses and mirror the parameters of the matching :mod:`torch.nn` loss:

* :class:`~jade.config.loss_functions.cross_entropy.cross_entropy_config` - :class:`torch.nn.CrossEntropyLoss` (expects logits; pair with ``use_softmax = False``)
* :class:`~jade.config.loss_functions.nll.nll_config`                     - :class:`torch.nn.NLLLoss` (expects log-probabilities; pair with ``use_softmax = True``)

The remaining ones describe a *custom* loss, i.e. a loss with no :mod:`torch.nn` counterpart, implemented inside :mod:`jade.training.custom_loss_function`:

* :class:`~jade.config.loss_functions.elbo.elbo_config` - :class:`~jade.training.custom_loss_function.elbo.elbo_loss` (expects a generative model, e.g. :class:`~jade.model.vEEGNet.vEEGNet`)

The config of a custom loss lives here, and not next to its implementation, because it is the ``[loss]`` section of a :class:`~jade.config.training.training_config`: it must be reachable through the registry below to be loadable from a file, and (like the rest of :mod:`jade.config`) it must stay torch-free.

The :data:`LOSS_CONFIG_REGISTRY` maps the lowercase loss name to its config class, and :func:`get_loss_config` promotes a plain dict into the proper config instance.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Internal imports
from jade.config.loss_functions.generic import loss_config
from jade.config.loss_functions.cross_entropy import cross_entropy_config
from jade.config.loss_functions.nll import nll_config
from jade.config.loss_functions.elbo import elbo_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Registry and factory

# Map the lowercase loss name to its config class.
LOSS_CONFIG_REGISTRY = {
    "cross_entropy" : cross_entropy_config,
    "nll"           : nll_config,
    "elbo"          : elbo_config,
}


def get_loss_config(config) -> loss_config :
    """
    Return a loss config instance from a dict or an already-built config.

    Parameters
    ----------
    config : loss_config or dict
        Either an already-built loss config (returned untouched) or a dict with at least the key ``name`` (promoted to the proper subclass through the registry).

    Returns
    -------
    loss_config
        The loss configuration instance.
    """

    # Already a config: nothing to do.
    if isinstance(config, loss_config) :
        return config

    # A dict: promote it through the polymorphic from_dict of the base class.
    if isinstance(config, dict) :
        return loss_config.from_dict(config)

    raise TypeError("get_loss_config expects a loss_config or a dict, got {}".format(type(config)))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Public API

__all__ = [
    "loss_config",
    "cross_entropy_config",
    "nll_config",
    "elbo_config",
    "LOSS_CONFIG_REGISTRY",
    "get_loss_config",
]
