"""
Base class for the loss-function configurations.

Every loss configuration in :mod:`jade.config.loss_functions` is a :func:`dataclasses.dataclass` that inherits from :class:`~jade.config.loss_functions.generic.loss_config`, which in turn inherits from :class:`~jade.config.generic.generic_config`.
As for the optimizer/scheduler configs, the base carries a single ``name`` discriminator and overrides :meth:`~jade.config.loss_functions.generic.loss_config.from_dict` so that a nested loss config can be rebuilt polymorphically from a plain dict.
The loss module itself is built from a config by :func:`~jade.training.support_training.get_loss_function`.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from dataclasses import dataclass

# Internal imports
from jade.config.generic import generic_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Loss-function config base class

@dataclass
class loss_config(generic_config) :
    """
    Base class for every loss-function configuration.

    Parameters
    ----------
    name : str
        Discriminator with the name of the concrete loss (e.g. ``'cross_entropy'``).
        Concrete subclasses set (and freeze) this value in their ``__post_init__``.

    Notes
    -----
    :meth:`from_dict` rejects the key ``weight``, which is the old name of the per-class coefficients now called ``class_weight``.
    The check exists because :meth:`~jade.config.generic.generic_config.from_dict` silently ignores every unknown key, so an outdated config file would otherwise train without its class weights and without any warning.
    """

    name : str = ""


    @classmethod
    def from_dict(cls, data : dict) -> "loss_config" :
        """
        Build a loss config from a (possibly nested) ``dict``.

        When called on the base class the concrete subclass is selected through the ``name`` field (polymorphic deserialization).
        When called on a concrete subclass the standard :meth:`~jade.config.generic.generic_config.from_dict` is used.

        Parameters
        ----------
        data : dict
            The dictionary to convert into a loss config instance.

        Returns
        -------
        loss_config
            An instance of the proper concrete subclass.
        """

        # Local import to avoid a circular import with the package __init__ (which imports this module).
        from jade.config.loss_functions import LOSS_CONFIG_REGISTRY

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Legacy key check

        # The per-class coefficients used to be called 'weight' (as in torch), which is easy to confuse with the trainable weights of a model, so they are now called 'class_weight'.
        # generic_config.from_dict ignores every unknown key, so an old file would silently train WITHOUT its class weights. Fail loudly instead.
        if isinstance(data, dict) and "weight" in data :
            raise ValueError("the loss field 'weight' has been renamed 'class_weight' (it holds the per-class coefficients, not the weights of a model): rename the key in your config")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Polymorphic dispatch (only when called on the base class)

        if cls is loss_config :
            name = str(data.get("name", "")).lower()
            if name not in LOSS_CONFIG_REGISTRY :
                raise ValueError("unknown loss name {}, known losses: {}".format(repr(name), tuple(LOSS_CONFIG_REGISTRY)))
            return LOSS_CONFIG_REGISTRY[name].from_dict(data)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Standard behavior (called on a concrete subclass)

        return super().from_dict(data)
