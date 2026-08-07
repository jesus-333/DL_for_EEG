"""
Base class for the optimizer configurations.

Every optimizer configuration in :mod:`jade.config.optimizers` is a :func:`dataclasses.dataclass` that inherits from :class:`~jade.config.optimizers.generic.optimizer_config`, which in turn inherits from :class:`~jade.config.generic.generic_config`.
The base carries a single field, ``name``, that acts as a discriminator: it tells apart the concrete optimizer (``adam``, ``sgd``, ...) when a config is (de)serialized.

The interesting bit is the overridden :meth:`~jade.config.optimizers.generic.optimizer_config.from_dict`.
When it is called on the base class (as happens when a :class:`~jade.config.training.training_config` rebuilds its nested ``optimizer`` field from a plain dict) it reads ``name`` and dispatches to the matching concrete subclass through the registry in :mod:`jade.config.optimizers`.
When it is called on a concrete subclass it just falls back to the standard :meth:`~jade.config.generic.generic_config.from_dict` behavior.
This way the existing recursive (de)serialization of :class:`~jade.config.generic.generic_config` keeps working for polymorphic nested configs without any change.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from dataclasses import dataclass

# Internal imports
from jade.config.generic import generic_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Optimizer config base class

@dataclass
class optimizer_config(generic_config) :
    """
    Base class for every optimizer configuration.

    Parameters
    ----------
    name : str
        Discriminator with the name of the concrete optimizer (e.g. ``'adam'``).
        Concrete subclasses set (and freeze) this value in their ``__post_init__``, so it normally does not need to be passed by hand.
    """

    name : str = ""


    @classmethod
    def from_dict(cls, data : dict) -> "optimizer_config" :
        """
        Build an optimizer config from a (possibly nested) ``dict``.

        When called on the base class the concrete subclass is selected through the ``name`` field (polymorphic deserialization).
        When called on a concrete subclass the standard :meth:`~jade.config.generic.generic_config.from_dict` is used.

        Parameters
        ----------
        data : dict
            The dictionary to convert into an optimizer config instance.

        Returns
        -------
        optimizer_config
            An instance of the proper concrete subclass.
        """

        # The import is done here (and not at the top of the module) to avoid a circular import.
        # The registry lives in the package __init__, which itself imports this module.
        from jade.config.optimizers import OPTIMIZER_CONFIG_REGISTRY

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Polymorphic dispatch (only when called on the base class)

        if cls is optimizer_config :
            name = str(data.get("name", "")).lower()
            if name not in OPTIMIZER_CONFIG_REGISTRY :
                raise ValueError("unknown optimizer name {}, known optimizers: {}".format(repr(name), tuple(OPTIMIZER_CONFIG_REGISTRY)))
            return OPTIMIZER_CONFIG_REGISTRY[name].from_dict(data)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Standard behavior (called on a concrete subclass)

        return super().from_dict(data)
