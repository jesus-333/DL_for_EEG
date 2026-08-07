"""
Base class for the learning-rate scheduler configurations.

Every scheduler configuration in :mod:`jade.config.lr_schedulers` is a :func:`dataclasses.dataclass` that inherits from :class:`~jade.config.lr_schedulers.generic.lr_scheduler_config`, which in turn inherits from :class:`~jade.config.generic.generic_config`.
As for the optimizer configs, the base carries a single ``name`` discriminator and overrides :meth:`~jade.config.lr_schedulers.generic.lr_scheduler_config.from_dict` so that a nested scheduler config can be rebuilt polymorphically from a plain dict.
The scheduler itself is built from a config by :func:`~jade.training.support_training.get_lr_scheduler`.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from dataclasses import dataclass

# Internal imports
from jade.config.generic import generic_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Learning-rate scheduler config base class

@dataclass
class lr_scheduler_config(generic_config) :
    """
    Base class for every learning-rate scheduler configuration.

    Parameters
    ----------
    name : str
        Discriminator with the name of the concrete scheduler (e.g. ``'exponential_lr'``).
        Concrete subclasses set (and freeze) this value in their ``__post_init__``.
    """

    name : str = ""


    @classmethod
    def from_dict(cls, data : dict) -> "lr_scheduler_config" :
        """
        Build a scheduler config from a (possibly nested) ``dict``.

        When called on the base class the concrete subclass is selected through the ``name`` field (polymorphic deserialization).
        When called on a concrete subclass the standard :meth:`~jade.config.generic.generic_config.from_dict` is used.

        Parameters
        ----------
        data : dict
            The dictionary to convert into a scheduler config instance.

        Returns
        -------
        lr_scheduler_config
            An instance of the proper concrete subclass.
        """

        # Local import to avoid a circular import with the package __init__ (which imports this module).
        from jade.config.lr_schedulers import LR_SCHEDULER_CONFIG_REGISTRY

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Polymorphic dispatch (only when called on the base class)

        if cls is lr_scheduler_config :
            name = str(data.get("name", "")).lower()
            if name not in LR_SCHEDULER_CONFIG_REGISTRY :
                raise ValueError("unknown lr scheduler name {}, known schedulers: {}".format(repr(name), tuple(LR_SCHEDULER_CONFIG_REGISTRY)))
            return LR_SCHEDULER_CONFIG_REGISTRY[name].from_dict(data)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Standard behavior (called on a concrete subclass)

        return super().from_dict(data)
