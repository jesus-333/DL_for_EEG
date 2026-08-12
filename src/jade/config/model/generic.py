"""
Base class for the model configurations.

Every model configuration in :mod:`jade.config.model` is a :func:`dataclasses.dataclass` that inherits from :class:`~jade.config.model.generic.model_config`, which in turn inherits from :class:`~jade.config.generic.generic_config`.
The base carries a single field, ``name``, that acts as a discriminator: it tells apart the concrete model (``eegnet``, ``eegnet_classifier``, ...) when a config is (de)serialized and it is the value used by :func:`~jade.model.get_model` to decide which model to build.

The interesting bit is the overridden :meth:`~jade.config.model.generic.model_config.from_dict`.
When it is called on the base class it reads ``name`` and dispatches to the matching concrete subclass through the registry in :mod:`jade.config.model`.
When it is called on a concrete subclass it just falls back to the standard :meth:`~jade.config.generic.generic_config.from_dict` behavior.
This mirrors the polymorphic pattern already used by :class:`~jade.config.optimizers.generic.optimizer_config` and the other config families.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from dataclasses import dataclass

# Internal imports
from jade.config.generic import generic_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Model config base class

@dataclass
class model_config(generic_config) :
    """
    Base class for every model configuration.

    Parameters
    ----------
    name : str
        Discriminator with the name of the concrete model (e.g. ``'eegnet_classifier'``).
        Concrete subclasses validate (and normalize) this value in their ``__post_init__``.
        It is also the key used by :func:`~jade.model.get_model` to select the model class through the ``MODEL_REGISTRY``.
    """

    name : str = ""


    @classmethod
    def from_dict(cls, data : dict) -> "model_config" :
        """
        Build a model config from a (possibly nested) ``dict``.

        When called on the base class the concrete subclass is selected through the ``name`` field (polymorphic deserialization).
        When called on a concrete subclass the standard :meth:`~jade.config.generic.generic_config.from_dict` is used.

        Parameters
        ----------
        data : dict
            The dictionary to convert into a model config instance.

        Returns
        -------
        model_config
            An instance of the proper concrete subclass.
        """

        # The import is done here (and not at the top of the module) to avoid a circular import.
        # The registry lives in the package __init__, which itself imports this module.
        from jade.config.model import MODEL_CONFIG_REGISTRY

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Polymorphic dispatch (only when called on the base class)

        if cls is model_config :
            # Get the model name
            name = str(data.get("name", "")).lower()

            # Check if the name is known, otherwise raise an error with the list of known models.
            if name not in MODEL_CONFIG_REGISTRY : raise ValueError(f"unknown model name {repr(name)}, known models: {tuple(MODEL_CONFIG_REGISTRY)}")
            
            # Return the proper concrete subclass instance by calling its from_dict method.
            return MODEL_CONFIG_REGISTRY[name].from_dict(data)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Standard behavior (called on a concrete subclass)

        return super().from_dict(data)
