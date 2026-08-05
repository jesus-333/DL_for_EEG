"""
Generic configuration base class.

Every configuration in :mod:`dl_eeg.config` is a :func:`dataclasses.dataclass` that inherits from :class:`~dl_eeg.config.generic.generic_config`.
The base class adds no fields of its own; it only provides the (de)serialization helpers shared by every config: :meth:`~dl_eeg.config.generic.generic_config.to_dict`, :meth:`~dl_eeg.config.generic.generic_config.from_dict`, :meth:`~dl_eeg.config.generic.generic_config.to_toml`, :meth:`~dl_eeg.config.generic.generic_config.from_toml`, :meth:`~dl_eeg.config.generic.generic_config.to_json` and :meth:`~dl_eeg.config.generic.generic_config.from_json`.

Nested configs are handled recursively: a config that holds another config as one of its fields (e.g. :class:`~dl_eeg.config.dataset.dataset_config` holds a :class:`~dl_eeg.config.filter.filter_config`) is serialized as a nested structure, which in TOML becomes a dedicated section (e.g. ``[filter]``).

Value validation is left to each subclass, which should implement ``__post_init__`` (dataclasses call it automatically after ``__init__``) and raize ``ValueError`` on bad values.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Full module imports
import dataclasses
import json
import os
import typing

# Specific imports
from dataclasses import dataclass, fields
from typing import Any, Union

# Optional third party import (toml is a declared dependency of the package)
try :
    import toml
except ImportError as exc :  # pragma: no cover
    toml = None
    _TOML_IMPORT_ERROR = exc

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Helper functions

def _require_toml() :
    """
    Raize a clear error if the optional ``toml`` package is not available.
    """

    if toml is None :  # pragma: no cover
        raise ImportError(
            "TOML support needs the 'toml' package (a declared dependency of this package). "
            "Install it with 'pip install toml'."
        ) from _TOML_IMPORT_ERROR


def _config_type_of(annotation : Any) :
    """
    Return the config subclass referenced by a field annotation, if any.

    It handles both a direct annotation (``filter_config``) and an ``Optional[filter_config]`` / ``Union[filter_config, None]`` annotation.
    It returns ``None`` when the field is not a nested config.

    Parameters
    ----------
    annotation : Any
        The resolved type annotation of a dataclass field.

    Returns
    -------
    type or None
        The :class:`~dl_eeg.config.generic.generic_config` subclass, or ``None``.
    """

    # Direct annotation (the field is exactly a config subclass)
    if isinstance(annotation, type) and issubclass(annotation, generic_config) :
        return annotation

    # Optional[...] / Union[..., None] annotation: scan the members for a config subclass
    if typing.get_origin(annotation) is Union :
        for arg in typing.get_args(annotation) :
            if isinstance(arg, type) and issubclass(arg, generic_config) :
                return arg

    return None


def _strip_none(value : Any) -> Any :
    """
    Recursively drop ``None`` values from dicts/lists.

    TOML has no representation for ``null``, so ``None`` fields are omitted when writing a TOML file; on read they simply fall back to their dataclass defaults.

    Parameters
    ----------
    value : Any
        A value (possibly a nested dict/list) to clean.

    Returns
    -------
    Any
        The same structure without any ``None`` entry.
    """

    if isinstance(value, dict) :
        return {key : _strip_none(inner) for key, inner in value.items() if inner is not None}
    if isinstance(value, (list, tuple)) :
        return [_strip_none(inner) for inner in value]
    return value

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Generic config class

@dataclass
class generic_config :
    """
    Base class for all configuration dataclasses.

    It carries no fields; it provides serialization helpers only.
    Subclasses add their fields and (optionally) a ``__post_init__`` that validates them.
    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # dict conversion

    def to_dict(self) -> dict :
        """
        Convert this config (recursively) into a plain nested ``dict``.

        Nested config fields become nested dicts, so the structure mirrors the sections used by the TOML/JSON serializers.

        Returns
        -------
        dict
            A nested dictionary representation of the config.
        """

        return dataclasses.asdict(self)


    @classmethod
    def from_dict(cls, data : dict) -> "generic_config" :
        """
        Build a config from a (possibly nested) ``dict``.

        Nested config fields whose value is a dict are reconstructed into the proper config subclass.
        Unknown keys are ignored so that forward/backward compatible files still load.

        Parameters
        ----------
        data : dict
            The dictionary to convert into a config instance.

        Returns
        -------
        generic_config
            An instance of the calling subclass.
        """

        # Input check
        if not isinstance(data, dict) :
            raise TypeError("{}.from_dict expects a dict, got {}".format(cls.__name__, type(data)))

        # Resolve annotations to real types.
        # This is needed because of `from __future__ import annotations`, which turns every annotation into a string.
        hints = typing.get_type_hints(cls)
        known_fields = {field.name for field in fields(cls)}

        # Build the keyword arguments, recursing into nested configs.
        kwargs = {}
        for key, value in data.items() :
            # Ignore keys that are not fields of this config.
            if key not in known_fields :
                continue

            # If the field is a nested config and the value is a dict, rebuild the nested config.
            nested_cls = _config_type_of(hints.get(key))
            if nested_cls is not None and isinstance(value, dict) :
                kwargs[key] = nested_cls.from_dict(value)
            else :
                kwargs[key] = value

        return cls(**kwargs)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # TOML conversion

    def to_toml(self, path : Union[str, "os.PathLike"]) -> None :
        """
        Write this config to a TOML file (nested configs become TOML sections).

        Parameters
        ----------
        path : str or path-like
            Destination file path.
        """

        _require_toml()

        # Drop None values (TOML has no null) and dump.
        data = _strip_none(self.to_dict())
        with open(path, "w", encoding = "utf-8") as file_handler :
            toml.dump(data, file_handler)


    @classmethod
    def from_toml(cls, path : Union[str, "os.PathLike"]) -> "generic_config" :
        """
        Load a config from a TOML file.

        Parameters
        ----------
        path : str or path-like
            Source file path.

        Returns
        -------
        generic_config
            An instance of the calling subclass.
        """

        _require_toml()

        with open(path, "r", encoding = "utf-8") as file_handler :
            data = toml.load(file_handler)
        return cls.from_dict(data)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # JSON conversion

    def to_json(self, path : Union[str, "os.PathLike"], indent : int = 4) -> None :
        """
        Write this config to a JSON file (nested configs become nested objects).

        Parameters
        ----------
        path : str or path-like
            Destination file path.
        indent : int, default 4
            Indentation passed to :func:`json.dump`.
        """

        with open(path, "w", encoding = "utf-8") as file_handler :
            json.dump(self.to_dict(), file_handler, indent = indent)


    @classmethod
    def from_json(cls, path : Union[str, "os.PathLike"]) -> "generic_config" :
        """
        Load a config from a JSON file.

        Parameters
        ----------
        path : str or path-like
            Source file path.

        Returns
        -------
        generic_config
            An instance of the calling subclass.
        """

        with open(path, "r", encoding = "utf-8") as file_handler :
            data = json.load(file_handler)
        return cls.from_dict(data)
