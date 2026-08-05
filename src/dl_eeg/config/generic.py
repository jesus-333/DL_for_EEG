"""
Generic configuration base class.
=================================

Every configuration in ``dl_eeg.config`` is a :func:`dataclasses.dataclass` that
inherits from :class:`GenericConfig`. The base class adds no fields of its own; it only
provides (de)serialization helpers shared by every config:

* :meth:`to_dict` / :meth:`from_dict`  - dataclass <-> (nested) plain dict
* :meth:`to_toml` / :meth:`from_toml`  - dataclass <-> TOML file
* :meth:`to_json` / :meth:`from_json`  - dataclass <-> JSON file

Nested configs are handled recursively: a config that holds another ``GenericConfig``
as one of its fields (e.g. ``DatasetConfig`` holds a ``FilterConfig``) is serialized as
a nested structure, which in TOML becomes a dedicated section (e.g. ``[filter]``).

Value validation is left to each subclass, which should implement ``__post_init__``
(dataclasses call it automatically after ``__init__``) and raise ``ValueError`` on bad
values.
"""

from __future__ import annotations

import dataclasses
import json
import typing
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Union

try:
    import toml
except ImportError as exc:  # pragma: no cover
    toml = None
    _TOML_IMPORT_ERROR = exc


def _require_toml():
    if toml is None:  # pragma: no cover
        raise ImportError(
            "TOML support needs the 'toml' package (a declared dependency of this "
            "package). Install it with 'pip install toml'."
        ) from _TOML_IMPORT_ERROR


def _config_type_of(annotation: Any):
    """
    Return the ``GenericConfig`` subclass referenced by a field annotation, if any.

    Handles both a direct annotation (``FilterConfig``) and an ``Optional[FilterConfig]``
    / ``Union[FilterConfig, None]`` annotation. Returns ``None`` when the field is not a
    nested config.
    """
    if isinstance(annotation, type) and issubclass(annotation, GenericConfig):
        return annotation
    if typing.get_origin(annotation) is Union:
        for arg in typing.get_args(annotation):
            if isinstance(arg, type) and issubclass(arg, GenericConfig):
                return arg
    return None


def _strip_none(value: Any) -> Any:
    """
    Recursively drop ``None`` values from dicts/lists.

    TOML has no representation for ``null``, so ``None`` fields are omitted when writing
    a TOML file; on read they simply fall back to their dataclass defaults.
    """
    if isinstance(value, dict):
        return {k: _strip_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_strip_none(v) for v in value]
    return value


@dataclass
class GenericConfig:
    """
    Base class for all configuration dataclasses.

    Carries no fields; provides serialization helpers only. Subclasses add their fields
    and (optionally) a ``__post_init__`` that validates them.
    """

    # -- dict -----------------------------------------------------------------
    def to_dict(self) -> dict:
        """
        Convert this config (recursively) into a plain nested ``dict``.

        Nested ``GenericConfig`` fields become nested dicts, so the structure mirrors
        the sections used by the TOML/JSON serializers.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GenericConfig":
        """
        Build a config from a (possibly nested) ``dict``.

        Nested-config fields whose value is a dict are reconstructed into the proper
        ``GenericConfig`` subclass. Unknown keys are ignored so that forward/backward
        compatible files still load.
        """
        if not isinstance(data, dict):
            raise TypeError(f"{cls.__name__}.from_dict expects a dict, got {type(data)!r}")

        # Resolve annotations to real types (needed because of `from __future__ import
        # annotations`, which turns every annotation into a string).
        hints = typing.get_type_hints(cls)
        known = {f.name for f in fields(cls)}

        kwargs = {}
        for key, value in data.items():
            if key not in known:
                continue  # ignore extra keys
            nested_cls = _config_type_of(hints.get(key))
            if nested_cls is not None and isinstance(value, dict):
                kwargs[key] = nested_cls.from_dict(value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    # -- TOML -----------------------------------------------------------------
    def to_toml(self, path: Union[str, Path]) -> None:
        """Write this config to a TOML file (nested configs -> TOML sections)."""
        _require_toml()
        data = _strip_none(self.to_dict())
        with open(path, "w", encoding="utf-8") as f:
            toml.dump(data, f)

    @classmethod
    def from_toml(cls, path: Union[str, Path]) -> "GenericConfig":
        """Load a config from a TOML file."""
        _require_toml()
        with open(path, "r", encoding="utf-8") as f:
            data = toml.load(f)
        return cls.from_dict(data)

    # -- JSON -----------------------------------------------------------------
    def to_json(self, path: Union[str, Path], indent: int = 4) -> None:
        """Write this config to a JSON file (nested configs -> nested objects)."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "GenericConfig":
        """Load a config from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
