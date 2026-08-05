"""
Configuration subpackage.
=========================

Dataclass-based configuration objects for the package, one file per concern:

* :class:`~dl_eeg.config.generic.GenericConfig` - base class with the
  ``to_dict`` / ``from_dict`` / ``to_toml`` / ``from_toml`` / ``to_json`` / ``from_json``
  (de)serialization helpers that every config inherits.
* :class:`~dl_eeg.config.filter.FilterConfig`   - moabb/MNE filtering parameters.
* :class:`~dl_eeg.config.dataset.DatasetConfig` - dataset download + preprocessing
  (holds a nested ``FilterConfig``).
* :class:`~dl_eeg.config.training.TrainingConfig` - training hyper-parameters.

Model configuration is intentionally not implemented yet.
"""

from __future__ import annotations

from dl_eeg.config.dataset import DatasetConfig
from dl_eeg.config.filter import FilterConfig
from dl_eeg.config.generic import GenericConfig
from dl_eeg.config.training import TrainingConfig

__all__ = [
    "GenericConfig",
    "FilterConfig",
    "DatasetConfig",
    "TrainingConfig",
]
