"""
Dataset configuration.

:class:`~jade.config.dataset.dataset_config` gathers everything needed to download and preprocess a moabb dataset before it is wrapped into an :class:`~jade.dataset.dataset_time.EEG_Dataset`:

* which dataset and which subject(s) to download,
* the (band-pass / notch / ...) filtering, as a nested :class:`~jade.config.filter.filter_config`,
* optional resampling (a flag plus the target frequency),
* the normalization scheme forwarded to :class:`~jade.dataset.dataset_time.EEG_Dataset`.

Serialized to TOML the nested filter becomes its own ``[filter]`` section.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from dataclasses import dataclass, field
from typing import List, Optional

# Internal imports
from jade.config.filter import filter_config
from jade.config.generic import generic_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module constants

# Normalization codes understood by jade.dataset.dataset_time.EEG_Dataset.
# -1 = none, 1 = min-max over the whole dataset, 2 = per-trial per-channel, 3 = per-trial.
NORMALIZE_CODES = (-1, 1, 2, 3)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Helper functions

def _default_filter() -> filter_config :
    """
    Build the default filter (a sensible band-pass close to moabb's motor-imagery default of 7-35 Hz).

    Returns
    -------
    filter_config
        An 8-32 Hz band-pass filter config.
    """

    return filter_config(filter_type = "bandpass", l_freq = 8.0, h_freq = 32.0)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dataset config class

@dataclass
class dataset_config(generic_config) :
    """
    Configuration to download and preprocess a moabb dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the moabb dataset to load (e.g. ``'BNCI2014_001'``). Required.
    subjects : list of int or None, default None
        Subjects to download. ``None`` means every subject of the dataset.
        Each entry must be a positive integer.
    filter : filter_config
        Filtering to apply to the raw signal (nested config). Defaults to an 8-32 Hz band-pass.
    resample : bool, default False
        Whether to resample the signal after filtering.
    resample_freq : float or None, default None
        Target sampling frequency (Hz). Required (and must be positive) when ``resample`` is True.
    normalize : int, default -1
        Normalization scheme forwarded to :class:`~jade.dataset.dataset_time.EEG_Dataset` (one of ``NORMALIZE_CODES``).
    """

    dataset_name  : str = ""
    subjects      : Optional[List[int]] = None
    filter        : filter_config = field(default_factory = _default_filter)
    resample      : bool = False
    resample_freq : Optional[float] = None
    normalize     : int = -1


    def __post_init__(self) -> None :
        """
        Validate the dataset parameters after initialization.
        """

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Check the dataset name

        if not self.dataset_name :
            raise ValueError("dataset_name must be a non-empty string")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Check the subjects list

        if self.subjects is not None :
            if not isinstance(self.subjects, (list, tuple)) :
                raise ValueError("subjects must be a list of ints or None, got {}".format(type(self.subjects)))

            # Every subject must be a positive int (bool is explicitly rejected, since bool is a subclass of int).
            cleaned_subjects = []
            for subject in self.subjects :
                if not isinstance(subject, int) or isinstance(subject, bool) or subject <= 0 :
                    raise ValueError("every subject must be a positive int, got {}".format(repr(subject)))
                cleaned_subjects.append(int(subject))
            self.subjects = cleaned_subjects

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Promote a dict filter to a filter_config

        # A dict coming from a TOML/JSON file is promoted to a filter_config so the nested config is always a proper dataclass instance.
        if isinstance(self.filter, dict) :
            self.filter = filter_config.from_dict(self.filter)
        elif not isinstance(self.filter, filter_config) :
            raise ValueError("filter must be a filter_config (or a dict), got {}".format(type(self.filter)))

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Check resampling and normalization

        if self.resample :
            if self.resample_freq is None or self.resample_freq <= 0 :
                raise ValueError("resample is True, so resample_freq must be a positive number (got {})".format(repr(self.resample_freq)))

        if self.normalize not in NORMALIZE_CODES :
            raise ValueError("normalize must be one of {}, got {}".format(NORMALIZE_CODES, repr(self.normalize)))
