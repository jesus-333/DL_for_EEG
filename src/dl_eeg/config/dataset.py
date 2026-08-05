"""
Dataset configuration.
======================

``DatasetConfig`` gathers everything needed to download and preprocess a moabb dataset
before it is wrapped into an ``EEG_Dataset``:

* which dataset and which subject(s) to download,
* the (band-pass / notch / ...) filtering, as a nested :class:`FilterConfig`,
* optional resampling (a flag plus the target frequency),
* the normalization scheme forwarded to ``EEG_Dataset``.

Serialized to TOML the nested filter becomes its own ``[filter]`` section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from dl_eeg.config.filter import FilterConfig
from dl_eeg.config.generic import GenericConfig

#: Normalization codes understood by ``dl_eeg.dataset.dataset_time.EEG_Dataset``.
#: -1 = none, 1 = min-max over the whole dataset, 2 = per-trial per-channel,
#: 3 = per-trial.
NORMALIZE_CODES = (-1, 1, 2, 3)


def _default_filter() -> FilterConfig:
    # A sensible default band-pass (close to moabb's motor-imagery default of 7-35 Hz).
    return FilterConfig(filter_type="bandpass", l_freq=8.0, h_freq=32.0)


@dataclass
class DatasetConfig(GenericConfig):
    """
    Configuration to download and preprocess a moabb dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the moabb dataset to load (e.g. ``'BNCI2014_001'``). Required.
    subjects : list of int or None, default None
        Subjects to download. ``None`` means every subject of the dataset. Each entry
        must be a positive integer.
    filter : FilterConfig
        Filtering to apply to the raw signal (nested config). Defaults to an 8-32 Hz
        band-pass.
    resample : bool, default False
        Whether to resample the signal after filtering.
    resample_freq : float or None, default None
        Target sampling frequency (Hz). Required (and must be positive) when
        ``resample`` is True.
    normalize : int, default -1
        Normalization scheme forwarded to ``EEG_Dataset`` (one of ``NORMALIZE_CODES``).
    """

    dataset_name: str = ""
    subjects: Optional[List[int]] = None
    filter: FilterConfig = field(default_factory=_default_filter)
    resample: bool = False
    resample_freq: Optional[float] = None
    normalize: int = -1

    def __post_init__(self) -> None:
        if not self.dataset_name:
            raise ValueError("dataset_name must be a non-empty string")

        if self.subjects is not None:
            if not isinstance(self.subjects, (list, tuple)):
                raise ValueError(
                    f"subjects must be a list of ints or None, got {type(self.subjects)!r}"
                )
            cleaned = []
            for s in self.subjects:
                if not isinstance(s, int) or isinstance(s, bool) or s <= 0:
                    raise ValueError(f"every subject must be a positive int, got {s!r}")
                cleaned.append(int(s))
            self.subjects = cleaned

        # A dict coming from a TOML/JSON file is promoted to a FilterConfig so the
        # nested config is always a proper dataclass instance.
        if isinstance(self.filter, dict):
            self.filter = FilterConfig.from_dict(self.filter)
        elif not isinstance(self.filter, FilterConfig):
            raise ValueError(
                f"filter must be a FilterConfig (or a dict), got {type(self.filter)!r}"
            )

        if self.resample:
            if self.resample_freq is None or self.resample_freq <= 0:
                raise ValueError(
                    "resample is True, so resample_freq must be a positive number "
                    f"(got {self.resample_freq!r})"
                )

        if self.normalize not in NORMALIZE_CODES:
            raise ValueError(
                f"normalize must be one of {NORMALIZE_CODES}, got {self.normalize!r}"
            )
