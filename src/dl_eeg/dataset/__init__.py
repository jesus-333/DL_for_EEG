"""
Dataset subpackage: PyTorch EEG datasets and loaders.

``EEG_Dataset`` / ``EEG_Dataset_ChWi`` are always available. The moabb loader
(``download_dataset`` / ``get_dataset``) is re-exported lazily via ``__getattr__`` so
that ``import dl_eeg.dataset`` does not require the optional ``moabb``/``mne`` extra;
those are only imported when the loader functions are actually accessed.
"""

from __future__ import annotations

from dl_eeg.dataset.dataset_time import EEG_Dataset, EEG_Dataset_ChWi

__all__ = [
    "EEG_Dataset",
    "EEG_Dataset_ChWi",
    "download_dataset",
    "get_dataset",
]


def __getattr__(name: str):
    # PEP 562 lazy attribute access: keeps the optional moabb/mne dependency out of
    # the import path unless the moabb loader is actually used.
    if name in ("download_dataset", "get_dataset"):
        from dl_eeg.dataset import moabb_dataset

        return getattr(moabb_dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
