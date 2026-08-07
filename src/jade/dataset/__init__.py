"""

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Internal imports
from jade.dataset.dataset_time import EEG_Dataset, EEG_Dataset_ChWi

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Public API

__all__ = [
    "EEG_Dataset",
    "EEG_Dataset_ChWi",
    "download_dataset",
    "get_dataset",
]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Lazy re-export of the moabb loader

def __getattr__(name : str) :
    """
    Lazily expose the moabb loader functions (PEP 562).

    This keeps the optional moabb/mne dependency out of the import path unless the moabb loader is actually used.

    Parameters
    ----------
    name : str
        The attribute being accessed.

    Returns
    -------
    Any
        The requested loader function.
    """

    if name in ("download_dataset", "get_dataset") :
        from jade.dataset import moabb_dataset
        return getattr(moabb_dataset, name)
    raise AttributeError("module {} has no attribute {}".format(repr(__name__), repr(name)))
