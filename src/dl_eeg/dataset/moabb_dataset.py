"""
Interaction with the `moabb <https://moabb.neurotechx.com/>`_ package.

This module turns the EEG datasets bundled with moabb into ``torch``-ready datasets,
i.e. instances of :class:`dl_eeg.dataset.dataset_time.EEG_Dataset`.

It exposes two public functions:

* :func:`download_dataset` - download the raw recordings of a moabb dataset (all
  subjects, or a specific subset).
* :func:`get_dataset` - load a moabb dataset, preprocess it (filtering + resampling
  through moabb/MNE's own routines), epoch it into trials and wrap it in an
  ``EEG_Dataset`` (channel names included). If the data is not on disk yet it is
  downloaded automatically via :func:`download_dataset`.

``moabb`` and ``mne`` are heavy, optional dependencies (declared in the ``dataset``
extra of ``pyproject.toml``). They are therefore imported lazily, inside the
functions that need them, so ``import dl_eeg`` works without them installed. Install
them with ``pip install jade[dataset]``.

Preprocessing config schema
---------------------------
The ``preprocess`` argument of :func:`get_dataset` is a plain dict::

    preprocess = {
        # Resampling: a boolean flag plus the target frequency (Hz).
        "resample": {"enabled": True, "freq": 128},

        # Filtering: an ordered list of filters, each applied in turn to the raw
        # signal using MNE's own filtering routines (no custom filters here).
        # Supported "type" values: "bandpass", "lowpass", "highpass", "notch".
        #   - bandpass  -> band = [l_freq, h_freq]
        #   - lowpass   -> band = [h_freq]            (or [None, h_freq])
        #   - highpass  -> band = [l_freq]            (or [l_freq, None])
        #   - notch     -> band = [f0, f1, ...]       (frequencies to notch out)
        "filters": [
            {"type": "bandpass", "band": [8, 32]},
            {"type": "notch",    "band": [50]},
        ],

        # Optional normalization forwarded to EEG_Dataset (see its docstring):
        # -1 = none (default), 1 = min-max over the whole dataset,
        #  2 = per-trial per-channel, 3 = per-trial.
        "normalize": -1,
    }

Any of the keys may be omitted: no ``"resample"``/``"filters"`` means the signal is
left untouched, and ``"normalize"`` defaults to ``-1``.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Union

import numpy as np

from dl_eeg.dataset.dataset_time import EEG_Dataset

__all__ = ["download_dataset", "get_dataset"]


# ---------------------------------------------------------------------------------
# Lazy dependency import
# ---------------------------------------------------------------------------------
def _import_moabb_mne():
    """
    Import ``moabb`` and ``mne`` on demand, with a helpful error if they are missing.

    Returns
    -------
    (moabb_datasets_module, mne_module)
    """
    try:
        import mne
        from moabb import datasets as moabb_datasets
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "moabb_dataset needs the optional 'dataset' dependencies (moabb + mne). "
            "Install them with:  pip install jade[dataset]  (or 'pip install moabb mne')."
        ) from exc
    return moabb_datasets, mne


# ---------------------------------------------------------------------------------
# Dataset name resolution
# ---------------------------------------------------------------------------------
def _resolve_dataset_class(dataset_name: str):
    """
    Resolve a (case-insensitive) dataset name to a moabb dataset class.

    moabb exposes dataset classes such as ``BNCI2014_001``, ``Cho2017``,
    ``PhysionetMI`` under ``moabb.datasets``. The lookup is case-insensitive and also
    ignores underscores, so ``"bnci2014_001"``, ``"BNCI2014001"`` and ``"BNCI2014_001"``
    all resolve to the same class.
    """
    moabb_datasets, _ = _import_moabb_mne()

    def _norm(s: str) -> str:
        return s.lower().replace("_", "")

    target = _norm(dataset_name)

    # Direct attribute hit first (fast path).
    if hasattr(moabb_datasets, dataset_name):
        return getattr(moabb_datasets, dataset_name)

    # Case-/underscore-insensitive scan over the module's public dataset classes.
    matches = {}
    for attr in dir(moabb_datasets):
        if attr.startswith("_"):
            continue
        obj = getattr(moabb_datasets, attr)
        if isinstance(obj, type):  # a class
            matches[_norm(attr)] = obj

    if target in matches:
        return matches[target]

    available = sorted(a for a in dir(moabb_datasets) if not a.startswith("_")
                       and isinstance(getattr(moabb_datasets, a), type))
    raise ValueError(
        f"unknown moabb dataset {dataset_name!r}. Available dataset classes under "
        f"moabb.datasets include: {available}"
    )


def _as_subject_list(subjects, dataset) -> list:
    """
    Normalise ``subjects`` to a list of subject ids.

    Accepts a single int, a list/tuple of ints, or ``None`` (meaning *all* subjects of
    the dataset, i.e. ``dataset.subject_list``).
    """
    if subjects is None:
        return list(dataset.subject_list)
    if isinstance(subjects, (int, np.integer)):
        return [int(subjects)]
    if isinstance(subjects, (list, tuple, set, np.ndarray)):
        return [int(s) for s in subjects]
    raise TypeError(
        f"subjects must be an int, a list of ints or None, got {type(subjects)!r}"
    )


# ---------------------------------------------------------------------------------
# Public: download
# ---------------------------------------------------------------------------------
def download_dataset(
    dataset_name: str,
    subjects: Optional[Union[int, Sequence[int]]] = None,
):
    """
    Download a moabb dataset (all subjects, or a specific subset).

    Parameters
    ----------
    dataset_name : str
        Name of a moabb dataset (case-insensitive), e.g. ``"BNCI2014_001"``.
    subjects : int or sequence of int or None, optional
        Which subject(s) to download. A single int or a list of ints selects those
        subjects; ``None`` (the default) downloads *every* subject of the dataset.

    Returns
    -------
    moabb.datasets.base.BaseDataset
        The instantiated dataset object (already pointing at the downloaded files),
        ready to be passed to ``get_data``.

    Notes
    -----
    moabb caches downloads under its data directory, so calling this repeatedly is
    cheap: already-present subjects are not re-downloaded.
    """
    dataset_cls = _resolve_dataset_class(dataset_name)
    dataset = dataset_cls()
    subject_list = _as_subject_list(subjects, dataset)
    # moabb's BaseDataset.download fetches every requested subject; already-cached
    # ones are skipped.
    dataset.download(subject_list=subject_list)
    return dataset


# ---------------------------------------------------------------------------------
# Train / test session selection
# ---------------------------------------------------------------------------------
def _session_matches(session_name: str, dataset_type: str) -> bool:
    """
    Heuristic: does ``session_name`` belong to the requested split (train/test)?

    Handles the common moabb session-naming conventions, e.g. BNCI2014_001's
    ``'0train'`` / ``'1test'`` and the older ``'session_T'`` / ``'session_E'``.
    """
    low = session_name.lower()
    is_test = ("test" in low) or ("eval" in low) or low.endswith("_e") or low == "e"
    if dataset_type == "test":
        return is_test
    # train
    is_train = ("train" in low) or low.endswith("_t") or low == "t"
    return is_train


def _select_sessions(session_names: Sequence[str], dataset_type: str) -> list:
    """
    Pick the session keys corresponding to ``dataset_type`` ('train' or 'test').

    Strategy
    --------
    1. If any session name matches the split by the naming heuristic, return those.
    2. Otherwise fall back on position: with a single session it is used for both
       splits (a warning is emitted); with several sessions the last one is the test
       split and the rest are the train split.
    """
    names = sorted(session_names)
    matched = [s for s in names if _session_matches(s, dataset_type)]
    if matched:
        return matched

    if len(names) == 1:
        warnings.warn(
            f"Dataset has a single session {names[0]!r}; using it for the "
            f"{dataset_type!r} split as well. Provide an explicit split upstream if "
            "you need train/test isolation.",
            RuntimeWarning,
            stacklevel=3,
        )
        return list(names)

    warnings.warn(
        f"Could not map sessions {names} to a {dataset_type!r} split by name; falling "
        "back to positional split (last session = test, the rest = train).",
        RuntimeWarning,
        stacklevel=3,
    )
    return names[:-1] if dataset_type == "train" else names[-1:]


# ---------------------------------------------------------------------------------
# Preprocessing (filtering + resampling) via MNE, on the raw objects moabb returns
# ---------------------------------------------------------------------------------
def _apply_preprocess(raw, preprocess: dict):
    """
    Apply the configured filters and resampling to an MNE ``Raw`` object, in place.

    Filtering uses MNE's own routines (``raw.filter`` / ``raw.notch_filter``) - no
    custom filter is implemented here, as requested. Filters are applied in the order
    given, and resampling (if enabled) is applied last so it acts on the already
    band-limited signal.
    """
    if preprocess is None:
        return raw

    # Filtering requires the data in memory.
    raw.load_data(verbose=False)

    for filt in preprocess.get("filters", []) or []:
        ftype = str(filt["type"]).lower()
        band = filt.get("band")

        if ftype == "bandpass":
            raw.filter(l_freq=band[0], h_freq=band[1], verbose=False)
        elif ftype == "lowpass":
            # band = [h_freq] or [None, h_freq]
            h_freq = band[-1]
            raw.filter(l_freq=None, h_freq=h_freq, verbose=False)
        elif ftype == "highpass":
            # band = [l_freq] or [l_freq, None]
            l_freq = band[0]
            raw.filter(l_freq=l_freq, h_freq=None, verbose=False)
        elif ftype == "notch":
            raw.notch_filter(freqs=band, verbose=False)
        else:
            raise ValueError(
                f"unknown filter type {ftype!r}; expected 'bandpass', 'lowpass', "
                "'highpass' or 'notch'"
            )

    resample = preprocess.get("resample")
    if resample and resample.get("enabled"):
        freq = resample.get("freq")
        if freq is None:
            raise ValueError(
                "resample is enabled but no 'freq' was provided in the preprocess "
                "config (expected {'enabled': True, 'freq': <Hz>})"
            )
        raw.resample(sfreq=float(freq), verbose=False)

    return raw


def _epoch_raw(raw, dataset, mne):
    """
    Cut an MNE ``Raw`` into trials, following moabb's own event-extraction logic.

    Returns
    -------
    (X, labels, ch_names) where
        X       : ndarray (n_trials, n_channels, n_times), EEG channels only
        labels  : ndarray (n_trials,) of int class codes (from the dataset event_id)
        ch_names: list[str] of the EEG channel names
    """
    event_id = dataset.event_id

    # moabb datasets carry events either in a stim channel or in annotations. Try the
    # stim channel first (mne.find_events raises ValueError when there is none) and
    # fall back to annotations - the same two sources moabb's own paradigm reads from.
    try:
        events = mne.find_events(raw, shortest_event=0, verbose=False)
    except (ValueError, RuntimeError):
        events, _ = mne.events_from_annotations(
            raw, event_id=event_id, verbose=False
        )

    # tmax is inclusive in mne.Epochs; subtract one sample to match moabb and keep the
    # trial length at exactly (tmax - tmin) * sfreq samples.
    tmin, tmax = dataset.interval[0], dataset.interval[1]
    tmax = tmax - 1.0 / raw.info["sfreq"]

    picks = mne.pick_types(raw.info, eeg=True, stim=False, misc=False, exclude="bads")

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        proj=False,
        baseline=None,
        preload=True,
        picks=picks,
        on_missing="ignore",
        verbose=False,
    )

    X = epochs.get_data()                        # (n_trials, n_channels, n_times)
    ch_names = list(epochs.info["ch_names"])

    # Map raw event codes -> deterministic 0-based class labels (sorted by class name).
    inv_event_id = {code: name for name, code in epochs.event_id.items()}
    ordered_names = sorted(epochs.event_id.keys())
    name_to_label = {name: i for i, name in enumerate(ordered_names)}
    labels = np.array(
        [name_to_label[inv_event_id[code]] for code in epochs.events[:, -1]],
        dtype=np.int64,
    )
    return X, labels, ch_names


# ---------------------------------------------------------------------------------
# Public: build an EEG_Dataset
# ---------------------------------------------------------------------------------
def get_dataset(
    dataset_name: str,
    subjects: Union[int, Sequence[int]],
    preprocess: Optional[dict] = None,
    dataset_type: str = "train",
) -> EEG_Dataset:
    """
    Load a moabb dataset as an :class:`EEG_Dataset`, ready for a ``torch`` DataLoader.

    Parameters
    ----------
    dataset_name : str
        Name of a moabb dataset (case-insensitive), e.g. ``"BNCI2014_001"``.
    subjects : int or sequence of int
        Subject(s) to load. If the requested data is not on disk it is downloaded
        automatically through :func:`download_dataset`.
    preprocess : dict or None, optional
        Preprocessing configuration (filtering + resampling + optional normalization).
        See the module docstring for the full schema. ``None`` means no filtering and
        no resampling.
    dataset_type : {'train', 'test'}, default 'train'
        Which split to return. moabb organises data by *session*; this selects the
        train or test session(s) using the naming heuristic in
        :func:`_select_sessions`.

    Returns
    -------
    EEG_Dataset
        Trials of shape ``[n_trials, 1, n_channels, n_times]`` with integer labels and
        the EEG channel names attached (``.ch_list``).
    """
    dataset_type = str(dataset_type).lower()
    if dataset_type not in ("train", "test"):
        raise ValueError(
            f"dataset_type must be 'train' or 'test', got {dataset_type!r}"
        )

    _, mne = _import_moabb_mne()

    # Downloading is idempotent (moabb caches), so this both ensures the data is
    # present and gives us the dataset object.
    dataset = download_dataset(dataset_name, subjects)
    subject_list = _as_subject_list(subjects, dataset)

    data = dataset.get_data(subjects=subject_list)  # {subj: {session: {run: Raw}}}

    X_all, y_all = [], []
    ch_list = None

    for subject in subject_list:
        sessions = data[subject]
        selected = _select_sessions(list(sessions.keys()), dataset_type)

        for session_name in selected:
            for _run_name, raw in sessions[session_name].items():
                raw = _apply_preprocess(raw, preprocess)
                X, labels, ch_names = _epoch_raw(raw, dataset, mne)
                if X.shape[0] == 0:
                    continue
                X_all.append(X)
                y_all.append(labels)
                if ch_list is None:
                    ch_list = ch_names
                elif ch_list != ch_names:
                    warnings.warn(
                        "Channel names differ across recordings; keeping the first "
                        f"set ({len(ch_list)} channels). This is unusual for a single "
                        "moabb dataset.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

    if not X_all:
        raise RuntimeError(
            f"No trials were extracted for dataset={dataset_name!r}, "
            f"subjects={subject_list}, dataset_type={dataset_type!r}."
        )

    X = np.concatenate(X_all, axis=0).astype(np.float32)   # (trials, channels, time)
    labels = np.concatenate(y_all, axis=0).astype(np.int64)

    # EEG_Dataset expects [trials, 1, channels, time] (the depth-1 axis for Conv2d).
    X = X[:, np.newaxis, :, :]

    normalize = (preprocess or {}).get("normalize", -1)
    return EEG_Dataset(X, labels, ch_list, normalize=normalize)
