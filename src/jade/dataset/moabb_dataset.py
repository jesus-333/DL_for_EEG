"""
Interaction with the `moabb <https://moabb.neurotechx.com/>`_ package.

This module turns the EEG datasets bundled with moabb into ``torch``-ready datasets, i.e. instances of :class:`~jade.dataset.dataset_time.EEG_Dataset`.

It exposes two public functions:

* :func:`~jade.dataset.moabb_dataset.download_dataset` - download the raw recordings of a moabb dataset (all subjects, or a specific subset).
* :func:`~jade.dataset.moabb_dataset.get_dataset` - load a moabb dataset, preprocess it (filtering + resampling through moabb/MNE's own routines), epoch it into trials and wrap it in an :class:`~jade.dataset.dataset_time.EEG_Dataset` (channel names included).
  If the data is not on disk yet it is downloaded automatically via :func:`~jade.dataset.moabb_dataset.download_dataset`.

``moabb`` and ``mne`` are heavy, optional dependencies (declared in the ``dataset`` extra of ``pyproject.toml``).
They are therefore imported lazily, inside the functions that need them, so ``import jade`` works without them installed.
Install them with ``pip install jade[dataset]``.

Configuration
-------------
:func:`~jade.dataset.moabb_dataset.get_dataset` is driven by a :class:`~jade.config.dataset.dataset_config`, which declares the dataset name, the subject(s), the filtering (a nested :class:`~jade.config.filter.filter_config`), the resampling and the normalization::

    from jade.config import dataset_config, filter_config

    config = dataset_config(
        dataset_name = "BNCI2014_001",
        subjects = [1, 3],
        filter = filter_config(filter_type = "bandpass", l_freq = 8, h_freq = 32),
        resample = True,
        resample_freq = 128,
        normalize = -1,
    )
    train_set = get_dataset(config, dataset_type = "train")

The filtering is applied by calling MNE's own routines (``Raw.filter`` for band-/low-/high-pass and band-stop, ``Raw.notch_filter`` for notch) with the parameters carried by the ``filter_config`` - the same routines moabb uses; no custom filter is implemented here.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Full module imports
import warnings

# Specific imports
from typing import Optional, Sequence, Union

# Third party imports
import numpy as np

# Internal imports
from jade.config.dataset import dataset_config
from jade.config.filter import filter_config
from jade.dataset.dataset_time import EEG_Dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Public API

__all__ = ["download_dataset", "get_dataset"]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Lazy dependency import

def _import_moabb_mne() :
    """
    Import ``moabb`` and ``mne`` on demand, with a helpful error if they are missing.

    Returns
    -------
    tuple
        The ``(moabb.datasets module, mne module)`` pair.
    """

    try :
        import mne
        from moabb import datasets as moabb_datasets
    except ImportError as exc :  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "moabb_dataset needs the optional 'dataset' dependencies (moabb + mne). "
            "Install them with:  pip install jade[dataset]  (or 'pip install moabb mne')."
        ) from exc
    return moabb_datasets, mne

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Download directory

def _set_download_dir(dataset_folder : str) -> None :
    """
    Point moabb at a custom directory where the datasets are downloaded and looked up.

    moabb has no download registry of its own: it stores (and searches for) everything under the MNE data directory (``~/mne_data`` by default).
    :func:`moabb.utils.set_download_dir` overrides that path, creating the folder if it does not exist yet, by writing the ``MNE_DATA`` entry of the MNE config.
    Because this is a single global setting, only the *last* directory set is consulted: a dataset downloaded into one folder is not seen by a later run pointing at a different one.

    Parameters
    ----------
    dataset_folder : str
        Filesystem path where the datasets should be stored. It is created if it does not exist.
    """

    try :
        from moabb.utils import set_download_dir
    except ImportError as exc :  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "moabb_dataset needs the optional 'dataset' dependencies (moabb + mne). "
            "Install them with:  pip install jade[dataset]  (or 'pip install moabb mne')."
        ) from exc

    # moabb writes the path into the MNE config (MNE_DATA) and creates the folder if needed.
    set_download_dir(dataset_folder)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dataset name resolution

def _resolve_dataset_class(dataset_name : str) :
    """
    Resolve a (case-insensitive) dataset name to a moabb dataset class.

    moabb exposes dataset classes such as ``BNCI2014_001``, ``Cho2017``, ``PhysionetMI`` under ``moabb.datasets``.
    The lookup is case-insensitive and also ignores underscores, so ``"bnci2014_001"``, ``"BNCI2014001"`` and ``"BNCI2014_001"`` all resolve to the same class.

    Parameters
    ----------
    dataset_name : str
        The (case-insensitive) name of the moabb dataset.

    Returns
    -------
    type
        The moabb dataset class.
    """

    moabb_datasets, _ = _import_moabb_mne()

    def _norm(name : str) -> str :
        return name.lower().replace("_", "")

    target = _norm(dataset_name)

    # Direct attribute hit first (fast path).
    if hasattr(moabb_datasets, dataset_name) :
        return getattr(moabb_datasets, dataset_name)

    # Case-/underscore-insensitive scan over the module's public dataset classes.
    matches = {}
    for attr in dir(moabb_datasets) :
        if attr.startswith("_") :
            continue
        obj = getattr(moabb_datasets, attr)
        if isinstance(obj, type) :  # a class
            matches[_norm(attr)] = obj

    if target in matches :
        return matches[target]

    available = sorted(attr for attr in dir(moabb_datasets)
                       if not attr.startswith("_") and isinstance(getattr(moabb_datasets, attr), type))
    raise ValueError(
        "unknown moabb dataset {}. Available dataset classes under moabb.datasets "
        "include: {}".format(repr(dataset_name), available)
    )


def _as_subject_list(subjects, dataset) -> list :
    """
    Normalize ``subjects`` to a list of subject ids.

    It accepts a single int, a list/tuple of ints, or ``None`` (meaning *all* subjects of the dataset, i.e. ``dataset.subject_list``).

    Parameters
    ----------
    subjects : int, sequence of int or None
        The requested subject(s).
    dataset : moabb dataset instance
        The dataset (used to read ``subject_list`` when ``subjects`` is ``None``).

    Returns
    -------
    list
        The list of subject ids.
    """

    if subjects is None :
        return list(dataset.subject_list)
    if isinstance(subjects, (int, np.integer)) :
        return [int(subjects)]
    if isinstance(subjects, (list, tuple, set, np.ndarray)) :
        return [int(subject) for subject in subjects]
    raise TypeError("subjects must be an int, a list of ints or None, got {}".format(type(subjects)))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Public: download

def download_dataset(
    dataset_name : str,
    subjects : Optional[Union[int, Sequence[int]]] = None,
    dataset_folder : Optional[str] = None,
) :
    """
    Download a moabb dataset (all subjects, or a specific subset).

    moabb caches downloads under its data directory, so calling this repeatedly is cheap: already-present subjects are not re-downloaded.

    Parameters
    ----------
    dataset_name : str
        Name of a moabb dataset (case-insensitive), e.g. ``"BNCI2014_001"``.
    subjects : int, sequence of int or None, optional
        Which subject(s) to download.
        A single int or a list of ints selects those subjects; ``None`` (the default) downloads *every* subject of the dataset.
    dataset_folder : str or None, optional
        Directory where the dataset is stored (and looked up). When ``None`` (the default) moabb's own default location (``~/mne_data``) is used.
        Otherwise the folder is created if needed and used as the download directory through :func:`_set_download_dir`.
        Note that this is a single global setting for moabb, so only the *last* folder set is consulted when checking for already-downloaded data.

    Returns
    -------
    moabb.datasets.base.BaseDataset
        The instantiated dataset object (already pointing at the downloaded files), ready to be passed to ``get_data``.
    """

    # Redirect the download directory before touching the dataset, so moabb both stores into and looks up the requested folder.
    if dataset_folder is not None :
        _set_download_dir(dataset_folder)

    dataset_cls = _resolve_dataset_class(dataset_name)
    dataset = dataset_cls()
    subject_list = _as_subject_list(subjects, dataset)

    # moabb's BaseDataset.download fetches every requested subject; already-cached ones are skipped.
    dataset.download(subject_list = subject_list)
    return dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Train / test session selection

def _session_matches(session_name : str, dataset_type : str) -> bool :
    """
    Heuristic: does ``session_name`` belong to the requested split (train/test)?

    It handles the common moabb session-naming conventions, e.g. BNCI2014_001's ``'0train'`` / ``'1test'`` and the older ``'session_T'`` / ``'session_E'``.

    Parameters
    ----------
    session_name : str
        The moabb session name.
    dataset_type : str
        ``'train'`` or ``'test'``.

    Returns
    -------
    bool
        Whether the session belongs to the requested split.
    """

    low = session_name.lower()
    is_test = ("test" in low) or ("eval" in low) or low.endswith("_e") or low == "e"
    if dataset_type == "test" :
        return is_test

    # train
    is_train = ("train" in low) or low.endswith("_t") or low == "t"
    return is_train


def _select_sessions(session_names : Sequence[str], dataset_type : str) -> list :
    """
    Pick the session keys corresponding to ``dataset_type`` ('train' or 'test').

    Strategy
    --------
    1. If any session name matches the split by the naming heuristic, return those.
    2. Otherwise fall back on position: with a single session it is used for both splits (a warning is emitted); with several sessions the last one is the test split and the rest are the train split.

    Parameters
    ----------
    session_names : sequence of str
        The available session names.
    dataset_type : str
        ``'train'`` or ``'test'``.

    Returns
    -------
    list
        The selected session names.
    """

    names = sorted(session_names)
    matched = [name for name in names if _session_matches(name, dataset_type)]
    if matched :
        return matched

    if len(names) == 1 :
        warnings.warn(
            "Dataset has a single session {}; using it for the {} split as well. "
            "Provide an explicit split upstream if you need train/test isolation.".format(repr(names[0]), repr(dataset_type)),
            RuntimeWarning,
            stacklevel = 3,
        )
        return list(names)

    warnings.warn(
        "Could not map sessions {} to a {} split by name; falling back to positional "
        "split (last session = test, the rest = train).".format(names, repr(dataset_type)),
        RuntimeWarning,
        stacklevel = 3,
    )
    return names[:-1] if dataset_type == "train" else names[-1:]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Preprocessing (filtering + resampling) via MNE, on the raw objects moabb returns

def _apply_filter(raw, config_filter : filter_config) :
    """
    Apply a single :class:`~jade.config.filter.filter_config` to an MNE ``Raw`` object, in place.

    A ``notch`` filter goes through ``Raw.notch_filter``; every other type goes through ``Raw.filter`` (the ``filter_type`` only fixes which cutoffs are used - MNE reads the kind of filter from ``l_freq``/``h_freq``: ``l_freq < h_freq`` is a band-pass, ``l_freq > h_freq`` a band-stop, one of them ``None`` a low-/high-pass).
    These are MNE's own routines, exactly as moabb uses them; no custom filter is implemented.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw recording to filter (modified in place).
    config_filter : filter_config
        The filter parameters.

    Returns
    -------
    mne.io.Raw
        The same (filtered) raw object.
    """

    filter_type = config_filter.filter_type

    if filter_type == "notch" :
        raw.notch_filter(
            freqs = config_filter.freqs,
            picks = config_filter.picks,
            method = config_filter.method,
            iir_params = config_filter.iir_params,
            filter_length = config_filter.filter_length,
            phase = config_filter.phase,
            fir_window = config_filter.fir_window,
            fir_design = config_filter.fir_design,
            verbose = False,
        )
    else :
        raw.filter(
            l_freq = config_filter.l_freq,
            h_freq = config_filter.h_freq,
            picks = config_filter.picks,
            method = config_filter.method,
            iir_params = config_filter.iir_params,
            filter_length = config_filter.filter_length,
            l_trans_bandwidth = config_filter.l_trans_bandwidth,
            h_trans_bandwidth = config_filter.h_trans_bandwidth,
            phase = config_filter.phase,
            fir_window = config_filter.fir_window,
            fir_design = config_filter.fir_design,
            verbose = False,
        )
    return raw


def _apply_preprocess(raw, config : dataset_config) :
    """
    Apply the config's filtering then resampling to an MNE ``Raw`` object, in place.

    Resampling is applied last, so it acts on the already band-limited signal.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw recording to preprocess (modified in place).
    config : dataset_config
        The dataset configuration (its filter + resampling fields are used).

    Returns
    -------
    mne.io.Raw
        The same (preprocessed) raw object.
    """

    # Filtering / resampling both require the data in memory.
    raw.load_data(verbose = False)

    if config.filter is not None :
        _apply_filter(raw, config.filter)

    if config.resample :
        raw.resample(sfreq = float(config.resample_freq), verbose = False)

    return raw


def _epoch_raw(raw, dataset, mne) :
    """
    Cut an MNE ``Raw`` into trials, following moabb's own event-extraction logic.

    Parameters
    ----------
    raw : mne.io.Raw
        The (preprocessed) raw recording.
    dataset : moabb dataset instance
        Provides ``event_id`` and ``interval``.
    mne : module
        The imported ``mne`` module.

    Returns
    -------
    X : numpy.ndarray
        Trials of shape (n_trials, n_channels, n_times), EEG channels only.
    labels : numpy.ndarray
        Integer class codes of shape (n_trials,).
    ch_names : list of str
        The EEG channel names.
    """

    event_id = dataset.event_id

    # moabb datasets carry events either in a stim channel or in annotations.
    # Try the stim channel first (mne.find_events raises ValueError when there is none) and fall back to annotations - the same two sources moabb's own paradigm reads from.
    try :
        events = mne.find_events(raw, shortest_event = 0, verbose = False)
    except (ValueError, RuntimeError) :
        events, _ = mne.events_from_annotations(raw, event_id = event_id, verbose = False)

    # tmax is inclusive in mne.Epochs; subtract one sample to match moabb and keep the trial length at exactly (tmax - tmin) * sfreq samples.
    tmin, tmax = dataset.interval[0], dataset.interval[1]
    tmax = tmax - 1.0 / raw.info["sfreq"]

    picks = mne.pick_types(raw.info, eeg = True, stim = False, misc = False, exclude = "bads")

    epochs = mne.Epochs(
        raw,
        events,
        event_id = event_id,
        tmin = tmin,
        tmax = tmax,
        proj = False,
        baseline = None,
        preload = True,
        picks = picks,
        on_missing = "ignore",
        verbose = False,
    )

    X = epochs.get_data()                        # (n_trials, n_channels, n_times)
    ch_names = list(epochs.info["ch_names"])

    # Map raw event codes -> deterministic 0-based class labels (sorted by class name).
    inv_event_id = {code : name for name, code in epochs.event_id.items()}
    ordered_names = sorted(epochs.event_id.keys())
    name_to_label = {name : index for index, name in enumerate(ordered_names)}
    labels = np.array(
        [name_to_label[inv_event_id[code]] for code in epochs.events[:, -1]],
        dtype = np.int64,
    )
    return X, labels, ch_names

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Public: build an EEG_Dataset

def get_dataset(
    config : Union[dataset_config, dict],
    dataset_type : str = "train",
) -> EEG_Dataset :
    """
    Load a moabb dataset as an :class:`~jade.dataset.dataset_time.EEG_Dataset`, ready for a ``torch`` DataLoader.

    Parameters
    ----------
    config : dataset_config or dict
        The dataset configuration (see :class:`~jade.config.dataset.dataset_config`): the dataset name, the subject(s), the filtering (nested ``filter_config``), the resampling and the normalization.
        A plain dict is accepted for convenience and promoted via ``dataset_config.from_dict``.
        If the requested data is not on disk it is downloaded automatically through :func:`~jade.dataset.moabb_dataset.download_dataset`.
    dataset_type : {'train', 'test'}, default 'train'
        Which split to return.
        moabb organizes data by *session*; this selects the train or test session(s) using the naming heuristic in :func:`_select_sessions`.

    Returns
    -------
    EEG_Dataset
        Trials of shape ``[n_trials, 1, n_channels, n_times]`` with integer labels and the EEG channel names attached (``.ch_list``).
    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Input check

    if isinstance(config, dict) :
        config = dataset_config.from_dict(config)
    elif not isinstance(config, dataset_config) :
        raise TypeError("config must be a dataset_config (or a dict), got {}".format(type(config)))

    dataset_type = str(dataset_type).lower()
    if dataset_type not in ("train", "test") :
        raise ValueError("dataset_type must be 'train' or 'test', got {}".format(repr(dataset_type)))

    _, mne = _import_moabb_mne()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Download (idempotent) and load the raw data

    # Downloading is idempotent (moabb caches), so this both ensures the data is present and gives us the dataset object.
    dataset = download_dataset(config.dataset_name, config.subjects)
    subject_list = _as_subject_list(config.subjects, dataset)

    data = dataset.get_data(subjects = subject_list)  # {subj: {session: {run: Raw}}}

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Preprocess + epoch every selected recording

    X_all, y_all = [], []
    ch_list = None

    for subject in subject_list :
        sessions = data[subject]
        selected = _select_sessions(list(sessions.keys()), dataset_type)

        for session_name in selected :
            for _run_name, raw in sessions[session_name].items() :
                raw = _apply_preprocess(raw, config)
                X, labels, ch_names = _epoch_raw(raw, dataset, mne)
                if X.shape[0] == 0 :
                    continue
                X_all.append(X)
                y_all.append(labels)
                if ch_list is None :
                    ch_list = ch_names
                elif ch_list != ch_names :
                    warnings.warn(
                        "Channel names differ across recordings; keeping the first set "
                        "({} channels). This is unusual for a single moabb dataset.".format(len(ch_list)),
                        RuntimeWarning,
                        stacklevel = 2,
                    )

    if not X_all :
        raise RuntimeError(
            "No trials were extracted for dataset={}, subjects={}, dataset_type={}.".format(
                repr(config.dataset_name), subject_list, repr(dataset_type)
            )
        )

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Assemble the EEG_Dataset

    X = np.concatenate(X_all, axis = 0).astype(np.float32)   # (trials, channels, time)
    labels = np.concatenate(y_all, axis = 0).astype(np.int64)

    # EEG_Dataset expects [trials, 1, channels, time] (the depth-1 axis for Conv2d).
    X = X[:, np.newaxis, :, :]

    return EEG_Dataset(X, labels, ch_list, normalize = config.normalize)
