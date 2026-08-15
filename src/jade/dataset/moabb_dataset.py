"""
Interaction with the `moabb <https://moabb.neurotechx.com/>`_ package.

This module turns the EEG datasets bundled with moabb into ``torch``-ready datasets, i.e. instances of :class:`~jade.dataset.dataset_time.EEG_Dataset`.

It exposes two public functions:

* :func:`~jade.dataset.moabb_dataset.download_dataset` - download the raw recordings of a moabb dataset (all subjects, or a specific subset).
* :func:`~jade.dataset.moabb_dataset.get_dataset` - load a moabb dataset, preprocess it (filtering + resampling through moabb/MNE's own routines), epoch it into trials and wrap it in an :class:`~jade.dataset.dataset_time.EEG_Dataset` (channel names and sampling frequency included).
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
        use_original_split = True,
        split_train_test = 0.5,
        seed = 42,
    )
    train_set, test_set = get_dataset(config)

The filtering is optional: when the ``dataset_config`` leaves it unspecified (``filter = None``, the default) no filter is applied and the raw signal is kept.
When it is set, the filtering is applied by calling MNE's own routines (``Raw.filter`` for band-/low-/high-pass and band-stop, ``Raw.notch_filter`` for notch) with the parameters carried by the ``filter_config`` - the same routines moabb uses; no custom filter is implemented here.

Train / test split
-------------------
:func:`get_dataset` returns **both** splits in a single call, as a ``(train_dataset, test_dataset)`` tuple. Two strategies decide which trials go where :

* **Original (session-name) split** - used when ``config.use_original_split`` is ``True`` and the moabb session names mark a train/test division (e.g. BNCI2014_001's ``'0train'`` / ``'1test'``). The trials of the ``train``-named sessions become the training set and those of the ``test``-named sessions the test set.
* **Random split** - used when ``config.use_original_split`` is ``False``, or when it is ``True`` but the session names carry no division (a single session, or names with no train/test marker). All the sessions are pooled together and split by :func:`create_train_test_split_indices` : a fraction ``config.split_train_test`` of the trials goes to training and the rest to test.
  Because both sets come from a single partition of the pooled trials, they are always complementary (disjoint, together covering every trial); ``config.seed`` only fixes *which* trials land in each set, making the split reproducible across runs.
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

__all__ = ["download_dataset", "get_dataset", "create_train_test_split_indices"]

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

    if target in matches : return matches[target]

    available = sorted(attr for attr in dir(moabb_datasets)
                       if not attr.startswith("_") and isinstance(getattr(moabb_datasets, attr), type))

    error_message = f"unknown moabb dataset. Available dataset classes under moabb.datasets include: {repr(dataset_name), available}"
    raise ValueError(error_message)

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
    dataset_name   : str,
    subjects       : Optional[Union[int, Sequence[int]]] = None,
    dataset_folder : Optional[str]                       = None,
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
    if dataset_folder is not None : _set_download_dir(dataset_folder)

    dataset_cls  = _resolve_dataset_class(dataset_name)
    dataset      = dataset_cls()
    subject_list = _as_subject_list(subjects, dataset) # Not that if the subjects are not specified this function will set the subjects list to all the subjects of the dataset

    # moabb's BaseDataset.download fetches every requested subject; already-cached ones are skipped.
    dataset.download(subject_list = subject_list)

    return dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Train / test split (session names + random)

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
    
    # Change session name to lower case
    session_name = session_name.lower()
    
    # Check if is the test set
    is_test = ("test" in session_name) or ("eval" in session_name) or session_name.endswith("_e") or session_name == "e"
    if dataset_type == "test" : return is_test

    # Check if is the train set
    is_train = ("train" in session_name) or session_name.endswith("_t") or session_name == "t"
    return is_train

def _original_split_available(session_names : Sequence[str]) -> bool :
    """
    Whether the session names carry an explicit train/test division (the "original" split).

    The original split is considered available only when at least one session matches the **train** split by name **and** at least one session matches the **test** split, so that both a non-empty train and a non-empty test set can be built from the session names alone (e.g. BNCI2014_001's ``'0train'`` / ``'1test'``).
    A dataset with a single session, or with sessions whose names carry no train/test marker, has no original split : in that case :func:`~jade.dataset.moabb_dataset.get_dataset` falls back to the random split.

    Parameters
    ----------
    session_names : sequence of str
        The available session names.

    Returns
    -------
    bool
        ``True`` when the names mark both a train and a test session, ``False`` otherwise.
    """

    # An original split needs both a train-marked and a test-marked session, otherwise one of the two splits would be empty.
    has_train = any(_session_matches(name, "train") for name in session_names)
    has_test  = any(_session_matches(name, "test") for name in session_names)

    return has_train and has_test


def create_train_test_split_indices(
    n_samples        : int,
    split_train_test : float             = 0.5,
    seed             : Optional[int]     = None,
) -> tuple :
    """
    Build the shuffled train/test index arrays for a random split of ``n_samples`` samples.

    All the samples (trials) of every session are considered together as a single pool of ``n_samples`` elements.
    An index array ``[0, 1, ..., n_samples - 1]`` is created, shuffled with a seeded random generator (for reproducibility) and split in two contiguous parts :

    * the first part - a fraction ``split_train_test`` of the samples - holds the **training** indices,
    * the second part - the remaining ``1 - split_train_test`` fraction - holds the **test** indices.

    Because the partition depends only on ``n_samples``, ``split_train_test`` and ``seed``, the same three values always reproduce the very same partition.
    :func:`~jade.dataset.moabb_dataset.get_dataset` calls this once on the pooled trials and keeps ``train_indices`` for the training set and ``test_indices`` for the test set, so the two returned sets are complementary (disjoint and together covering every trial) by construction; ``config.seed`` only makes the partition reproducible across runs.

    Parameters
    ----------
    n_samples : int
        Total number of samples (trials) to split. Must be a non-negative int.
    split_train_test : float, default 0.5
        Fraction of the samples assigned to the **training** split (the remaining ``1 - split_train_test`` goes to test). Must be between 0 and 1.
        The training split gets ``round(split_train_test * n_samples)`` samples.
    seed : int or None, default None
        Seed of the random generator used to shuffle the indices.
        A concrete int makes the split reproducible; ``None`` draws a fresh (non-reproducible) shuffle from the OS entropy.

    Returns
    -------
    train_indices : numpy.ndarray
        1-D int array with the indices of the training samples.
    test_indices : numpy.ndarray
        1-D int array with the indices of the test samples.
        Together with ``train_indices`` it forms a partition of ``range(n_samples)`` : the two arrays are disjoint and cover every index exactly once.
    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Input check

    # n_samples must be a non-negative int (bool is rejected first, being a subclass of int).
    if not isinstance(n_samples, (int, np.integer)) or isinstance(n_samples, bool) or n_samples < 0 :
        raise ValueError(f"n_samples must be a non-negative int, got {repr(n_samples)}")

    # split_train_test is a fraction, so it must lie in [0, 1].
    if not (0.0 <= float(split_train_test) <= 1.0) :
        raise ValueError(f"split_train_test must be between 0 and 1, got {repr(split_train_test)}")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Build and split the (shuffled) index array

    # Seeded generator : same seed -> same shuffle -> reproducible (and complementary) train/test split.
    random_generator = np.random.default_rng(seed)

    # Indices of all the samples, then shuffled in place.
    indices = np.arange(int(n_samples))
    random_generator.shuffle(indices)

    # Size of the training split (the rest of the samples form the test split).
    n_train = int(round(float(split_train_test) * int(n_samples)))

    # The first n_train shuffled indices are the training samples, the remaining ones are the test samples.
    train_indices = indices[:n_train]
    test_indices  = indices[n_train:]

    return train_indices, test_indices

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
        event_id   = event_id,
        tmin       = tmin,
        tmax       = tmax,
        proj       = False,
        baseline   = None,
        preload    = True,
        picks      = picks,
        on_missing = "ignore",
        verbose    = False,
    )

    # Get data [n_trials x n_channels x n_times]
    x = epochs.get_data()

    # Get channels name
    ch_names = list(epochs.info["ch_names"])

    # Map raw event codes -> deterministic 0-based class labels (sorted by class name).
    inv_event_id = {code : name for name, code in epochs.event_id.items()}
    ordered_names = sorted(epochs.event_id.keys())
    name_to_label = {name : index for index, name in enumerate(ordered_names)}
    labels = np.array(
        [name_to_label[inv_event_id[code]] for code in epochs.events[:, -1]],
        dtype = np.int64,
    )
    return x, labels, ch_names


def _collect_trials(data, subject_list, session_predicate, dataset, mne, config) :
    """
    Preprocess, epoch and concatenate the trials of the sessions passing ``session_predicate``.

    For every subject the sessions are visited in sorted order and, when ``session_predicate(session_name)`` is true, each of their runs (also in sorted order) is preprocessed (:func:`_apply_preprocess`) and epoched (:func:`_epoch_raw`); the resulting trials are concatenated.
    Iterating sessions and runs in a deterministic (sorted) order keeps the concatenation order stable across runs, which is what makes the downstream random split reproducible.

    Parameters
    ----------
    data : dict
        The nested ``{subject: {session: {run: mne.Raw}}}`` mapping returned by ``dataset.get_data``.
    subject_list : list of int
        The subjects to visit.
    session_predicate : callable
        A predicate ``session_name -> bool`` selecting which sessions contribute their trials.
    dataset : moabb dataset instance
        Passed to :func:`_epoch_raw` (provides ``event_id`` and ``interval``).
    mne : module
        The imported ``mne`` module.
    config : dataset_config
        The dataset configuration (its preprocessing fields drive :func:`_apply_preprocess`).

    Returns
    -------
    x : numpy.ndarray or None
        Trials of shape ``(n_trials, n_channels, n_times)`` (``float32``), or ``None`` when no trial matched the predicate.
    labels : numpy.ndarray or None
        Integer class labels of shape ``(n_trials,)`` (``int64``), or ``None`` when ``x`` is ``None``.
    ch_list : list of str or None
        The EEG channel names of the first recording (``None`` when no trial matched).
    sampling_freq : float or None
        The original sampling frequency captured from the first recording, before any resampling (``None`` when no recording was visited).
    """

    x_all, y_all = [], []
    ch_list = None

    # Sampling frequency of the recordings, read from the raw moabb data (before any resampling).
    # moabb/MNE store it in raw.info["sfreq"] and it is the same across the runs of a dataset, so the first recording is enough.
    sampling_freq = None

    for subject in subject_list :
        sessions = data[subject]

        for session_name in sorted(sessions.keys()) :
            # Skip the sessions that do not belong to the requested selection.
            if not session_predicate(session_name) :
                continue

            # Runs are iterated in a deterministic (sorted) order, so the concatenation order is stable across runs (required for a reproducible random split).
            for run_name in sorted(sessions[session_name].keys()) :
                raw = sessions[session_name][run_name]

                # Capture the original sampling frequency before preprocessing (resampling would change it).
                if sampling_freq is None :
                    sampling_freq = float(raw.info["sfreq"])

                raw = _apply_preprocess(raw, config)
                x, labels, ch_names = _epoch_raw(raw, dataset, mne)
                if x.shape[0] == 0 :
                    continue
                x_all.append(x)
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

    # No recording matched the predicate (or every matching recording was empty).
    if not x_all :
        return None, None, ch_list, sampling_freq

    x = np.concatenate(x_all, axis = 0).astype(np.float32)   # (trials, channels, time)
    labels = np.concatenate(y_all, axis = 0).astype(np.int64)

    return x, labels, ch_list, sampling_freq

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Public: build an EEG_Dataset

def get_dataset(config : Union[dataset_config, dict]) -> tuple :
    """
    Load a moabb dataset as a ``(train_dataset, test_dataset)`` pair of :class:`~jade.dataset.dataset_time.EEG_Dataset`, ready for a ``torch`` DataLoader.

    Both splits are built in a **single call** (there is no ``dataset_type`` argument): the data is downloaded and epoched once, then divided into a training and a test :class:`~jade.dataset.dataset_time.EEG_Dataset`.

    Parameters
    ----------
    config : dataset_config or dict
        The dataset configuration (see :class:`~jade.config.dataset.dataset_config`): the dataset name, the subject(s), the optional filtering (nested ``filter_config``; ``None`` means no filtering), the resampling, the normalization, the train/test split options (``use_original_split`` / ``split_train_test`` / ``seed``) and the download folder.
        A plain dict is accepted for convenience and promoted via ``dataset_config.from_dict``.
        If the requested data is not on disk it is downloaded automatically through :func:`~jade.dataset.moabb_dataset.download_dataset` (into ``config.dataset_folder`` when set, otherwise moabb's default location).

    Returns
    -------
    train_dataset : EEG_Dataset
        The training split.
    test_dataset : EEG_Dataset
        The test split.

    Both are :class:`~jade.dataset.dataset_time.EEG_Dataset` with trials of shape ``[n_trials, 1, n_channels, n_times]``, integer labels, the EEG channel names attached (``.ch_list``) and the sampling frequency attached (``.fs`` - the one read from the raw moabb recordings, or ``config.resample_freq`` when resampling is enabled).

    How the two splits are built depends on the config :

    * ``config.use_original_split = True`` **and** the moabb session names mark a train/test division (checked by :func:`_original_split_available`) : the trials of the ``train``-named sessions form ``train_dataset`` and those of the ``test``-named sessions form ``test_dataset`` (the "original" moabb split, e.g. BNCI2014_001's ``'0train'`` / ``'1test'``).
    * otherwise (``use_original_split = False``, or ``True`` but the session names carry no division - a single session, or names with no train/test marker) : every session is pooled together and partitioned by :func:`create_train_test_split_indices`, using ``config.split_train_test`` (the training fraction) and ``config.seed``. The two sets are complementary (disjoint, together covering all the trials) by construction; ``config.seed`` only makes the partition reproducible across runs.

    Raises
    ------
    RuntimeError
        If no trials can be extracted for a split (e.g. an empty dataset, or a degenerate ``split_train_test`` of 0 or 1 that leaves the training or the test set empty).
    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Input check

    # Check if config is a dictionary.
    if isinstance(config, dict) :
        # In this case convert it to dataset_config class
        config = dataset_config.from_dict(config)
    elif not isinstance(config, dataset_config) :
        # Otherwise raise an error
        raise TypeError(f"config must be a dataset_config (or a dict), got {type(config)}")

    _, mne = _import_moabb_mne()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Download (idempotent) and load the raw data

    # Downloading is idempotent (moabb caches), so this both ensures the data is present and gives us the dataset object.
    # config.dataset_folder (when set) redirects where moabb stores and looks up the data.
    dataset = download_dataset(config.dataset_name, config.subjects, config.dataset_folder)
    subject_list = _as_subject_list(config.subjects, dataset)

    data = dataset.get_data(subjects = subject_list)  # {subj: {session: {run: Raw}}}

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Decide the train / test split strategy

    # Collect the session names of every subject (normally identical across the subjects of a moabb dataset).
    all_session_names = sorted({session_name for subject in subject_list for session_name in data[subject].keys()})

    # Use the original (session-name based) split only when the user asked for it AND the session names actually
    # mark a train/test division. Otherwise fall back to the configurable random split (this also covers the
    # single-session case, where the old positional fallback returned identical train and test sets).
    use_name_split = bool(config.use_original_split) and _original_split_available(all_session_names)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Build the train and the test trials

    if use_name_split :
        # Original split : the train-named sessions form the training set, the test-named ones the test set.
        # The two predicates select disjoint sessions, so each recording is still preprocessed exactly once.
        x_train, y_train, ch_list, sampling_freq = _collect_trials(
            data, subject_list, lambda session_name : _session_matches(session_name, "train"), dataset, mne, config
        )
        x_test, y_test, _, _ = _collect_trials(
            data, subject_list, lambda session_name : _session_matches(session_name, "test"), dataset, mne, config
        )

        # With a name-based split available both sets should be non-empty, but a session could still epoch to zero trials.
        if x_train is None :
            raise RuntimeError(f"No training trials were extracted for dataset={repr(config.dataset_name)}, subjects={subject_list}.")
        if x_test is None :
            raise RuntimeError(f"No test trials were extracted for dataset={repr(config.dataset_name)}, subjects={subject_list}.")
    else :
        # Random split : pool every session once, then partition the trials by index.
        x_all, y_all, ch_list, sampling_freq = _collect_trials(
            data, subject_list, lambda session_name : True, dataset, mne, config
        )

        if x_all is None : raise RuntimeError(f"No trials were extracted for dataset={repr(config.dataset_name)}, subjects={subject_list}.")

        # Reproducible partition of the pooled trials into complementary train / test index sets.
        train_indices, test_indices = create_train_test_split_indices(
            n_samples        = x_all.shape[0],
            split_train_test = config.split_train_test,
            seed             = config.seed,
        )

        # A degenerate split_train_test (0 or 1), or too few trials, can leave one of the two splits empty.
        if train_indices.size == 0 or test_indices.size == 0 :
            raise RuntimeError(
                "The random split leaves one side empty (dataset={}, {} trials in total, split_train_test={}). "
                "Adjust split_train_test or provide more data.".format(
                    repr(config.dataset_name), x_all.shape[0], repr(config.split_train_test)
                )
            )

        x_train, y_train = x_all[train_indices], y_all[train_indices]
        x_test , y_test  = x_all[test_indices] , y_all[test_indices]

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Assemble the two EEG_Dataset

    # EEG_Dataset expects [trials, 1, channels, time] (the depth-1 axis for Conv2d).
    x_train = x_train[:, np.newaxis, :, :]
    x_test = x_test[:, np.newaxis, :, :]

    # The sampling frequency stored in the datasets must match the returned signal.
    # If the data was resampled, the user-provided resample_freq replaces the original moabb sampling frequency.
    fs = float(config.resample_freq) if config.resample else sampling_freq

    train_dataset = EEG_Dataset(x_train, y_train, ch_list, normalize = config.normalize, fs = fs)
    test_dataset  = EEG_Dataset(x_test , y_test , ch_list, normalize = config.normalize, fs = fs)

    return train_dataset, test_dataset
