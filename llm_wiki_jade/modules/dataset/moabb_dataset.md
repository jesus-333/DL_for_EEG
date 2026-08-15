# `jade.dataset.moabb_dataset`

Source: `src/jade/dataset/moabb_dataset.py`

Turns the EEG datasets bundled with [`moabb`](https://moabb.neurotechx.com/) into `torch`-ready [`EEG_Dataset`](./dataset_time.md) objects: download → preprocess (optional MNE filter + optional resample) → epoch → wrap. Three public functions (`download_dataset`, `get_dataset`, `create_train_test_split_indices`) plus internal helpers. `moabb`/`mne` are optional (the `dataset` extra) and imported lazily inside the functions that need them.

## Public API
```python
__all__ = ["download_dataset", "get_dataset", "create_train_test_split_indices"]
```
All three are re-exported lazily from the package via `__init__.__getattr__` (see [`__init__.md`](./__init__.md)). Note that `create_train_test_split_indices` needs only `numpy`, but touching any of the three still imports the `moabb_dataset` module (which does not itself import `moabb`/`mne` — those are deferred to `_import_moabb_mne()`).

## Imports & configuration
- Internal: `dataset_config` (`jade.config.dataset`), `filter_config` (`jade.config.filter`), `EEG_Dataset` (`jade.dataset.dataset_time`).
- Third-party eager: `numpy`, `warnings`.
- `moabb`/`mne`: imported **only** through `_import_moabb_mne()`.

`get_dataset` is driven by a `dataset_config`, which declares the dataset name, subject(s), the **optional** filtering (a nested `filter_config`; `None` = no filtering, the raw signal is kept), resampling, and normalization. No custom DSP is implemented here — filtering/resampling/epoching all call MNE's own routines (the same ones moabb uses).

---

## Internal: lazy dependency import

### `_import_moabb_mne() -> (moabb.datasets, mne)`
Imports `mne` and `moabb.datasets` on demand. On `ImportError` re-raises a helpful `ImportError` telling the user to `pip install jade[dataset]` (or `pip install moabb mne`). This is the second laziness layer (the first being `__init__.__getattr__`): even after `moabb_dataset` is imported, `moabb`/`mne` are not loaded until a function actually calls this.

---

## Internal: download directory

### `_set_download_dir(dataset_folder : str) -> None`
Points moabb (and the MNE backend it relies on) at a custom download directory, so datasets are both stored into and looked up under `dataset_folder`. Lazily imports `moabb.utils.set_download_dir` (same helpful `ImportError` as `_import_moabb_mne` if the extra is missing), which creates the folder if needed and writes the `MNE_DATA` entry of the MNE config.

**Gotcha (single global setting):** moabb has no download registry of its own — this overrides *the* MNE data path, so only the **last** folder set is consulted. A dataset downloaded into one folder is invisible to a later run pointing at a different one.

---

## Internal: dataset-name resolution

### `_resolve_dataset_class(dataset_name : str) -> type`
Resolves a (case- and underscore-insensitive) name to a moabb dataset **class** under `moabb.datasets`.
- Fast path: direct `hasattr(moabb_datasets, dataset_name)`.
- Otherwise scans public classes, normalizing names with `name.lower().replace("_", "")`, so `"bnci2014_001"`, `"BNCI2014001"`, `"BNCI2014_001"` all match the same class.
- **Raises `ValueError`** (listing available dataset classes) if nothing matches.

### `_as_subject_list(subjects, dataset) -> list`
Normalizes the `subjects` argument to a list of ints:
| `subjects` | result |
|------------|--------|
| `None` | `list(dataset.subject_list)` (all subjects) |
| `int` / `np.integer` | `[int(subjects)]` |
| list / tuple / set / `np.ndarray` | `[int(s) for s in subjects]` |
| anything else | raises `TypeError` |

---

## Public: `download_dataset(dataset_name, subjects=None, dataset_folder=None)`

Download a moabb dataset (all subjects, or a subset).

| Param | Type | Meaning | Default |
|-------|------|---------|---------|
| `dataset_name` | str | moabb dataset name (case-insensitive), e.g. `"BNCI2014_001"`. | required |
| `subjects` | int, sequence of int, or None | which subjects to download; `None` → every subject. | `None` |
| `dataset_folder` | str or None | directory to store/look up the data; `None` → moabb's default (`~/mne_data`). Non-`None` routes through `_set_download_dir` (folder created if needed). | `None` |

**Returns** the instantiated moabb dataset object (a `moabb.datasets.base.BaseDataset`), already pointing at the downloaded files and ready for `get_data`.

**Side effect / disk:** if `dataset_folder` is given, `_set_download_dir(dataset_folder)` is called **first** (so moabb both stores into and searches the folder); then `dataset.download(subject_list=...)` fetches the raw recordings into that directory. **Idempotent** — moabb skips already-cached subjects, so repeated calls are cheap. Mind the single-global-folder gotcha noted under `_set_download_dir`.

Internally: (optional) `_set_download_dir(dataset_folder)`, `_resolve_dataset_class(name)()` → instance, `_as_subject_list(subjects, dataset)` → id list, then `dataset.download(...)`.

This is the function the `download-dataset-moabb` cli command wraps (see [`modules/cli/download_dataset.md`](../cli/download_dataset.md)).

---

## Public / internal: train / test split (session names + random)

moabb organizes data by *session*; `get_dataset` must decide which trials are "train" and which are "test". Two strategies exist (see the `get_dataset` pipeline below): the **original** split reads the moabb session names; the **random** split pools every session together and partitions the trials by index. The config field `use_original_split` chooses the preferred strategy, with a graceful fall-back to the random split whenever the session names carry no train/test division.

### `_session_matches(session_name, dataset_type) -> bool` (internal)
Heuristic on the lowercased session name:
- **test** if it contains `"test"`/`"eval"`, or ends with `"_e"`, or equals `"e"`.
- **train** if it contains `"train"`, or ends with `"_t"`, or equals `"t"`.

Handles common moabb conventions like BNCI2014_001's `'0train'`/`'1test'` and the older `'session_T'`/`'session_E'`.

### `_original_split_available(session_names) -> bool` (internal)
Whether the session names carry an explicit train/test division. Returns `True` **only** when at least one session matches the **train** split by name **and** at least one matches the **test** split (so both a non-empty train and a non-empty test set can be built from the names alone). A single session, or names with no train/test marker (or only-train / only-test names), give `False` → `get_dataset` falls back to the random split.

### `create_train_test_split_indices(n_samples, split_train_test=0.5, seed=None) -> (train_indices, test_indices)` (public)
The ad-hoc function that builds the **random** split. It creates the index array `[0, 1, …, n_samples-1]`, shuffles it with a **seeded** `numpy` generator (`np.random.default_rng(seed)`), then cuts it in two contiguous parts:
- `train_indices` = the first `round(split_train_test * n_samples)` shuffled indices,
- `test_indices` = the remaining ones.

Both returned arrays are `numpy` int arrays; together they form a **partition** of `range(n_samples)` (disjoint, covering every index once). `get_dataset` calls this once on the pooled trials and keeps `train_indices` for its training set and `test_indices` for its test set — so the two `EEG_Dataset` it returns are **complementary by construction**. Because the result depends only on `(n_samples, split_train_test, seed)`, the same triple reproduces the identical partition, so `config.seed` makes the split reproducible across runs.

| Param | Type | Meaning | Default |
|-------|------|---------|---------|
| `n_samples` | int | total number of trials to split; must be a non-negative int. | required |
| `split_train_test` | float | fraction of samples assigned to **training** (rest → test); must be in `[0, 1]`. | `0.5` |
| `seed` | int or None | shuffle seed; a concrete int → reproducible, `None` → fresh (non-reproducible) shuffle. | `None` |

**Raises `ValueError`** for a negative/non-int `n_samples` (bool rejected) or a `split_train_test` outside `[0, 1]`. Uses only `numpy` (no moabb/mne).

---

## Internal: preprocessing (MNE, in place on the `Raw` objects)

### `_apply_filter(raw, config_filter : filter_config) -> raw`
Applies one filter to an MNE `Raw`, **in place**.
- `filter_type == "notch"` → `raw.notch_filter(freqs, picks, method, iir_params, filter_length, phase, fir_window, fir_design, verbose=False)`.
- Any other type → `raw.filter(l_freq, h_freq, picks, method, iir_params, filter_length, l_trans_bandwidth, h_trans_bandwidth, phase, fir_window, fir_design, verbose=False)`.

Note: the actual filter *kind* is inferred by MNE from `l_freq`/`h_freq` (band-pass if `l_freq < h_freq`, band-stop if `l_freq > h_freq`, low-/high-pass if one is `None`); `filter_type` only fixes which cutoffs the `filter_config` requires. No custom filter is implemented — these are MNE's routines, as moabb uses them.

### `_apply_preprocess(raw, config : dataset_config) -> raw`
Full preprocessing of one `Raw`, **in place**:
1. `raw.load_data(verbose=False)` (filtering/resampling need data in memory).
2. If `config.filter is not None` → `_apply_filter`. When no filter is set (`config.filter is None`, the default) this step is skipped entirely and the raw signal is kept unfiltered.
3. If `config.resample` → `raw.resample(sfreq=float(config.resample_freq), verbose=False)`.

Both steps are optional and independent: an unspecified filter leaves the signal unfiltered; resampling runs **last**, so when a filter *is* set it acts on the already band-limited signal.

### `_epoch_raw(raw, dataset, mne) -> (X, labels, ch_names)`
Cuts a preprocessed `Raw` into trials, mirroring moabb's event logic.
- **Events:** tries `mne.find_events(raw, shortest_event=0)` (stim channel) first; on `ValueError`/`RuntimeError` falls back to `mne.events_from_annotations(raw, event_id=dataset.event_id)`.
- **Interval:** `tmin, tmax = dataset.interval`; then `tmax -= 1/sfreq` (because `mne.Epochs` treats `tmax` as inclusive — this keeps exactly `(tmax - tmin) * sfreq` samples, matching moabb).
- **Picks:** `mne.pick_types(..., eeg=True, stim=False, misc=False, exclude="bads")` → EEG channels only.
- Builds `mne.Epochs(..., baseline=None, preload=True, on_missing="ignore")`.
- **Labels:** maps raw event codes → **0-based class ids sorted by class name** (`ordered_names = sorted(epochs.event_id.keys())`), giving deterministic, reproducible labels as `np.int64`.

**Returns:** `X` shape `(n_trials, n_channels, n_times)`, `labels` shape `(n_trials,)`, `ch_names` list of EEG channel names.

### `_collect_trials(data, subject_list, session_predicate, dataset, mne, config) -> (X, labels, ch_list, sampling_freq)` (internal)
Shared collection loop used by `get_dataset`. For every subject it visits the sessions in **sorted** order and, when `session_predicate(session_name)` is true, preprocesses + epochs each of their runs (also sorted) and concatenates the trials. Captures the original `sampling_freq` from the first recording and tracks `ch_list` (same channel-mismatch `RuntimeWarning` as before). Returns `(X, labels, ch_list, sampling_freq)` with `X` shape `(n_trials, n_channels, n_times)` (`float32`) and `labels` `int64`; `X`/`labels` are `None` when no session matched (or every match epoched to zero trials). The sorted iteration keeps the concatenation order deterministic, which is what makes the random split reproducible.

---

## Public: `get_dataset(config) -> (train_dataset, test_dataset)`

The one-call pipeline: config → **both** ready `EEG_Dataset` splits. There is no `dataset_type` argument — the data is downloaded and epoched once, then divided into a training and a test dataset.

| Param | Type | Meaning | Default |
|-------|------|---------|---------|
| `config` | `dataset_config` **or** `dict` | dataset name, subject(s), optional filtering (nested `filter_config`; `None` = no filtering), resampling, normalization, the train/test split options (`use_original_split` / `split_train_test` / `seed`), and `dataset_folder`. A dict is promoted via `dataset_config.from_dict`. | required |

**Returns** a tuple `(train_dataset, test_dataset)` of two [`EEG_Dataset`](./dataset_time.md) (train first), each with trials of shape **`[n_trials, 1, n_channels, n_times]`** (`float32`), integer labels, channel names attached (`.ch_list`), and the sampling frequency attached (`.fs`).

### How the two splits are built
After loading the data, `get_dataset` picks one of two strategies:
- **Original (session-name) split** — used when `config.use_original_split` is `True` **and** `_original_split_available(all_session_names)` is `True`. The trials of the `train`-named sessions form `train_dataset`, those of the `test`-named sessions form `test_dataset` (each selected via `_session_matches`). The two `_collect_trials` calls use disjoint session predicates, so every recording is still preprocessed exactly once.
- **Random split** — used otherwise (`use_original_split=False`, or `True` but no session-name division: a single session, or names with no train/test marker). **All** sessions are pooled and epoched **once**, then `create_train_test_split_indices(n_trials, config.split_train_test, config.seed)` partitions the trials into `train_indices` / `test_indices`. The two sets are complementary by construction; `config.seed` only makes the partition reproducible across runs. (Building both splits in a single call also removes the old redundancy of preprocessing every session twice.)

### Pipeline steps
1. **Input check:** dict → `dataset_config.from_dict`; else must be a `dataset_config` (`TypeError` otherwise). `_import_moabb_mne()` (fails early with the install hint if the extra is missing).
2. **Download + load:** `download_dataset(config.dataset_name, config.subjects, config.dataset_folder)` (idempotent; also yields the dataset object; `config.dataset_folder` redirects the download/look-up directory when set). `dataset.get_data(subjects=...)` → nested dict `{subject: {session: {run: Raw}}}`.
3. **Decide the split:** gather the sorted union of all session names; `use_name_split = config.use_original_split and _original_split_available(names)` (see above).
4. **Collect trials** via `_collect_trials`: for the original split, call it twice (train-named then test-named sessions); for the random split, call it once (all sessions) and split the pooled trials by index. A `RuntimeError` is raised if a needed set has no trials, or if the random split leaves the train or the test side empty (e.g. a degenerate `split_train_test` of 0/1 or too few trials).
5. **Assemble:** add the depth-1 axis `X = X[:, np.newaxis, :, :]` → `[trials, 1, channels, time]` to each split; the sampling frequency is `float(config.resample_freq)` when resampling is enabled, otherwise the captured original one; then `train_dataset` and `test_dataset` are built via `EEG_Dataset(X, labels, ch_list, normalize=config.normalize, fs=fs)` and returned as a tuple.

### Side effects
- May download data to disk (via `download_dataset`).
- Emits a `RuntimeWarning` on channel-name mismatches. (The old positional/single-session *split* fallbacks — and their warnings — are gone: those cases now take the random split.)

### Errors raised
| Error | When |
|-------|------|
| `ImportError` | `moabb`/`mne` not installed (via `_import_moabb_mne`) |
| `TypeError` | `config` is neither a `dataset_config` nor a dict |
| `ValueError` | unknown dataset name (from `_resolve_dataset_class`) |
| `RuntimeError` | no trials extracted for a split; or the random split leaves the train or the test side empty |

### Connection notes
- `config.normalize` flows straight into `EEG_Dataset`'s `normalize` argument (codes `-1/1/2/3`; see [`dataset_time.md`](./dataset_time.md)).
- The sampling frequency is forwarded into `EEG_Dataset`'s `fs` argument, so each returned dataset knows its own `fs` (used by the plot methods). It is `config.resample_freq` when resampling, else the original moabb sampling frequency.
- The `[trials, 1, channels, time]` shape is chosen for `Conv2d` models in `jade.model`.
- Both returned datasets are meant to be handed to `jade.training.support_training.get_dataloader` (one loader per split).
