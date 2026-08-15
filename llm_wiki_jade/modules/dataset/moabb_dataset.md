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

Both returned arrays are `numpy` int arrays; together they form a **partition** of `range(n_samples)` (disjoint, covering every index once). Because the result depends only on `(n_samples, split_train_test, seed)`, two calls with the same triple return the identical partition — this is exactly what makes the two separate `get_dataset(…, 'train')` / `get_dataset(…, 'test')` calls **complementary** (they share the `seed` carried by the `dataset_config`).

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

---

## Public: `get_dataset(config, dataset_type="train") -> EEG_Dataset`

The one-call pipeline: config → ready `EEG_Dataset`.

| Param | Type | Meaning | Default |
|-------|------|---------|---------|
| `config` | `dataset_config` **or** `dict` | dataset name, subject(s), optional filtering (nested `filter_config`; `None` = no filtering), resampling, normalization, the train/test split options (`use_original_split` / `split_train_test` / `seed`), and `dataset_folder`. A dict is promoted via `dataset_config.from_dict`. | required |
| `dataset_type` | `{'train', 'test'}` | which split to return. | `"train"` |

**Returns** an [`EEG_Dataset`](./dataset_time.md) with trials of shape **`[n_trials, 1, n_channels, n_times]`** (`float32`), integer labels, channel names attached (`.ch_list`), and the sampling frequency attached (`.fs`).

### How the split is decided
After loading the data, `get_dataset` picks one of two strategies:
- **Original (session-name) split** — used when `config.use_original_split` is `True` **and** `_original_split_available(all_session_names)` is `True`. Only the sessions whose name matches `dataset_type` (via `_session_matches`) are loaded and returned.
- **Random split** — used otherwise (`use_original_split=False`, or `True` but no session-name division: a single session, or names with no train/test marker). **All** sessions are pooled and epoched, then `create_train_test_split_indices(n_trials, config.split_train_test, config.seed)` partitions the trials; the `'train'` call keeps `train_indices`, the `'test'` call keeps `test_indices`. Sharing `config.seed` across the two calls makes the results reproducible and complementary. Note this means the random branch loads **all** sessions in *each* call (train and test), so the preprocessing is recomputed per call — the trade-off for the two-call API.

### Pipeline steps
1. **Input check:** dict → `dataset_config.from_dict`; else must be a `dataset_config` (`TypeError` otherwise). `dataset_type` lowercased and validated to `train`/`test` (`ValueError` otherwise). `_import_moabb_mne()` (fails early with the install hint if the extra is missing).
2. **Download + load:** `download_dataset(config.dataset_name, config.subjects, config.dataset_folder)` (idempotent; also yields the dataset object; `config.dataset_folder` redirects the download/look-up directory when set). `dataset.get_data(subjects=...)` → nested dict `{subject: {session: {run: Raw}}}`.
3. **Decide the split:** gather the sorted union of all session names; `use_name_split = config.use_original_split and _original_split_available(names)` (see above).
4. **Per-recording preprocess + epoch:** for each subject, choose the sessions to load — the name-matched ones (`_session_matches`) under the original split, or **all** sessions under the random split — then, iterating sessions/runs in **sorted** order (deterministic, required for a reproducible random split), `_apply_preprocess` then `_epoch_raw`. The **original sampling frequency** is captured once from the first recording (`raw.info["sfreq"]`) *before* preprocessing. Empty-epoch recordings (`X.shape[0] == 0`) are skipped. Channel names from the first recording are kept; a mismatch across recordings emits a `RuntimeWarning` (keeps the first set).
5. **Assemble:** if nothing was extracted → `RuntimeError`. Else `np.concatenate` all `X` (`float32`) and labels (`int64`). Under the random split, `create_train_test_split_indices` then selects the `dataset_type` subset (a `RuntimeError` is raised if that subset is empty, e.g. a degenerate `split_train_test` of 0/1 or too few trials). Finally add the depth-1 axis `X = X[:, np.newaxis, :, :]` → `[trials, 1, channels, time]`; the sampling frequency is `float(config.resample_freq)` when resampling is enabled, otherwise the captured original one; then `EEG_Dataset(X, labels, ch_list, normalize=config.normalize, fs=fs)` is returned.

### Side effects
- May download data to disk (via `download_dataset`).
- Emits a `RuntimeWarning` on channel-name mismatches. (The old positional/single-session *split* fallbacks — and their warnings — are gone: those cases now take the random split.)

### Errors raised
| Error | When |
|-------|------|
| `ImportError` | `moabb`/`mne` not installed (via `_import_moabb_mne`) |
| `TypeError` | `config` is neither a `dataset_config` nor a dict |
| `ValueError` | `dataset_type` not `train`/`test`; or unknown dataset name (from `_resolve_dataset_class`) |
| `RuntimeError` | no trials extracted for the requested dataset/subjects/split; or the random split leaves the requested `dataset_type` subset empty |

### Connection notes
- `config.normalize` flows straight into `EEG_Dataset`'s `normalize` argument (codes `-1/1/2/3`; see [`dataset_time.md`](./dataset_time.md)).
- The sampling frequency is forwarded into `EEG_Dataset`'s `fs` argument, so the returned dataset knows its own `fs` (used by the plot methods). It is `config.resample_freq` when resampling, else the original moabb sampling frequency.
- The `[trials, 1, channels, time]` shape is chosen for `Conv2d` models in `jade.model`.
- The returned dataset is meant to be handed to `jade.training.support_training.get_dataloader`.
