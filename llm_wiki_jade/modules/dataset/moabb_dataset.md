# `jade.dataset.moabb_dataset`

Source: `src/jade/dataset/moabb_dataset.py`

Turns the EEG datasets bundled with [`moabb`](https://moabb.neurotechx.com/) into `torch`-ready [`EEG_Dataset`](./dataset_time.md) objects: download → preprocess (MNE filter + resample) → epoch → wrap. Two public functions (`download_dataset`, `get_dataset`) plus internal helpers. `moabb`/`mne` are optional (the `dataset` extra) and imported lazily inside the functions that need them.

## Public API
```python
__all__ = ["download_dataset", "get_dataset"]
```
Both are re-exported lazily from the package via `__init__.__getattr__` (see [`__init__.md`](./__init__.md)).

## Imports & configuration
- Internal: `dataset_config` (`jade.config.dataset`), `filter_config` (`jade.config.filter`), `EEG_Dataset` (`jade.dataset.dataset_time`).
- Third-party eager: `numpy`, `warnings`.
- `moabb`/`mne`: imported **only** through `_import_moabb_mne()`.

`get_dataset` is driven by a `dataset_config`, which declares the dataset name, subject(s), filtering (a nested `filter_config`), resampling, and normalization. No custom DSP is implemented here — filtering/resampling/epoching all call MNE's own routines (the same ones moabb uses).

---

## Internal: lazy dependency import

### `_import_moabb_mne() -> (moabb.datasets, mne)`
Imports `mne` and `moabb.datasets` on demand. On `ImportError` re-raises a helpful `ImportError` telling the user to `pip install jade[dataset]` (or `pip install moabb mne`). This is the second laziness layer (the first being `__init__.__getattr__`): even after `moabb_dataset` is imported, `moabb`/`mne` are not loaded until a function actually calls this.

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

## Public: `download_dataset(dataset_name, subjects=None)`

Download a moabb dataset (all subjects, or a subset).

| Param | Type | Meaning | Default |
|-------|------|---------|---------|
| `dataset_name` | str | moabb dataset name (case-insensitive), e.g. `"BNCI2014_001"`. | required |
| `subjects` | int, sequence of int, or None | which subjects to download; `None` → every subject. | `None` |

**Returns** the instantiated moabb dataset object (a `moabb.datasets.base.BaseDataset`), already pointing at the downloaded files and ready for `get_data`.

**Side effect / disk:** calls `dataset.download(subject_list=...)`, which fetches the raw recordings into moabb's cache directory. **Idempotent** — moabb skips already-cached subjects, so repeated calls are cheap.

Internally: `_resolve_dataset_class(name)()` → instance, `_as_subject_list(subjects, dataset)` → id list, then `dataset.download(...)`.

---

## Internal: train / test session selection

moabb organizes data by *session*; `get_dataset` must decide which sessions are "train" and which are "test".

### `_session_matches(session_name, dataset_type) -> bool`
Heuristic on the lowercased session name:
- **test** if it contains `"test"`/`"eval"`, or ends with `"_e"`, or equals `"e"`.
- **train** if it contains `"train"`, or ends with `"_t"`, or equals `"t"`.

Handles common moabb conventions like BNCI2014_001's `'0train'`/`'1test'` and the older `'session_T'`/`'session_E'`.

### `_select_sessions(session_names, dataset_type) -> list`
1. Sort names; return those matching the split by `_session_matches`.
2. If none match **and** there is a single session → use it for **both** splits and emit a `RuntimeWarning`.
3. If none match and there are several → positional fallback (`RuntimeWarning`): **last session = test, the rest = train**.

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
2. If `config.filter is not None` → `_apply_filter`.
3. If `config.resample` → `raw.resample(sfreq=float(config.resample_freq), verbose=False)`.

Resampling runs **last**, so it acts on the already band-limited signal.

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
| `config` | `dataset_config` **or** `dict` | dataset name, subject(s), filtering (nested `filter_config`), resampling, normalization. A dict is promoted via `dataset_config.from_dict`. | required |
| `dataset_type` | `{'train', 'test'}` | which split to return. | `"train"` |

**Returns** an [`EEG_Dataset`](./dataset_time.md) with trials of shape **`[n_trials, 1, n_channels, n_times]`** (`float32`), integer labels, and channel names attached (`.ch_list`).

### Pipeline steps
1. **Input check:** dict → `dataset_config.from_dict`; else must be a `dataset_config` (`TypeError` otherwise). `dataset_type` lowercased and validated to `train`/`test` (`ValueError` otherwise). `_import_moabb_mne()` (fails early with the install hint if the extra is missing).
2. **Download + load:** `download_dataset(config.dataset_name, config.subjects)` (idempotent; also yields the dataset object). `dataset.get_data(subjects=...)` → nested dict `{subject: {session: {run: Raw}}}`.
3. **Per-recording preprocess + epoch:** for each subject → `_select_sessions` → for each selected session/run → `_apply_preprocess` then `_epoch_raw`. Empty-epoch recordings (`X.shape[0] == 0`) are skipped. Channel names from the first recording are kept; a mismatch across recordings emits a `RuntimeWarning` (keeps the first set).
4. **Assemble:** if nothing was extracted → `RuntimeError`. Else `np.concatenate` all `X` (`float32`) and labels (`int64`), add the depth-1 axis `X = X[:, np.newaxis, :, :]` → `[trials, 1, channels, time]`, and return `EEG_Dataset(X, labels, ch_list, normalize=config.normalize)`.

### Side effects
- May download data to disk (via `download_dataset`).
- Emits `RuntimeWarning`s for session-mapping fallbacks and channel-name mismatches.

### Errors raised
| Error | When |
|-------|------|
| `ImportError` | `moabb`/`mne` not installed (via `_import_moabb_mne`) |
| `TypeError` | `config` is neither a `dataset_config` nor a dict |
| `ValueError` | `dataset_type` not `train`/`test`; or unknown dataset name (from `_resolve_dataset_class`) |
| `RuntimeError` | no trials extracted for the requested dataset/subjects/split |

### Connection notes
- `config.normalize` flows straight into `EEG_Dataset`'s `normalize` argument (codes `-1/1/2/3`; see [`dataset_time.md`](./dataset_time.md)).
- The `[trials, 1, channels, time]` shape is chosen for `Conv2d` models in `jade.model`.
- The returned dataset is meant to be handed to `jade.training.support_training.get_dataloader`.
