# `jade.dataset` — Module Quickstart

> **Read this first.** After reading only this page you should understand the whole `dataset` module without opening the source. The other pages (`__init__.md`, `dataset_time.md`, `moabb_dataset.md`, `support_plot_freq.md`) are deep dives for when you need exact signatures, gotchas, or tensor shapes.

## Purpose

`jade.dataset` is the **data-management layer** of the `jade` package (Just Another Deep-learning EEG package). It covers everything between "raw public EEG recordings" and "a `torch` object a training loop can consume":

1. **Download** EEG datasets bundled with [`moabb`](https://moabb.neurotechx.com/).
2. **Preprocess** them (optional band-/notch-filtering + optional resampling) through MNE's own routines. Filtering is applied only when the `dataset_config` sets a `filter`; with no filter (the default) the raw signal is kept.
3. **Epoch** continuous recordings into per-trial tensors.
4. **Wrap** them into PyTorch `Dataset` objects (`EEG_Dataset` / `EEG_Dataset_ChWi`) with optional normalization.
5. **Inspect** individual trials with time- and frequency-domain plotting helpers.

## Structure (4 source files)

| File | Role |
|------|------|
| [`__init__.py`](./__init__.md) | Package surface. Eagerly re-exports the two Dataset classes; **lazily** re-exports the moabb loader via PEP 562 `__getattr__` so `import jade.dataset` never pulls in the heavy optional `moabb`/`mne`. |
| [`dataset_time.py`](./dataset_time.md) | The two `torch.utils.data.Dataset` subclasses (`EEG_Dataset`, `EEG_Dataset_ChWi`), their normalization schemes, and per-trial plotting methods. The always-available core. |
| [`moabb_dataset.py`](./moabb_dataset.md) | `download_dataset` / `get_dataset`: the moabb+MNE pipeline that turns a `dataset_config` into a ready-to-train `EEG_Dataset`. Optional (needs the `dataset` extra). |
| [`support_plot_freq.py`](./support_plot_freq.md) | `compute_frequency_spectrum` + `check_freq_config`: a Welch-method PSD estimator with a scipy → torch → numpy backend fallback chain. Used by the frequency plots. |

## End-to-end data flow

```
dataset_config  ──►  get_dataset(config, dataset_type)          [moabb_dataset.py]
                        │
                        ├─ download_dataset()  → moabb caches raw files on disk (idempotent)
                        ├─ dataset.get_data()  → {subject: {session: {run: mne.Raw}}}
                        ├─ _select_sessions()  → pick train/test sessions (name heuristic)
                        ├─ _apply_preprocess() → MNE Raw.filter / notch_filter (only if a filter is set) + Raw.resample (in place)
                        ├─ _epoch_raw()        → mne.Epochs → X (trials, ch, time) + int labels + ch_names
                        └─ np.concatenate + add depth axis → X (trials, 1, ch, time)
                        ▼
                   EEG_Dataset(X, labels, ch_list, normalize, fs)    [dataset_time.py]
                        ▼
   jade.training.support_training.get_dataloader(dataset, ...) → torch DataLoader
                        ▼
                   training loop consumes (data, label) batches
```

The `EEG_Dataset_ChWi` variant is built **directly** from an EEG array (not through `get_dataset`); it feeds the "ChWi" channel-wise autoencoder rather than a trial classifier.

## Key classes / functions

| Name | Kind | One-liner |
|------|------|-----------|
| `EEG_Dataset` | class | Trial-level dataset. Stores data as `[trials, 1, channels, time]` (the depth-1 axis is for `Conv2d`). `__getitem__` → `(data, label)`. |
| `EEG_Dataset_ChWi` | class | Channel-wise dataset for the ChWi autoencoder. Flattens `[trials, C, T]` → `[(trials·C), 1, T]` (depth-1 axis for `Conv1d`). Labels optional. |
| `get_dataset(config, dataset_type)` | func | Full moabb pipeline → an `EEG_Dataset`. Downloads if needed. |
| `download_dataset(name, subjects, dataset_folder)` | func | Instantiate + download a moabb dataset (idempotent). `dataset_folder` optionally redirects where moabb stores/looks up the data (single global setting; `None` → `~/mne_data`). Wrapped by the `download-dataset-moabb` cli command. |
| `compute_frequency_spectrum(signal, fs, freq_config)` | func | Welch PSD of a 1-D signal, backend-agnostic (scipy/torch/numpy). |
| `check_plot_config` / `check_freq_config` | func | Fill-and-validate config dicts for plots / PSD. |

## Tensor shapes produced

| Producer | Shape | Notes |
|----------|-------|-------|
| `_epoch_raw` (internal) | `(n_trials, n_channels, n_times)` | EEG channels only, `float64` from MNE |
| `get_dataset` before wrapping | `(n_trials, 1, n_channels, n_times)` `float32` | depth-1 axis added for `Conv2d` |
| `EEG_Dataset.data` | `[trials, 1, channels, time]` `torch.float32` | labels `torch.long` |
| `EEG_Dataset_ChWi.data` | `[(trials·channels), 1, time]` `torch.float32` | one row per (trial, channel) pair |

## Normalization schemes

Two classes, **two different conventions** — do not confuse them.

`EEG_Dataset(normalize=…)` — range hard-wired to **[-1, 1]**:

| code | method | scope |
|------|--------|-------|
| `-1` | (none) | — |
| `1` | `minmax_normalize_all_dataset` | one min/max over the **whole** dataset |
| `2` | `normalize_trial_by_trial_channel_by_channel` | per trial, per channel |
| `3` | `normalize_trial_by_trial` | per trial (all channels together) |

These codes match `jade.config.dataset.NORMALIZE_CODES = (-1, 1, 2, 3)`.

`EEG_Dataset_ChWi(norm_type=…)` — per-signal (each `[1, T]` row):

| code | method |
|------|--------|
| `<= 0` | (none) |
| `1` | min-max to **[0, 1]** |
| `2` | standardization (mean 0, std 1) |

## Important design decisions

- **`moabb`/`mne` are optional and imported lazily.** They are heavy and only needed for downloading/preprocessing. `import jade.dataset` gives you the Dataset classes with zero heavy imports; the loader is materialized only when `download_dataset`/`get_dataset` are first accessed (PEP 562 `__getattr__` in `__init__.py`, plus an in-function `_import_moabb_mne()`). Declared as the `dataset` extra in `pyproject.toml` (`pip install jade[dataset]`).
- **No custom DSP.** Filtering/resampling/epoching all delegate to MNE's own routines — exactly the ones moabb uses — so results match moabb. Filtering and resampling are both **optional**: `_apply_preprocess` filters only when `config.filter is not None` and resamples only when `config.resample` is set, so an unspecified filter leaves the raw signal untouched.
- **`scipy` is also optional** for PSD plots: `compute_frequency_spectrum` falls back scipy → torch → numpy, so frequency plots work in a bare environment.
- **The ChWi "channel-wise" variant** treats every channel of every trial as an independent 1-D sample (`[(B·C), 1, T]`), discarding cross-channel/trial structure. That is what an autoencoder learning a per-channel EEG representation wants; the classifier `EEG_Dataset` keeps the `[trials, 1, channels, time]` grid instead.
- **Deterministic labels.** `_epoch_raw` maps moabb event names → 0-based class ids sorted by class name, so labels are reproducible across runs.

## Dependencies

- **External (always):** `numpy`, `torch`, `matplotlib`.
- **External (optional):** `moabb`, `mne` (the `dataset` extra, for the loader); `scipy` (preferred PSD backend).
- **Internal:** `jade.config.dataset.dataset_config` and `jade.config.filter.filter_config` drive `get_dataset`; `dataset_time` imports `compute_frequency_spectrum` from `support_plot_freq`.

## How `dataset` connects to the rest of `jade`

- **`jade.config`** → drives it. `dataset_config` (which nests an optional `filter_config`; `None` = no filtering) is the single input object to `get_dataset`; `dataset_config.normalize` is forwarded straight to `EEG_Dataset`, and the sampling frequency (from the raw moabb data, or `resample_freq` when resampling) is forwarded to `EEG_Dataset.fs`.
- **`jade.training`** → consumes it. `jade.training.support_training.get_dataloader(dataset, batch_size, …)` wraps any of these `Dataset` objects into a `torch` `DataLoader`; the classifier training loop then pulls `(data, label)` batches — exactly the tuple `EEG_Dataset.__getitem__` yields.
- **`jade.model`** → the shapes here are chosen for it: the depth-1 axis feeds `Conv2d` models (e.g. `EEGNet`), and the ChWi flattening feeds the channel-wise autoencoder.
