# `jade.dataset` — Module Index

Table of contents for the `dataset` module wiki. Start with the quickstart; open a per-file page for exact signatures, parameters, shapes, and gotchas.

| Page | Source file | Description |
|------|-------------|-------------|
| [`0_module_quickstart.md`](./0_module_quickstart.md) | — | **Read first.** Complete-but-concise overview: purpose, structure, end-to-end data flow, key API, tensor shapes, normalization schemes, and how `dataset` links to `config`, `training`, and `model`. |
| [`__init__.md`](./__init__.md) | `src/jade/dataset/__init__.py` | Package surface: eager re-export of the two Dataset classes and the **lazy PEP 562 `__getattr__`** re-export of the moabb loader (keeps `moabb`/`mne` off the import path). |
| [`dataset_time.md`](./dataset_time.md) | `src/jade/dataset/dataset_time.py` | The core `torch` datasets `EEG_Dataset` and `EEG_Dataset_ChWi`: `__init__`/`__len__`/`__getitem__`, normalization schemes, plotting methods, plus the `check_plot_config` helper. |
| [`moabb_dataset.md`](./moabb_dataset.md) | `src/jade/dataset/moabb_dataset.py` | The moabb+MNE loader: `download_dataset` and `get_dataset`, plus the internal dataset-name resolution, train/test session selection, filtering/resampling, and epoching helpers. |
| [`support_plot_freq.md`](./support_plot_freq.md) | `src/jade/dataset/support_plot_freq.py` | Frequency-domain support: `compute_frequency_spectrum` (Welch PSD with scipy → torch → numpy fallback), `check_freq_config`, and the numpy/torch Welch backends. |
