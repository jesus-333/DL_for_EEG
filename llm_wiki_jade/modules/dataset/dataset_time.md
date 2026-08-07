# `jade.dataset.dataset_time`

Source: `src/jade/dataset/dataset_time.py`

Defines the two PyTorch `Dataset` subclasses that hold EEG data in **channels × time-samples** form (`EEG_Dataset` for trial classification, `EEG_Dataset_ChWi` for the channel-wise autoencoder), plus per-trial plotting methods and a plot-config validator. This is the always-available core of the module (only `numpy`, `torch`, `matplotlib` are needed).

## Module-level items

### `VALID_PLOT_FORMATS`
```python
VALID_PLOT_FORMATS = ('png', 'jpeg', 'pdf', 'eps')
```
The image extensions accepted when a figure is saved. `check_plot_config` rejects anything else.

### `check_plot_config(plot_config=None) -> dict`

Validate a plot-configuration dict and fill missing keys with defaults. **Never mutates the input** (works on a copy). If `plot_config` is `None` or empty, returns a fresh dict of all defaults.

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `figsize` | tuple | `(12, 8)` | figure size |
| `fontsize` | int | `16` | font size for labels/ticks/title |
| `linewidth` | float | `1.3` | plotted line width |
| `color` | str | `'black'` | line color |
| `x_lim` | tuple or None | `None` | x-axis limits (auto if None) |
| `y_lim` | tuple or None | `None` | y-axis limits (auto if None) |
| `add_title` | bool | `True` | add a title to the plot |
| `title_prefix` | str or None | `None` | prefix for both title and file name |
| `save_plot` | bool | `False` | save the figure to disk |
| `plot_format` | list of str | `['png']` | save extensions (must be in `VALID_PLOT_FORMATS`) |
| `plot_path` | str | `'plot/'` | destination folder (created if absent) |

Behavior / gotchas:
- `x_lim`, `y_lim`, `title_prefix` may legitimately be `None`, so an explicit `None` for them is **kept**; for all other keys an explicit `None` is replaced by the default (they are in `keys_that_allow_none`).
- `add_title` and `save_plot` are coerced through `_to_bool`, so the strings `'True'`/`'False'` (any case) are accepted.
- A single-string `plot_format` is wrapped into a one-element list.
- **Raises `ValueError`** if any requested format is not in `VALID_PLOT_FORMATS`.

### `_to_bool(value) -> bool`
Convert to bool, additionally accepting the strings `'True'`/`'False'` (case-insensitive, stripped). Any other string raises `ValueError`; non-strings go through `bool(value)`.

---

## class `EEG_Dataset(torch.utils.data.Dataset)`

Trial-level dataset for classification. Stores every trial as a 4-D tensor `[trials, 1, channels, time]`; the size-1 "depth" axis exists so the data feeds `torch.nn.Conv2d` directly.

### `__init__(self, data, labels, ch_list, normalize = -1)`

| Param | Type | Meaning | Default |
|-------|------|---------|---------|
| `data` | `numpy.ndarray` | EEG trials, shape **`[Trials, 1, channels, time samples]`** (4-D, axis 1 == 1). | required |
| `labels` | `numpy.ndarray` | integer class labels, shape `[Trials]`. | required |
| `ch_list` | list / sequence or None | channel names, used to resolve channel names in the plot methods and returned by `get_dataset`. | required (may be `None`) |
| `normalize` | int | normalization code (see table below). | `-1` |

Behavior:
- **Shape check:** raises `ValueError` if `data.ndim != 4` or `data.shape[1] != 1`.
- **Type conversion:** `self.data = torch.from_numpy(data).float()` (→ `torch.float32`), `self.labels = torch.from_numpy(labels).long()` (→ `torch.int64`). **`data`/`labels` must be numpy arrays** — unlike `EEG_Dataset_ChWi`, this class has no torch-tensor branch, so passing a torch tensor raises.
- Stores `self.ch_list` verbatim.
- Applies normalization by dispatching on `normalize` (see below). Note `normalize = -1` (and any value other than 1/2/3) means **no normalization**.

### Attributes stored
| Attribute | Type | Shape / content |
|-----------|------|-----------------|
| `self.data` | `torch.float32` | `[trials, 1, channels, time]` |
| `self.labels` | `torch.int64` | `[trials]` |
| `self.ch_list` | list or None | channel names |

### `__getitem__(self, idx) -> (Tensor, Tensor)`
Returns the tuple `(self.data[idx], self.labels[idx])`, i.e. one trial `[1, channels, time]` and its scalar label. This is exactly the `(data, label)` pair the `jade.training` loops expect from a `DataLoader`.

### `__len__(self) -> int`
Returns `len(self.labels)` (number of trials).

### Normalization

`__init__` dispatches on the `normalize` code; **the target range is hard-wired to `[-1, 1]`** at the call sites:

| `normalize` | method called | effect |
|-------------|---------------|--------|
| `1` | `minmax_normalize_all_dataset(-1, 1)` | single global min/max over the entire tensor |
| `2` | `normalize_trial_by_trial_channel_by_channel(-1, 1)` | independent min/max per (trial, channel) |
| `3` | `normalize_trial_by_trial(-1, 1)` | independent min/max per trial (all channels jointly) |
| other (e.g. `-1`) | — | no normalization |

The three methods are also public and take explicit `(a, b)` bounds:

- **`minmax_normalize_all_dataset(self, a, b)`** — vectorized:
  `self.data = ((self.data - min) / (max - min)) * (b - a) + a` using the global min/max.
- **`normalize_trial_by_trial_channel_by_channel(self, a, b)`** — Python double loop over trials × channels, min-max scaling each `self.data[i, 0, j]` (a `[time]` vector).
- **`normalize_trial_by_trial(self, a, b)`** — Python loop over trials, min-max scaling each `self.data[i, 0, :]` (a `[channels, time]` slice) with a single shared min/max.

Gotchas:
- The per-trial/per-channel variants loop in pure Python, so they are **slow on large datasets**.
- No guard against a constant signal (`max == min` → division by zero → `nan`/`inf`).
- The codes here match `jade.config.dataset.NORMALIZE_CODES = (-1, 1, 2, 3)`, and `get_dataset` forwards `dataset_config.normalize` straight into this argument.

### Plotting methods

All three public plot methods share the signature pattern `(idx_sample, idx_ch, fs=1, plot_config=None[, freq_config=None])` and return matplotlib objects. They call `check_plot_config` (so `plot_config=None` is fine), resolve the channel via `_resolve_channel`, draw, optionally title, `tight_layout`, and optionally save via `_save_figure`.

| Method | Draws | Returns |
|--------|-------|---------|
| `plot_trial_time(idx_sample, idx_ch, fs=1, plot_config=None)` | time-domain signal of one channel of one trial | `(fig, ax)` |
| `plot_trial_freq(idx_sample, idx_ch, fs=1, plot_config=None, freq_config=None)` | PSD of one channel of one trial | `(fig, ax)` |
| `plot_trial_time_and_freq(idx_sample, idx_ch, fs=1, plot_config=None, freq_config=None)` | time + freq side by side (1×2) | `(fig, axs)` |

Common parameters:
- `idx_sample : int` — trial index.
- `idx_ch : int or str` — channel index, **or** channel name (resolved against `self.ch_list`).
- `fs : int` — sampling frequency (Hz). Default `1` = normalized frequency: the time axis then treats each sample as 1 ms, and the frequency plot switches axis labels to "Normalized frequency" / drops the `/Hz` unit.
- `plot_config` — see `check_plot_config`.
- `freq_config` — see [`support_plot_freq.check_freq_config`](./support_plot_freq.md) (frequency methods only).

**Side effect:** when `plot_config['save_plot']` is truthy these methods write image files to disk (and create `plot_path`).

### Plot helper (private) methods

| Helper | Purpose / notes |
|--------|-----------------|
| `_resolve_channel(idx_ch) -> (idx_ch_int, ch_label)` | Turns an int index or a channel-name string into `(int index, label)`. Name lookup uses `self.ch_list`; raises `ValueError` if `ch_list is None` or the name is absent. For an int index the label is the channel name when available, else the int. |
| `_plot_time_on_ax(ax, idx_sample, idx_ch_int, fs, plot_config)` | Extracts `self.data[idx_sample, 0, idx_ch_int]` → numpy, builds a ms time axis `arange(N)/fs*1000`, plots, labels ("Time [ms]" / "Amplitude [µV]"), grid, optional limits/ticks. |
| `_plot_freq_on_ax(ax, idx_sample, idx_ch_int, fs, plot_config, freq_config)` | Extracts the signal, calls `compute_frequency_spectrum`, plots PSD. Axis labels depend on `fs`: `fs != 1` → "Frequency [Hz]" / "PSD [µV²/Hz]"; `fs == 1` → "Normalized frequency" / "PSD [µV²]". |
| `_build_title(plot_config, idx_sample, ch_label) -> str` | `"{title_prefix} - {idx_trial} - {idx_ch}"`; drops the prefix (and its `-`) when `title_prefix is None`. |
| `_save_figure(fig, plot_config, idx_sample, ch_label)` | No-op unless `save_plot`. Creates `plot_path` (`os.makedirs(..., exist_ok=True)`), builds file name `"{title_prefix}_{idx_trial}_{idx_ch}"` (prefix dropped when None), and `fig.savefig` in every requested format with `bbox_inches='tight'`. |

---

## class `EEG_Dataset_ChWi(torch.utils.data.Dataset)`

"Channel-Wise" (ChWi) dataset used by the ChWi autoencoder. **Every channel of every trial becomes an independent 1-D sample.** An input EEG matrix `B × C × T` is stored as `(B·C) × 1 × T`; the size-1 axis is the depth map for `torch.nn.Conv1d`. Cross-channel and cross-trial structure is intentionally discarded — the autoencoder learns a per-channel EEG representation.

### `__init__(self, data, labels=None, ch_list=None, norm_type=-1)`

| Param | Type | Meaning | Default |
|-------|------|---------|---------|
| `data` | `numpy.ndarray` **or** `torch.Tensor` | EEG, shape `[Trials, C, T]` (`C` = channels, `T` = time). | required |
| `labels` | array / tensor or None | optional labels. If `None`, a `torch.ones(N)` placeholder is created (keeps the dataset shape-compatible with the training functions). | `None` |
| `ch_list` | list or None | channel names, stored verbatim. | `None` |
| `norm_type` | int | per-signal normalization code (see below). | `-1` |

Behavior:
- Accepts numpy **or** torch: `'torch' in str(type(data))` → `data.float()`, else `torch.from_numpy(data).float()`.
- **Reshape:** `self.data = self.data.reshape(-1, 1, self.data.shape[2])` → `[(Trials·C), 1, T]`. This relies on `data` being 3-D; `shape[2]` is the time dimension.
- Normalizes when `norm_type > 0` (via `__normalize`).
- **Labels:** if `None` → `self.labels = torch.ones(self.data.shape[0])` (length = number of flattened rows). Otherwise converted with `.long()` / `torch.from_numpy(...).long()`.
- Sets `self.return_only_data = False` (a public flag; set it to `True` after construction to make `__getitem__` return only the data — useful for pure-reconstruction autoencoder training).

### Attributes stored
| Attribute | Type | Shape / content |
|-----------|------|-----------------|
| `self.data` | `torch.float32` | `[(trials·channels), 1, time]` |
| `self.labels` | `torch` tensor | `[(trials·channels)]` — real labels (`.long`) or a ones-placeholder (`float`) |
| `self.ch_list` | list or None | channel names |
| `self.return_only_data` | bool | toggles `__getitem__` output |

### `__getitem__(self, idx)`
If `self.return_only_data` → returns `self.data[idx]` (shape `[1, T]`); otherwise returns `(self.data[idx], self.labels[idx])`.

### `__len__(self) -> int`
Returns `self.data.shape[0]` (number of flattened channel-trials).

### `__normalize(self, norm_type)` (private)
Loops over rows (`self.data[i, 0]`, a `[T]` vector) and scales each **independently**:

| `norm_type` | effect |
|-------------|--------|
| `1` | min-max to **`[0, 1]`**: `(x - min) / (max - min)` |
| `2` | standardization: `(x - mean) / std` (mean 0, std 1) |
| other | raises `ValueError` |

> **Note the divergence from `EEG_Dataset`.** Here `norm_type` codes mean min-max-to-`[0,1]` / standardization and are applied per 1-D signal; in `EEG_Dataset` the `normalize` codes mean whole-dataset / per-trial-per-channel / per-trial min-max, all to `[-1, 1]`. Same-looking integer arguments, different semantics.

### Gotchas
- **Labels-conversion branch reads the wrong variable.** When `labels is not None`, the code chooses the conversion path with `if 'torch' in str(type(data))` — it inspects **`data`**, not `labels`. If `data` is numpy but `labels` is already a torch tensor (or vice-versa), the wrong branch runs and the conversion can fail. Pass `data` and `labels` as the same array kind to stay safe.
- The ones-placeholder labels are `float` (from `torch.ones`), whereas real labels are cast to `long`; downstream code relying on integer labels should supply real ones.
- No `max == min` guard in `norm_type == 1` (constant signal → division by zero).
