# `config/dataset.py`

Source: `src/jade/config/dataset.py`

Defines `dataset_config`, which gathers everything needed to download and preprocess a moabb dataset before it is wrapped into `jade.dataset.dataset_time.EEG_Dataset`: which dataset/subjects to download, the filtering (a nested [`filter_config`](filter.md)), optional resampling, and the normalization scheme. Serialized to TOML, the nested filter becomes its own `[filter]` section.

## Module constant

| Constant | Value | Meaning |
|---|---|---|
| `NORMALIZE_CODES` | `(-1, 1, 2, 3)` | Normalization schemes understood by `EEG_Dataset`: `-1` none, `1` min-max over whole dataset, `2` per-trial per-channel, `3` per-trial. |

## Private helper

### `_default_filter() -> filter_config`
Default-factory for the `filter` field. Returns `filter_config(filter_type="bandpass", l_freq=8.0, h_freq=32.0)` — an 8–32 Hz band-pass, close to moabb's motor-imagery default of 7–35 Hz. Used via `field(default_factory=...)` so each `dataset_config` gets its own filter instance (never a shared mutable default).

## Class `dataset_config(generic_config)`

`@dataclass`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `dataset_name` | `str` | `""` | moabb dataset name (e.g. `"BNCI2014_001"`). **Required** — empty string fails validation. |
| `subjects` | `list[int] \| None` | `None` | Subjects to download; `None` = all. Each entry must be a positive int. |
| `filter` | `filter_config` | `_default_filter()` | Nested filtering config (8–32 Hz band-pass by default). |
| `resample` | `bool` | `False` | Whether to resample after filtering. |
| `resample_freq` | `float \| None` | `None` | Target sampling frequency (Hz). Required and positive when `resample=True`. |
| `normalize` | `int` | `-1` | Normalization scheme forwarded to `EEG_Dataset`; one of `NORMALIZE_CODES`. |

## `__post_init__` — validation

Raises `ValueError` on any violation, in order:

1. **`dataset_name`**: must be a non-empty string.
2. **`subjects`**: if not `None`, must be a list/tuple; every element must be a positive `int` (`bool` is **explicitly rejected** since `bool ⊂ int`). Cleaned entries are re-stored as `int` in `self.subjects`.
3. **`filter` promotion**: if `filter` is a `dict` (e.g. from a TOML/JSON file), it is promoted via `filter_config.from_dict`; if it is neither a dict nor a `filter_config`, `ValueError`. This guarantees the nested field is always a real dataclass instance.
4. **`resample`/`resample_freq`**: when `resample` is truthy, `resample_freq` must be a positive number.
5. **`normalize`**: must be in `NORMALIZE_CODES`.

## Gotchas

- The `filter` promotion mirrors the same pattern used by [`training_config`](training.md) for its nested configs; it is what lets `dataset_config.from_toml(...)` rebuild the whole tree correctly.
- `bool` is a subclass of `int`, so `subjects=[True]` is rejected on purpose.
- The normalization codes are just integer flags here; the actual normalization logic lives in `EEG_Dataset` (`jade.dataset`).
