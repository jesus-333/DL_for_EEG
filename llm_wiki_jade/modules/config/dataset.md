# `config/dataset.py`

Source: `src/jade/config/dataset.py`

Defines `dataset_config`, which gathers everything needed to download and preprocess a moabb dataset before it is wrapped into `jade.dataset.dataset_time.EEG_Dataset`: which dataset/subjects to download, where to download it, the **optional** filtering (a nested [`filter_config`](filter.md)), optional resampling, the normalization scheme, and the **train/test split** options (`use_original_split` / `split_train_test` / `seed`). Filtering is optional — when it is left unspecified (`filter = None`, the default) no filter is applied and the raw signal is kept. Serialized to TOML, the nested filter becomes its own `[filter]` section (omitted entirely when no filter is set).

## Module constant

| Constant | Value | Meaning |
|---|---|---|
| `NORMALIZE_CODES` | `(-1, 1, 2, 3)` | Normalization schemes understood by `EEG_Dataset`: `-1` none, `1` min-max over whole dataset, `2` per-trial per-channel, `3` per-trial. |

## Class `dataset_config(generic_config)`

`@dataclass`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `dataset_name` | `str` | `""` | moabb dataset name (e.g. `"BNCI2014_001"`). **Required** — empty string fails validation. |
| `subjects` | `list[int] \| None` | `None` | Subjects to download; `None` = all. Each entry must be a positive int. |
| `filter` | `filter_config \| None` | `None` | Optional nested filtering config. `None` (the default) = **no filtering**, the raw signal is kept. A `dict` is promoted to a `filter_config`; an empty dict is treated as `None`. |
| `resample` | `bool` | `False` | Whether to resample after filtering. |
| `resample_freq` | `float \| None` | `None` | Target sampling frequency (Hz). Required and positive when `resample=True`. |
| `normalize` | `int` | `-1` | Normalization scheme forwarded to `EEG_Dataset`; one of `NORMALIZE_CODES`. |
| `use_original_split` | `bool` | `True` | Prefer the moabb *session-name* train/test split (e.g. `'0train'`/`'1test'`). Falls back to the random split when the names carry no division; `False` always uses the random split. Consumed by [`get_dataset`](../dataset/moabb_dataset.md). |
| `split_train_test` | `float` | `0.5` | Fraction of samples used for **training** in the random split (rest → test). Must be in `[0, 1]`. Used when `use_original_split=False`, or `True` but no session-name split exists. Coerced to `float`. |
| `seed` | `int` | `42` | Non-negative seed for the random split (via [`create_train_test_split_indices`](../dataset/moabb_dataset.md)). `get_dataset` builds both splits from one partition, so they are complementary regardless of the seed; the seed only makes that partition reproducible across runs. Kept a concrete int (**not** `None`, default `42`) for reproducible splits out of the box. |
| `dataset_folder` | `str \| None` | `None` | Download/look-up folder, forwarded to [`download_dataset`](../dataset/moabb_dataset.md); `None` = moabb default (`~/mne_data`). Single global moabb setting — only the last folder set is consulted. |

## `__post_init__` — validation

Raises `ValueError` on any violation, in order:

1. **`dataset_name`**: must be a non-empty string.
2. **`subjects`**: if not `None`, must be a list/tuple; every element must be a positive `int` (`bool` is **explicitly rejected** since `bool ⊂ int`). Cleaned entries are re-stored as `int` in `self.subjects`.
3. **`filter` normalization**: `filter` is optional. `None` is kept as-is (no filtering). A `dict` (e.g. from a TOML/JSON file) is promoted via `filter_config.from_dict` — except an **empty dict**, which is treated as `None` (no filtering). Anything else (not `None`, dict or `filter_config`) raises `ValueError`. So the nested field is always either a real `filter_config` instance or `None`.
4. **`resample`/`resample_freq`**: when `resample` is truthy, `resample_freq` must be a positive number.
5. **`normalize`**: must be in `NORMALIZE_CODES`.
6. **`use_original_split`**: must be a `bool`.
7. **`split_train_test`**: must be a number (`bool` **rejected**, being a subclass of `int`) in `[0, 1]`; then stored as `float`.
8. **`seed`**: must be a non-negative `int` (`bool` rejected); `None` is **not** allowed here (unlike [`training_config.seed`](training.md)) — see gotchas.
9. **`dataset_folder`**: if not `None`, must be a string path.

## Gotchas

- The `filter` promotion mirrors the same pattern used by [`training_config`](training.md) for its nested configs; it is what lets `dataset_config.from_toml(...)` rebuild the whole tree correctly.
- **No filter ⇒ raw data.** When `filter is None` (the default, or an omitted/empty `[filter]` section), [`get_dataset` / `_apply_preprocess`](../dataset/moabb_dataset.md) skips filtering entirely (`if config.filter is not None`) and keeps the raw signal.
- `bool` is a subclass of `int`, so `subjects=[True]` is rejected on purpose (same guard on `split_train_test` and `seed`).
- The normalization codes are just integer flags here; the actual normalization logic lives in `EEG_Dataset` (`jade.dataset`).
- **Why `seed` is a fixed int (not `None`).** [`get_dataset`](../dataset/moabb_dataset.md) builds both splits in one call from a single partition of the pooled trials, so train/test are complementary (disjoint, covering all trials) *by construction* — the seed is not needed for that. It is kept a concrete non-negative int (default `42`, not `None`) so the random split is reproducible across runs out of the box; change it to draw a different (still reproducible) split. This is the deliberate difference from `training_config.seed`, which *is* optional.
- **`use_original_split=True` is not a guarantee.** It's a *preference*: the original split is used only if the session names actually mark train/test (`_original_split_available`). Single-session datasets, or names with no marker, silently fall back to the random split — which is the fix for the old behavior where a single session produced *identical* train and test sets.
