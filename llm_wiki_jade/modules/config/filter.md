# `config/filter.py`

Source: `src/jade/config/filter.py`

Defines `filter_config`, the parameters of **one** moabb/MNE filtering operation. moabb filters a raw signal via MNE's `Raw.filter` (band-pass by default, `method="iir"`, `picks="data"`) or `Raw.notch_filter`; this config mirrors those parameters. Used as the nested `filter` field of [`dataset_config`](dataset.md).

## Module constants

| Constant | Value | Meaning |
|---|---|---|
| `FILTER_TYPES` | `("bandpass", "bandstop", "lowpass", "highpass", "notch")` | Allowed `filter_type` values. |
| `FILTER_METHODS` | `("iir", "fir")` | Allowed filtering backends. |
| `FILTER_PHASES` | `("zero", "zero-double", "minimum", "minimum-half")` | Allowed `phase` values (MNE). |

## Class `filter_config(generic_config)`

`@dataclass`. The `filter_type` selects the kind of filter and dictates which cutoff(s) are required; all frequencies must be strictly positive.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `filter_type` | `str` | `"bandpass"` | One of `FILTER_TYPES`. Lower-cased in `__post_init__`. |
| `l_freq` | `float \| None` | `None` | Low transition-band edge (high-pass cutoff). Required for `highpass` and one edge of `bandpass`/`bandstop`; must be unset for `lowpass`. |
| `h_freq` | `float \| None` | `None` | High transition-band edge (low-pass cutoff). Required for `lowpass` and one edge of `bandpass`/`bandstop`; must be unset for `highpass`. |
| `freqs` | `float \| list[float] \| None` | `None` | Frequencies to remove; required for `notch`, ignored otherwise. |
| `method` | `str` | `"iir"` | Backend `"iir"` (moabb default) or `"fir"`. Lower-cased. |
| `iir_params` | `dict \| None` | `None` | IIR design params forwarded to MNE (e.g. `{'order':4,'ftype':'butter'}`); `None` = MNE defaults. |
| `filter_length` | `str \| int` | `"auto"` | FIR filter length (MNE). Only relevant for `method="fir"`. |
| `l_trans_bandwidth` | `str \| float` | `"auto"` | Lower transition-band width (MNE). |
| `h_trans_bandwidth` | `str \| float` | `"auto"` | Upper transition-band width (MNE). |
| `phase` | `str` | `"zero"` | Filter phase (MNE), one of `FILTER_PHASES`. |
| `fir_window` | `str` | `"hamming"` | FIR window (MNE). Only relevant for `method="fir"`. |
| `fir_design` | `str` | `"firwin"` | FIR design (MNE). Only relevant for `method="fir"`. |
| `picks` | `str \| None` | `"data"` | Channels to filter (MNE `picks`); moabb uses `"data"`. |

## `__post_init__` — validation

Runs in three blocks and raises `ValueError` on any violation:

1. **Categorical normalization/checks**: `filter_type` and `method` are `str(...).lower()`-normalized and checked against their allowed tuples; `phase` is checked (not lowered) against `FILTER_PHASES`.
2. **Cutoff positivity**: any provided `l_freq`/`h_freq` must be `> 0`.
3. **Type-specific required cutoffs** (via the private `_require`):
   - `bandpass` → needs `l_freq` and `h_freq` with `l_freq < h_freq`.
   - `bandstop` → needs `l_freq` and `h_freq` with `l_freq > h_freq`.
   - `lowpass` → needs `h_freq`; `l_freq` must be `None`.
   - `highpass` → needs `l_freq`; `h_freq` must be `None`.
   - `notch` → needs `freqs`; every frequency (scalar or in the list/tuple) must be positive.

### private `_require(name)`
Raises `ValueError` if the cutoff attribute `name` (`"l_freq"` or `"h_freq"`) is `None` for the current filter type. A small helper to keep the type-specific checks terse.

## Gotchas

- `bandstop` deliberately requires `l_freq > h_freq` (reversed relative to `bandpass`), matching MNE's convention for band-stop edges.
- Only `filter_type` and `method` are lower-cased; `phase` and `fir_*` are checked/stored verbatim.
- This module imports neither MNE nor moabb — it only names their parameters; actual filtering happens in the dataset layer.
