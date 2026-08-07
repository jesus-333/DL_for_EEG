# `pooling_1d/__main__.py` — `python -m jade.model.pooling_1d`

Source: `src/jade/model/pooling_1d/__main__.py`

Runnable entry point. Reproduces what running the original single-file module directly did: it runs [`self_test()`](__init__.md) and then prints a **figures-of-merit table** for the two EEGNet pooling stages (BCI IV-2a dataset, 250 Hz sampling). Imports `warnings` plus `get_pooling_layer` and `self_test` from the package.

Guarded by `if __name__ == "__main__":` (`# pragma: no cover`).

## What it prints

For each decimation factor `M in (4, 8)` it sets the stage input rate (`fs_in = 250.0` Hz for `M=4`, `62.5` Hz for `M=8` — i.e. the second stage sees the rate after the first stage) and prints a table over a fixed list of ablation arms:

| Arm label | Config |
|---|---|
| `avg_pool (boxcar)` | `{pooling_type: avg}` |
| `blur_pool binomial L=3 (2D default)` | `{pooling_type: blur, kernel_size: 3, warn_if_worse_than_boxcar: False}` |
| `blur_pool binomial L=4M+1` | `{pooling_type: blur}` (default length) |
| `blur_pool gaussian L=4M+1` | `{pooling_type: blur, kernel: gaussian}` |
| `sinc_pool Kaiser N=8M+1` | `{pooling_type: sinc}` (default length) |
| `sinc_pool 60dB/25% spec (capped)` | `{pooling_type: sinc, stopband_attenuation_db: 60, transition_width: 0.25, max_taps: 61}` |
| `dwt_pool db4` | `{pooling_type: dwt, wavelet: db4}` |
| `dwt_pool sym4` | `{pooling_type: dwt, wavelet: sym4}` |

Each arm gets `decimation_factor = M` merged in, is built with `get_pooling_layer` (inside a `warnings.catch_warnings()` that ignores `RuntimeWarning`, so the deliberately-bad `L=3` arm does not spam the boxcar warning), then its `layer.summary()` drives the row.

## Columns

| Column | Source | Meaning |
|---|---|---|
| `M` | `summary["decimation_factor"]` | decimation factor. |
| `L` | `summary["kernel_length"]` | effective kernel length. |
| `leak%` | `summary["stopband_leakage_pct"]` | **primary metric** — % of filter energy above `π/M` folded back by subsampling (lower better). |
| `att@pi/M` | `summary["stopband_attenuation_db"]` | attenuation measured *at* the folding frequency `π/M`; reads conservatively for `cutoff_scale < 1` (see [support_pooling](support_pooling.md#stopband_attenuation_db)). |
| `delay` | `summary["group_delay_samples"]` | group delay in input samples. |
| `minT` | computed as `(kernel_length - 1) // 2 + 1` | shortest input usable with reflect padding (reflect needs `pad < T`, with `pad = (L-1)//2`). |

The `max_taps: 61` cap on the spec-driven sinc arm is there because an uncapped 60 dB / 25% design asks for 259 taps at `M=8`, which would exceed the 250-sample stage-2 input.
