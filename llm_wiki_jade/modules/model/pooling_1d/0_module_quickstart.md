# `jade.model.pooling_1d` — Sub-package Quickstart

> **Read this first.** Anti-aliased 1D decimation ("pooling") layers for EEG signals, built as a controlled ablation study. After this page you understand the whole sub-package without opening source. Background DSP theory: `other/antialiased_pooling_theory.pdf`.

## The core idea

A pooling layer with kernel size `M` and stride `M` is **decimation**: it reduces the sampling rate by a factor `M`. By Nyquist–Shannon, any decimation must be preceded by a **low-pass (anti-aliasing) filter** — otherwise every spectral component above the new Nyquist `π/M` is *folded* back into the retained band and becomes indistinguishable from genuine low-frequency content (aliasing).

Every operator here follows the canonical two-step decomposition (Zhang, ICML 2019, *"Making Convolutional Networks Shift-Invariant Again"*):

```
pooling  =  dense FIR filtering (stride 1)  →  subsampling (stride M)
```

and the arms differ **only** in the low-pass filter `h` used in step 1. This is what makes the comparison airtight: all arms of an ablation differ in exactly one designed object, `h`.

## The family

| Layer | Filter `h` | Role | Page |
|---|---|---|---|
| `avg_pool_1d` | boxcar / moving average, length `M` | **baseline** (usual default) | [avg_pool](avg_pool.md) |
| `blur_pool_1d` | binomial (Pascal) / Gaussian / boxcar | fixed heuristic (1D BlurPool) | [blur_pool](blur_pool.md) |
| `sinc_pool_1d` | windowed-sinc, Kaiser window | designed "principled ceiling" | [sinc_pool](sinc_pool.md) |
| `dwt_pool_1d` | orthogonal wavelet analysis low-pass | filter-bank band-split | [dwt_pool](dwt_pool.md) |
| `max_pool_1d` | *none* (nonlinear) | un-anti-aliased reference | [max_pool](max_pool.md) |

**Provably nested family:** average pooling *is* the boxcar special case, so `avg_pool_1d` and `blur_pool_1d(kernel='boxcar', kernel_size=M)` are numerically identical (up to boundary handling). `self_test()` asserts this.

`max_pool_1d` is the odd one out: the max operator is nonlinear, has no frequency response, and can *create* new high-frequency content before subsampling — so it both fails to suppress out-of-band energy and manufactures more. It is included only so the ordering (max < avg < properly filtered) can be *demonstrated* rather than assumed.

## Invariants shared by every layer

- **Tensor convention:** consume and produce `(B, D, T)` (batch, depth/feature channels, time). Filtering is **depthwise** — the same `h` applied independently to all `D` channels.
- **Output length is exactly `T // M`**, matching `nn.AvgPool1d(M, stride=M)`. Deliberate: swapping the pooling operator never changes a downstream tensor shape, so classifier head, parameter count and receptive field stay fixed across an ablation.
- **Zero trainable parameters** in every layer. The kernel is a non-persistent buffer (moves with `.to()`/`.half()`/`.double()`, excluded from `state_dict()`).
- **Unit DC gain by default** (`sum(h) == 1`, config `gain='dc'`). Critical because EEGNet has no normalization layer *after* pooling (order is `BatchNorm → activation → pool → dropout`); a different DC gain would rescale all activations and confound the comparison with a change in effective learning rate / init scale.

## The central methodological warning (baked into the code)

Do **not** port the standard 2D BlurPool kernels (`[1,2,1]`, `[1,4,6,4,1]`) to a layer with large `M`. Those kernels have *fixed* bandwidth, but the required cutoff `π/M` shrinks as `M` grows. At `M=8` a length-3 binomial kernel passes ~67% of its energy above the post-decimation Nyquist vs ~22% for a plain boxcar — **three times worse than the average pooling it was meant to improve on**. Therefore:
- default kernel lengths **scale with M** (`blur`: `4M+1`; `sinc`: `8M+1`),
- `stopband_leakage()` is the headline figure of merit,
- `warn_if_worse_than_boxcar=True` fires a `RuntimeWarning` at construction if a config is worse than boxcar.

## Figures of merit (all on the *effective* single-stage kernel)

Every FIR layer inherits frequency-domain diagnostics from the base class:

| Metric | Meaning | Better |
|---|---|---|
| `stopband_leakage()` | fraction of kernel energy above `π/M` (the folded region) | lower (0 = perfect) |
| `stopband_attenuation_db()` | min attenuation over the stopband | higher |
| `passband_ripple_db()` | in-band ripple (or, for monotone kernels, in-band droop) | lower |
| `group_delay_samples()` | temporal centre of mass vs kernel centre; 0 for symmetric kernels | 0 for linear phase |
| `summary()` | all of the above in one dict | — |

## Public API and structure

- **Factory:** `get_pooling_layer(config)` — single entry point; `pooling_type='identity'` or a decimation factor of 1 → `nn.Identity`. See [`__init__`](__init__.md).
- **Registry:** `POOLING_REGISTRY` (aliases → class).
- **Self-test:** `self_test()` verifies the study's invariants.
- **Runnable:** `python -m jade.model.pooling_1d` runs the self-test and prints a figures-of-merit table for the two EEGNet pooling stages (BCI IV-2a, 250 Hz). See [`__main__`](__main__.md).

```
pooling_1d/
├── __init__.py         factory + registry + self_test + motivation docstring
├── __main__.py         python -m entry point (self-test + FOM table)
├── support_pooling.py  base class _fir_decimation_pool_1d + kernel helpers + DSP math + functional core
├── avg_pool.py         avg_pool_1d   (boxcar baseline)
├── blur_pool.py        blur_pool_1d  (binomial / gaussian / boxcar)
├── max_pool.py         max_pool_1d   (nonlinear reference)
├── sinc_pool.py        sinc_pool_1d  (Kaiser windowed-sinc)
└── dwt_pool.py         dwt_pool_1d + WAVELET_FILTERS (orthogonal wavelet bank)
```

## Dependencies & connection to the rest of `jade`

- **External:** `torch` + `numpy` (always); `scipy.signal` (lazy, only `sinc_pool_1d`); `pywt` (lazy, only for `dwt_pool_1d` wavelets outside the built-in `WAVELET_FILTERS`).
- **Config aliases** (`pool_size`, `factor`, `p_kernel` for the decimation factor) are chosen to accept EEGNet-style config keys like `p_kernel_1` unchanged — designed so a layer can be dropped into an existing config dict.
- These layers are **not yet used** by any model in `jade.model` (EEGNet uses `nn.AvgPool2d`). They are standalone research infrastructure; the planned consumer is an anti-aliased EEGNet variant.
