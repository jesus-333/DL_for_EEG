# `pooling_1d/dwt_pool.py` — `dwt_pool_1d` + `WAVELET_FILTERS`

Source: `src/jade/model/pooling_1d/dwt_pool.py`

Decimation by the **low-pass branch of an orthogonal wavelet filter bank** (DWT). Mechanistically distinct from the other FIR layers: rather than *designing* a low-pass to approximate a brick wall, the DWT uses a perfect-reconstruction quadrature-mirror filter pair. Subclass of [`_fir_decimation_pool_1d`](support_pooling.md); `_kind = "wavelet"`. Imports `math`, `numpy`, `torch`, and the helpers `_as_decimation_factor`, `_filter_and_subsample`, `_get`, `_upsample_zeros`.

## DSP idea

A wavelet defines a filter pair `(h low-pass, g high-pass)` forming a two-channel perfect-reconstruction bank: the two subbands together lose no information, so the aliasing from decimating each branch by 2 is exactly cancelled when they are recombined. Keeping only the approximation (low-pass) branch is a **principled band-split** rather than an ad-hoc blur — the point WaveCNet makes against BlurPool's Pascal-triangle kernels being "ad hoc with no theoretical justification".

Decimation by `M` is obtained by cascading `log2(M)` single-level (filter, ↓2) stages, so **`M` must be a power of two**.

## `WAVELET_FILTERS` (module-level dict)

Built-in orthonormal **reconstruction** low-pass (scaling) filters `rec_lo`, in the standard convention:
- `sum(h) = sqrt(2)` (DC gain), `sum(h²) = 1` (unit energy), `sum_n h[n]·h[n+2k] = 0` for `k ≠ 0` (double-shift orthogonality). Verified numerically to ~1e-12.
- The analysis (decomposition) filter is the time reverse `dec_lo = rec_lo[::-1]`.

| Name | Taps | Notes |
|---|---|---|
| `haar` / `db1` | 2 | 2-tap boxcar → very weak anti-aliasing; a deliberate lower-bound / sanity arm, not a serious candidate. |
| `db2` | 4 | Daubechies-2. |
| `db3` | 6 | Daubechies-3. |
| `db4` | 8 | **usual default** — good compromise of attenuation vs length; 4 vanishing moments; asymmetric. |
| `sym4` | 8 | "least asymmetric" 8-tap wavelet: nearly linear phase → less temporal smearing of event-related features (ERPs, QRS). |

## `class dwt_pool_1d`

### `__init__(self, config)`

Resolves the decimation factor *before* calling `super().__init__` (the base builds the kernel, but here the kernel is a single **stage** while `M` describes the whole cascade). Validates `M` is a power of two (`factor & (factor-1) == 0`); otherwise `ValueError` (suggests `sinc_pool_1d`/`blur_pool_1d` for other factors). Sets `self.levels = log2(M)` (0 when `M == 1`).

### Config

| Key | Type | Meaning | Default |
|---|---|---|---|
| `decimation_factor` | int or `(1, M)` | factor `M`, **must be a power of two**. | **required** |
| `wavelet` | str | one of `WAVELET_FILTERS`; or, if PyWavelets is installed, any `pywt` orthogonal wavelet name. | `db4` |
| shared keys | — | `padding_mode`, `phase`(*), `gain`, `warn_if_worse_than_boxcar`. | — |

(*) The cascade uses **offset 0** at every level (the single-stage `center` offset for `M=2` is `(2-1)//2 = 0` anyway, so it is consistent with the FIR layers).

### `_build_kernel(self, config)`

Reads `wavelet` (stored as `self.wavelet_name`). If in `WAVELET_FILTERS`, uses those coefficients; else lazily imports `pywt` and uses `pywt.Wavelet(name).rec_lo` (unknown name without PyWavelets → `ValueError` listing the built-ins). **Returns the time reverse `rec_lo[::-1]`** so that `F.conv1d`'s cross-correlation implements true *convolution* with `rec_lo` — matching the standard DWT analysis convention (`pywt`'s `dec_lo`). Note the built-in `rec_lo` have DC gain `sqrt(2)`, but the base class re-normalizes to unit DC gain by default (`gain='dc'`).

### `forward(self, x)`

Validates 3D input, then applies `self.levels` successive `_filter_and_subsample(..., factor=2, offset=0, padding_mode=...)` stages. Input `(B, D, T)` → output `(B, D, T // M)`.

### `effective_kernel(self) -> np.ndarray`

Collapses the multi-level cascade into one equivalent impulse response via the **noble identity** (`↓2 then filter h` ≡ `filter h↑2 then ↓2`):

```
h_eff = h * (h↑2) * (h↑4) * ...
```

so the whole cascade is a single FIR followed by decimation by `2**levels`. This is what makes the wavelet arm directly comparable to the FIR arms — every inherited analysis method operates on `h_eff`. Implemented by convolving `h` with `_upsample_zeros(h, 2**level)` for `level = 1..levels-1` (returns `h` directly if `levels <= 1`). Verified: cascade and equivalent single-stage agree to ~1e-16 under circular padding (where the equivalence is exactly LTI), using sampling offset `(len(h_eff)-1)//2 - ((L-1)//2)·(2**levels - 1)` (this exact check is `self_test()` #4).

### `extra_repr(self)`

`"M={} ({} levels), wavelet={}, L_stage={}, L_eff={}, padding_mode={}, gain={}"`.

## Trade-offs worth knowing

- **Daubechies filters are asymmetric** → nonlinear phase → transients smeared asymmetrically in time. Prefer `wavelet='sym4'` when timing matters; `group_delay_samples()` quantifies it (nonzero here, unlike the symmetric FIR arms).
- **`haar`** is a 2-tap boxcar — a deliberately weak filter (lower bound), not a serious candidate.
- The cascade's effective filter is longer and sharper than one stage: at `M=8` the db4 cascade has a ~50-tap equivalent response with ~9% leakage — between a mid-length binomial and a good Kaiser design.

## Shapes

Input `(B, D, T)` → output `(B, D, T // M)`. Zero trainable parameters.

## Multichannel variant

`dwt_pool_1d_multichannel(config)` wraps `dwt_pool_1d` for 4D EEG tensors `(B, D, C, T)`, applying it channel-wise (see [`_multichannel_pool_1d`](support_pooling.md#multichannel-wrapper)). Same `config` (decimation factor must be a power of two); output `(B, D, C, T // M)`.
