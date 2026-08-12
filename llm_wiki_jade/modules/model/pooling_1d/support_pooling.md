# `pooling_1d/support_pooling.py` — base class, functional core, DSP math

Source: `src/jade/model/pooling_1d/support_pooling.py`

The heaviest file in the sub-package (~730 lines). Everything the individual pooling layers share:
- kernel/utility helpers (`_get`, `_as_decimation_factor`, `_binomial_kernel`, `_gaussian_kernel`, `_upsample_zeros`),
- the functional core `_filter_and_subsample` and figure-of-merit helper `_leakage_of`,
- the base class **`_fir_decimation_pool_1d`** implementing `pooling = dense FIR filtering → subsampling`,
- the base class **`_multichannel_pool_1d`** for the channel-wise 4D wrappers.

Imports `warnings`, `numpy`, `torch`, `torch.nn.functional as F`, `torch.nn`. Subclasses only supply the kernel; the base class owns padding, depthwise application, phase/offset, output-length guarantees, and the frequency-domain analysis utilities.

---

## Small helpers

### `_get(config, key, default=None, aliases=()) -> Any`

Fetch `key` from `config`, accepting alternative spellings; returns the **first non-`None`** value found among `(key, *aliases)`, else `default`. Rationale: different configs spell the decimation factor differently (`pool_size`, `factor`, `p_kernel`), so a pooling layer can be dropped into an existing config dict without renaming keys.

### `_as_decimation_factor(value) -> int`

Normalize a raw factor to a positive int. Accepts `4`, `(4,)`, `(1, 4)`, `[1, 4]`.

| Input form | Result |
|---|---|
| `int` | that int |
| length-1 tuple/list | its single element |
| length-2 tuple/list `(1, M)` | `M` (height must be 1 — purely temporal 2D pooling) |

**Raises `ValueError`** if a length-2 form has height ≠ 1 (it would pool across the non-temporal axis, which these 1D layers do not do), for length > 2, or if the resulting factor `< 1`. The `(1, M)` form is accepted because EEGNet configs specify 2D pooling kernels `(1, M)` acting on time only.

### `_binomial_kernel(length) -> np.ndarray`

Binomial ("Pascal row") kernel, normalized to sum 1, built by repeatedly convolving `[1,1]` with itself (`length-1` times). Frequency response has closed form `|H(w)| = |cos(w/2)|**(length-1)`: monotonically decreasing on `[0, π]`, **no sidelobes, no passband ripple** — the qualitative improvement over the boxcar — at the price of a slow roll-off. Converges to a Gaussian as `length` grows (CLT). Raises `ValueError` if `length < 1`.

### `_gaussian_kernel(length, sigma=None) -> np.ndarray`

Truncated Gaussian, normalized to sum 1. If `sigma` is `None` it defaults to `max(length/6, 1e-3)`, i.e. truncation at ±3σ (keeps truncation error / spurious sidelobes negligible). Gaussian frequency response: strictly positive, no sidelobes, slowest roll-off of the fixed kernels. Raises `ValueError` if `length < 1`.

### `_upsample_zeros(h, factor) -> np.ndarray`

Insert `factor - 1` zeros between consecutive taps of `h` (output length `(len(h)-1)*factor + 1`). Implements the **noble identity** used to collapse a multi-stage decimating filter bank into one equivalent single-stage filter (see [`dwt_pool_1d.effective_kernel`](dwt_pool.md)). Upsampling a filter by `L` compresses its frequency response by `L`. `factor == 1` returns a copy.

---

## `class _fir_decimation_pool_1d(nn.Module)`

Base class: `pooling = dense FIR filtering (stride 1) → subsampling (stride M)`. Class attribute `_kind: str = "fir"` (subclasses override for nicer `__repr__`).

### Shared configuration keys

| Key | Type | Meaning | Default |
|---|---|---|---|
| `decimation_factor` | int or `(1, int)` | factor `M` reducing the time axis. Aliases: `pool_size`, `factor`, `p_kernel`. **`kernel_size` is deliberately NOT an alias** — it always means filter length. | **required** |
| `padding_mode` | str | how to extend the signal so the stride-1 filtered output keeps length `T`. One of `reflect`, `replicate`, `constant` (zeros), `circular`. | `reflect` |
| `phase` | `'center'`, `'first'`, or int | which sample of each block of `M` to keep. `center` → offset `(M-1)//2` (reproduces `nn.AvgPool1d` alignment); `first` → offset 0; int → explicit offset in `[0, M)`. | `center` |
| `gain` | `'dc'`, `'orthonormal'`, `'none'` | kernel normalization: `dc` → `sum(h)=1` (unit DC gain, comparable with an average); `orthonormal` → `‖h‖₂=1` (energy-preserving); `none` → leave as built. | `dc` |
| `warn_if_worse_than_boxcar` | bool | warn at construction if the filter passes *more* energy above `π/M` than a boxcar of length `M`. | `True` |

**Why `reflect` is default:** EEG trials are short; zero-padding would inject an artificial step at both ends whose broadband spectrum is exactly the high-frequency content the filter is meant to suppress.

### `__init__(self, config)` — flow

1. **Decimation factor:** read via `_get(...)` (with aliases); `KeyError` if missing (message reminds that `kernel_size` is *not* an alias). Normalized via `_as_decimation_factor` into `self.decimation_factor`.
2. **Generic options:** validate `padding_mode` (`ValueError` if unsupported), `gain` (`ValueError` if unsupported), resolve `phase` into `self.offset` (int `phase` outside `[0, M)` → `ValueError`).
3. **Build the kernel:** `kernel = np.asarray(self._build_kernel(config)).ravel()`; empty kernel → `ValueError`; then `self._normalize(kernel)`.
4. **Register the kernel** as a **non-persistent buffer** (`register_buffer("kernel", ..., persistent=False)`): it follows `.to(device)`/`.float()`/`.half()` like a parameter but is excluded from `state_dict()` — keeps checkpoints free of constants and avoids shape-mismatch on reload with a different kernel length.
5. **Sanity check:** if `warn_if_worse_than_boxcar` and `M > 1`, call `_check_against_boxcar()`.

### `_build_kernel(self, config) -> np.ndarray` (hook)

Abstract — raises `NotImplementedError`. Each subclass builds its low-pass kernel here.

### `_normalize(self, kernel) -> np.ndarray`

Applies the `gain` convention:
- `dc` → `kernel / kernel.sum()`; if `|sum| < 1e-12` raises `ValueError` ("not a low-pass filter" — cannot DC-normalize a near-zero-DC kernel).
- `orthonormal` → `kernel / sqrt(sum(kernel**2))`.
- `none` → unchanged.

### `_check_against_boxcar(self)`

Compares `self.stopband_leakage()` against `_leakage_of(np.ones(M)/M, M)` (the boxcar). If the layer leaks more, emits a `RuntimeWarning` (`stacklevel=3`) explaining the kernel is too short for this `M` (required cutoff `π/M`), suggesting `length >= 4*M+1` or `sinc_pool_1d`, and how to silence it (`warn_if_worse_than_boxcar=False`). `avg_pool_1d` overrides this to a no-op (it *is* the boxcar).

### `forward(self, x) -> torch.Tensor`

Validates `x` is 3D (`ValueError` otherwise) and delegates to `_filter_and_subsample(x, kernel=self.kernel, factor=self.decimation_factor, offset=self.offset, padding_mode=self.padding_mode)`.

- **Input:** `(B, D, T)`. **Output:** `(B, D, T // M)`.

---

## Analysis utilities (no autograd; for figures/tables)

All operate on the **effective kernel** (`effective_kernel()`), so the wavelet cascade is analyzed on equal footing with the FIR layers.

### `effective_kernel(self) -> np.ndarray`

The single-stage impulse response equivalent to the layer, as a float64 NumPy array. For FIR layers this is just `self.kernel`; `dwt_pool_1d` overrides it to collapse its multi-level cascade via the noble identity.

### `frequency_response(self, n_fft=8192) -> (w, H)`

`H = np.fft.rfft(effective_kernel, n=n_fft)`; `w` is linear in `[0, π]` matching `H.size`. Returns normalized angular frequency `w` and complex response `H`. To convert to Hz at input rate `fs`: `f = w / (2π) * fs`.

### `plot_frequency_response(self, n_fft=8192, db=True, unwrap_phase=True, fs=None, axs=None) -> (fig, axs)`

Plots the layer's frequency response — **magnitude** (top) and **phase** (bottom) — on two stacked axes, on top of `frequency_response()`. Every FIR pooling layer (avg/blur/sinc/dwt) is a linear filter, so it inherits this from the base class. The nonlinear `max_pool_1d` **overrides it to raise `NotImplementedError`** (no frequency response — see [`max_pool.md`](max_pool.md)).

- `db` — magnitude in dB (`20·log10|H|`, clamped at `1e-12` before the log) vs linear `|H|`.
- `unwrap_phase` — `np.unwrap(np.angle(H))` vs the wrapped phase.
- `fs` — when given, the x-axis is Hz (`f = w/(2π)·fs`); when `None` it is the normalized angular frequency `[0, π]`.
- `axs` — draw on two provided axes, else a new 2-row figure is created.
- A dashed red line marks the **post-decimation Nyquist `π/M`** (the folding frequency): everything to its right is aliased back by the subsampling.

The actual drawing lives in the module-level helper **`_plot_frequency_response(w, H, decimation_factor=None, db=True, unwrap_phase=True, fs=None, axs=None, title=None)`** (lazy `import matplotlib.pyplot` so importing the pooling layers never pulls in matplotlib). The multichannel wrapper `_multichannel_pool_1d` exposes a thin `plot_frequency_response(...)` that **delegates to the wrapped 1D layer** (`self.pool`).

### `stopband_leakage(self, n_fft=8192) -> float`

**The headline figure of merit.** Fraction of the filter's energy *above* the post-decimation Nyquist:

```
leakage = ∫_{π/M}^{π} |H(w)|² dw  /  ∫_0^{π} |H(w)|² dw
```

Exactly the portion of the response that subsampling folds back into the retained band. `0` = perfect anti-aliasing. A property of the *filter*, independent of signal (the complementary signal-dependent diagnostic — energy of actual activations above `π/M` — is suggested to log during training). Implemented via `_leakage_of`.

### `passband_ripple_db(self, passband_edge=0.8, n_fft=8192) -> float`

Peak-to-peak variation of `|H|` in dB over `[0, passband_edge · π/M]`. `passband_edge < 1` because a realizable filter needs a transition band. **Interpretation depends on the kernel:** for oscillating responses (boxcar, windowed-sinc) this is genuine ripple; for *monotone* responses (binomial, Gaussian) there is no oscillation and it instead measures in-band **droop** (how much in-band signal you throw away). The default Gaussian (σ = L/6) reads ~14 dB — it attenuates the top of its own passband by ~5×. Read alongside `stopband_leakage`: low leakage bought with high droop is not a better filter, just a narrower one. Returns `0.0` if the band has < 2 bins; `inf` if the min magnitude ≤ 0.

### `stopband_attenuation_db(self, stopband_edge=1.0, n_fft=8192) -> float` {#stopband_attenuation_db}

Minimum attenuation in dB (relative to DC) over `[stopband_edge · π/M, π]`. Larger is better. **Caveat:** with the default `stopband_edge=1.0` this measures from the folding frequency `π/M` *itself*. Any design whose cutoff sits *below* `π/M` (e.g. `sinc_pool_1d` with `cutoff_scale < 1`) still has its transition band straddling that point, so the number is the attenuation partway down the transition slope, **not** the true stopband floor — e.g. the default Kaiser design reads only ~9.5 dB while actually leaking just 1.3% of its energy. Pass `stopband_edge=1.2`+ to measure past the transition; prefer `stopband_leakage` as the headline metric. Returns `inf` if the band is empty or its peak ≤ 0.

### `group_delay_samples(self) -> float`

Centre of mass of `|h|` in input samples, relative to the kernel centre: `(Σ h·pos)/(Σ h) - (size-1)/2`. **Zero** for any symmetric (linear-phase) kernel — boxcar, binomial, Gaussian, windowed-sinc are all symmetric. **Non-zero** for Daubechies wavelets (asymmetric → asymmetric temporal smearing of transients). Report whenever event timing matters.

### `summary(self) -> dict`

All figures of merit in one dict: `layer`, `kind`, `decimation_factor`, `kernel_length`, `stopband_leakage_pct` (100× leakage), `stopband_attenuation_db`, `passband_ripple_db`, `group_delay_samples`, `dc_gain` (`effective_kernel().sum()`). Drives the [`__main__`](__main__.md) table.

### `extra_repr(self) -> str`

`"M={}, L={}, padding_mode={}, offset={}, gain={}"` — appears in `torch.nn.Module.__repr__`. (`sinc_pool_1d` and `dwt_pool_1d` extend it.)

---

## Shared functional core

### `_filter_and_subsample(x, kernel, factor, offset, padding_mode) -> torch.Tensor`

Depthwise FIR filtering at stride 1, then subsampling by `factor`. The single computational primitive used by every layer (and, per level, by the DWT cascade).

| Parameter | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | input `(B, D, T)`. |
| `kernel` | `torch.Tensor` | the 1D low-pass kernel (length `L`). |
| `factor` | int | subsampling factor. |
| `offset` | int | which sample of each block of `factor` to retain. |
| `padding_mode` | str | `reflect`/`replicate`/`constant`/`circular`. |

**Returns** `(B, D, T // factor)`.

**Internals / notes:**
- Centred padding `pad_left = (L-1)//2`, `pad_right = L-1-pad_left`, so the stride-1 filtered signal has length exactly `T`.
- **Reflect guard:** if `padding_mode == 'reflect'` and `max(pad_left, pad_right) >= T` → `ValueError` (reflect needs `pad < T`; suggests a shorter kernel or `replicate`).
- **Depthwise trick:** `x` is reshaped `(B, D, T) → (B*D, 1, T)` so a single-channel kernel serves all channels. Equivalent to a grouped conv with `groups=D` but avoids materializing `D` kernel copies and the contiguity requirement of grouped-conv weights.
- **Cross-correlation:** `F.conv1d` computes cross-correlation, not convolution. For symmetric kernels the two coincide; the asymmetric wavelet layer pre-flips its kernel to compensate.
- **Dtype/device:** kernel is cast to the input's dtype/device (`view(1,1,L)`), so it works under autocast / half / double.
- **Filter-then-subsample (not strided conv):** filtering is done at stride 1 and subsampling applied afterwards (`y[..., offset::factor]` then truncate to `n_out = T // factor`). This is ~`M`× more FLOPs but (a) keeps the two conceptual steps separate and inspectable, (b) makes the exact output length trivial to guarantee, (c) is negligible vs the rest of the network at these kernel sizes. Result is `.contiguous()`.
- `L == 1` skips padding entirely; `factor == 1` skips the subsampling slice.

### `_leakage_of(h, factor, n_fft=8192) -> float`

Stopband leakage of an arbitrary kernel (the math behind `stopband_leakage`): `rfft` the kernel, integrate `|H|²` over `w >= π/factor` divided by the total. Returns `0.0` if total power ≤ 0. Used both by the base class and by `_check_against_boxcar` (to score the boxcar reference).

---

## Multichannel wrapper

### `class _multichannel_pool_1d(nn.Module)`

Base class for applying a 1D pooling layer channel-wise to a **4D** multichannel EEG tensor `(B, D, C, T)` (batch, depth, EEG channels, time) — the shape used by [EEGNet](../EEGNet.md). The concrete `{name}_multichannel` wrappers (one per operator, defined in each pooling file) subclass this and just build the 1D layer to wrap.

`__init__(self, pool_1d: nn.Module)` stores the wrapped 1D layer as `self.pool`. `forward(x)`:

1. `ValueError` unless `x` is 4D.
2. Flatten `(B, D, C, T) → (B, D*C, T)` so the `(D, C)` axes become one depthwise axis.
3. Run `self.pool` once → `(B, D*C, T // M)`.
4. Restore `→ (B, D, C, T // M)`.

Because the wrapped 1D layer already pools every row of its second axis independently, flattening `(D, C)` makes it pool every `(depth, channel)` pair independently — **exactly equivalent to looping over channels, but vectorized** into a single call (asserted bit-identical by `self_test` check 6). No trainable parameters (inherited from the wrapped layer).

- **Input:** `(B, D, C, T)`. **Output:** `(B, D, C, T // M)`.

---

## Math notes at a glance

- **Aliasing / folding:** decimation by `M` reflects everything above `π/M` back into `[0, π/M]`; the low-pass filter's job is to remove that energy first. `stopband_leakage` measures exactly the residual.
- **Noble identity:** `(↓2 then filter h) ≡ (filter h↑2 then ↓2)`; applied repeatedly it collapses a cascade of decimating filters into one equivalent FIR (`_upsample_zeros` builds the `h↑` factors; `dwt_pool_1d.effective_kernel` uses it).
- **Linear phase:** symmetric kernels have zero group delay; the centred padding cancels the `(N-1)/2` FIR delay exactly, so outputs stay time-aligned across the family.
