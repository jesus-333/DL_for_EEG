# `pooling_1d/sinc_pool.py` — `sinc_pool_1d` (windowed-sinc / Kaiser)

Source: `src/jade/model/pooling_1d/sinc_pool.py`

Windowed-sinc anti-aliasing filter — what a DSP engineer would actually use, and the **principled ceiling** arm of the ablation. Subclass of [`_fir_decimation_pool_1d`](support_pooling.md); `_kind = "windowed-sinc"`. Imports `numpy`, the base class, `_get`. **Needs SciPy** for filter design (`scipy.signal.firwin`, imported lazily inside `_build_kernel`).

## DSP idea

The ideal anti-aliasing filter for decimation by `M` is a brick wall at `w_c = π/M`, whose impulse response is the (doubly infinite) sinc `h_ideal[n] = (1/M)·sinc(n/M)`. Truncating it produces Gibbs oscillations (a fixed ~−21 dB first sidelobe regardless of length). Multiplying by a smooth **window** trades transition width for stopband attenuation and removes that floor. The **Kaiser** window is standard because its single parameter `beta` sets attenuation directly, via the classical formulas:

```
beta ≈ 0.1102·(A − 8.7)          for A > 50 dB
N    ≈ (A − 8) / (2.285·Δw)      taps
```

where `A` = desired stopband attenuation (dB) and `Δw` = transition width (rad/sample). Both are implemented (via `scipy.signal.kaiser_beta` and `scipy.signal.kaiserord`), so a filter can be specified the way a spec is actually written — "60 dB stopband, 10% transition" — instead of guessing a kernel length.

## Config

| Key | Type | Meaning | Default |
|---|---|---|---|
| `decimation_factor` | int or `(1, M)` | factor `M`. | **required** |
| `num_taps` | int | filter length `N`. Aliases: `kernel_size`, `N`, `L`. Forced odd (Type-I linear phase, exactly zero delay). | `8*M + 1` |
| `beta` | float | Kaiser shape parameter (used only when no window/spec overrides it). | `8.6` (~80 dB) |
| `stopband_attenuation_db` | float | target attenuation `A`; if given, `beta` derived from it. | optional |
| `transition_width` | float | transition-band width as a **fraction of the cutoff `π/M`** (0.2 → 20% of the cutoff). With `stopband_attenuation_db`, `num_taps` is derived via `kaiserord`. | optional |
| `max_taps` | int | safety cap on a *derived* `num_taps` (long filters cost FLOPs and, with reflect padding, need `T > N/2`). | optional |
| `cutoff_scale` | float | places the −6 dB cutoff at `cutoff_scale · π/M` instead of exactly `π/M`. Must be in `(0, 1]`. | `0.9` |
| `window` | str or tuple | any `scipy.signal.firwin` window (`'hamming'`, `('kaiser', 8.6)`, ...), for comparing window choices. | `('kaiser', beta)` |
| shared keys | — | `padding_mode`, `phase`, `gain`, `warn_if_worse_than_boxcar`. | — |

**`cutoff_scale` is the single most consequential knob** for signals with meaningful high-frequency content: a real filter's transition band straddles its cutoff, so putting the cutoff slightly below Nyquist keeps the whole transition inside the retained band at the price of attenuating the top ~10% of the passband. This is exactly the anti-aliasing / signal-loss trade-off the study should characterize. Set `1.0` for a textbook cutoff.

## `_build_kernel(self, config)` — design flow

1. **Lazy SciPy import** — `from scipy import signal as sps`; a clean `ImportError` (suggesting `blur_pool_1d`, whose kernels are closed-form) if SciPy is missing.
2. **Cutoff:** `M == 1` returns `np.ones(1)` (no-op). Validate `0 < cutoff_scale <= 1` (`ValueError`). `firwin` uses `fs=2.0` (units where `1.0 == Nyquist == π`), so the cutoff is `cutoff_scale / M`.
3. **Window:** if none given, derive `beta` from `stopband_attenuation_db` (`sps.kaiser_beta`) or use the `beta` config value, then `window = ('kaiser', beta)`. Stored on `self._design_window`.
4. **Length:** if `num_taps` not given: with both `stopband_attenuation_db` and `transition_width`, `width = transition_width · cutoff` and `sps.kaiserord` sizes `num_taps` (capped by `max_taps`); otherwise default `8*M + 1`. Then `num_taps` is forced odd (`+1` if even) → Type-I linear-phase FIR whose integer group delay `(N-1)/2` is cancelled exactly by the centred padding. `num_taps < 1` → `ValueError`.
5. **Design:** `h = sps.firwin(num_taps, cutoff, window=window, fs=2.0, pass_zero=True)`. Stores `self._num_taps`, `self._cutoff_normalized`.

## `extra_repr(self)`

Extends the base: appends `window=...` and `cutoff={:.4f}*Nyquist`.

## Gotcha — reading its metrics

Because the default `cutoff_scale=0.9` places the cutoff *below* `π/M`, `stopband_attenuation_db()` (measured at `π/M` by default) reads misleadingly low (~9.5 dB) even though `stopband_leakage()` is only ~1.3%. Prefer `stopband_leakage`, or pass `stopband_edge >= 1.2` to `stopband_attenuation_db`. See the [base class note](support_pooling.md#stopband_attenuation_db).

## Shapes

Input `(B, D, T)` → output `(B, D, T // M)`. Zero trainable parameters.

## Multichannel variant

`sinc_pool_1d_multichannel(config)` wraps `sinc_pool_1d` for 4D EEG tensors `(B, D, C, T)`, applying it channel-wise (see [`_multichannel_pool_1d`](support_pooling.md#multichannel-wrapper)). Same `config` (and, like it, needs SciPy); output `(B, D, C, T // M)`.
