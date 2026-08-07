# `jade.dataset.support_plot_freq`

Source: `src/jade/dataset/support_plot_freq.py`

Frequency-domain support for the plots defined in [`dataset_time`](./dataset_time.md). Its job is to move a 1-D EEG signal from the time domain to a **power spectral density (PSD)** estimate via the Welch method, with a three-backend fallback chain so the frequency plots keep working even without the optional `scipy` dependency. Every backend returns plain numpy arrays, ready for matplotlib.

Eager imports: `numpy`, `warnings`. `scipy`/`torch` are imported **inside** the functions that use them.

## The three backends

| Backend | Based on | Role |
|---------|----------|------|
| `scipy` | `scipy.signal.welch` | **default**; used when `method='welch'` and scipy is importable |
| `torch` | `torch.fft` | automatic fallback when scipy is missing; also selectable explicitly |
| `numpy` | `numpy.fft` | last-resort backup when the torch backend raises; also selectable explicitly |

Fallback order at call time: `welch (scipy)` → `torch` → `numpy`. Each downgrade emits a `warnings.warn`.

---

## `check_freq_config(freq_config=None) -> dict`

Validate a frequency-config dict and fill missing keys with defaults. Keys mirror `scipy.signal.welch` so all three backends behave the same. **Never mutates the input.** `None`/empty → a fresh all-defaults dict.

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `method` | str | `'welch'` | backend: `'welch'` (scipy), `'torch'`, or `'numpy'`. `'welch'` auto-falls-back to torch then numpy. |
| `window` | str | `'hann'` | segment window: `'hann'`, `'hamming'`, `'blackman'`, `'bartlett'`, `'boxcar'` (rectangular). |
| `nperseg` | int or None | `None` | segment length; `None` → whole signal as one segment (a periodogram). |
| `noverlap` | int or None | `None` | overlap; `None` → `nperseg // 2`. |
| `nfft` | int or None | `None` | FFT length; `None` → `nperseg`. |
| `detrend` | str or bool | `'constant'` | `'constant'` (remove mean), `'linear'` (remove linear trend), or `False` (none). |
| `scaling` | str | `'density'` | `'density'` (V²/Hz) or `'spectrum'` (V²). |

Behavior / gotchas:
- Keys `nperseg`, `noverlap`, `nfft` keep an explicitly-provided `None` (they treat `None` as a valid value); all other keys have a missing/`None` entry replaced by the default.
- **Raises `ValueError`** if `method` is not one of the three, or `scaling` is not `density`/`spectrum`.

---

## `compute_frequency_spectrum(signal, fs=1, freq_config=None) -> (freqs, psd)`

Estimate the PSD of a 1-D signal by the Welch method. This is the single entry point used by `EEG_Dataset._plot_freq_on_ax`.

| Param | Type | Meaning | Default |
|-------|------|---------|---------|
| `signal` | `numpy.ndarray` / array_like | one channel of one trial (1-D after squeeze). | required |
| `fs` | float | sampling frequency; `1` = normalized frequency. | `1` |
| `freq_config` | dict or None | see `check_freq_config`. | `None` |

**Returns** `(freqs, psd)` — both plain numpy arrays (one-sided). When `fs=1`, `freqs` are normalized frequencies. `psd` unit is V²/Hz for `scaling='density'`, V² for `'spectrum'`.

Behavior:
1. `check_freq_config(freq_config)`; coerce `signal` to a 1-D float numpy array via `np.asarray(...).squeeze()`. **Raises `ValueError`** if not 1-D.
2. Resolve segmentation against the signal length `n_samples`:
   - `nperseg = n_samples` if it is `None` **or greater than** `n_samples` (clamped to the signal).
   - `noverlap = nperseg // 2` if `None`; `nfft = nperseg` if `None`.
3. Dispatch on `method`:
   - `'welch'` → try `scipy.signal.welch(..., return_onesided=True)`. On `ImportError` → warn, set `method='torch'`.
   - `'torch'` → try `_welch_torch(...)`. On **any** `Exception` → warn (with the error text), set `method='numpy'`.
   - `'numpy'` → `_welch_numpy(...)` (also the explicit-request path and the final backup).

With the default `nperseg=None` the whole signal is a single segment, so the estimate reduces to a plain periodogram.

---

## Internal helpers

### `_get_window(window, nperseg) -> numpy.ndarray`
Builds a **periodic** window of length `nperseg` (as scipy does for spectral analysis): samples `nperseg + 1` points of the symmetric window and drops the last. `'boxcar'`/`'rectangular'`/`'none'` → `np.ones(nperseg)`. `'hann'`/`'hamming'`/`'blackman'`/`'bartlett'` → the matching `np.*(nperseg+1)[:-1]`. Anything else → `ValueError`.

### `_detrend_segment(segment, detrend)` (numpy)
`False`/`None` → unchanged; `'constant'` → subtract mean; `'linear'` → subtract a least-squares line (`np.polyfit`/`np.polyval`, deg 1); else `ValueError`.

### `_welch_numpy(signal, fs, window, nperseg, noverlap, nfft, detrend, scaling) -> (freqs, psd)`
Pure-numpy Welch, scaled to match `scipy.signal.welch`:
- Window via `_get_window`. Scale: `density` → `1/(fs·Σw²)`; `spectrum` → `1/((Σw)²)`; else `ValueError`.
- Step `= nperseg - noverlap` (guarded to `nperseg` if `<= 0`). Segment starts `range(0, len(signal)-nperseg+1, step)` (falls back to `[0]` if empty — a safety net, since `nperseg` is clamped).
- Each segment: `_detrend_segment` → multiply by window → `np.fft.rfft(n=nfft)` → accumulate `|X|²·scale`. Average over segments.
- **One-sided correction:** even `nfft` → double `psd[1:-1]` (keep DC and Nyquist); odd `nfft` → double `psd[1:]` (no Nyquist bin). `freqs = np.fft.rfftfreq(nfft, d=1/fs)`.

### `_detrend_segment_torch(segment, detrend)` (torch)
Torch counterpart of `_detrend_segment`. `'linear'` uses `torch.linalg.lstsq` on a `[x, 1]` design matrix. Imports `torch` locally.

### `_welch_torch(signal, fs, window, nperseg, noverlap, nfft, detrend, scaling) -> (freqs, psd)`
Torch counterpart of `_welch_numpy`, step-for-step identical so results match. Notes:
- Uses `float64` throughout to stay close to scipy/numpy precision.
- Builds the window in numpy (`_get_window`) then moves it to torch, reusing the exact same window definition.
- Uses `torch.fft.rfft` / `torch.fft.rfftfreq`; same one-sided doubling rule.
- **Returns `.numpy()` arrays** so matplotlib can plot directly (and so the public function's contract — always numpy — holds regardless of backend).

## Design decisions / gotchas
- **scipy is optional** for frequency plots; the fallback chain guarantees a PSD estimate in a bare `numpy`+`torch` environment. The torch backend catches *any* `Exception` (not just `ImportError`), so a torch-runtime problem still degrades gracefully to numpy.
- The numpy and torch Welch implementations replicate scipy's periodic-window, scaling, and one-sided-doubling conventions specifically so the three backends are interchangeable — do not "simplify" one without matching the others.
- No `max == min` / zero-variance guards; a constant segment with `detrend='constant'` yields an all-zero segment (PSD ≈ 0), which is expected but worth knowing.
