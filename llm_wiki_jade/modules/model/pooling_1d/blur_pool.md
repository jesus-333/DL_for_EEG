# `pooling_1d/blur_pool.py` — `blur_pool_1d` (fixed heuristic kernel)

Source: `src/jade/model/pooling_1d/blur_pool.py`

Fixed-kernel anti-aliased pooling — the 1D analogue of Zhang's BlurPool (2019). Same structure as [average pooling](avg_pool.md), better kernel. Subclass of [`_fir_decimation_pool_1d`](support_pooling.md); `_kind` defaults to `"binomial"` but is overwritten in `_build_kernel` to the chosen family. Imports `numpy`, the base class, and the kernel helpers `_binomial_kernel`, `_gaussian_kernel`, `_get`.

## DSP idea

The binomial (Pascal-row) kernel has response `|H(w)| = |cos(w/2)|**(L-1)`: monotonically decreasing, **no sidelobes, no passband ripple** — a qualitative improvement over the boxcar's oscillating Dirichlet response. The cost is a slow roll-off: suppression is bought only through length `L`.

## THE ONE THING TO GET RIGHT — length must scale with `M`

`|cos(w/2)|**(L-1)` has a bandwidth set by `L` alone, while the required cutoff is `π/M`. The canonical 2D kernels `[1,2,1]` (L=3) and `[1,4,6,4,1]` (L=5) were chosen for stride-2 layers; reusing them at large `M` gives *worse* aliasing than plain averaging:

| | boxcar | L=3 | L=5 | longer |
|---|---|---|---|---|
| **M = 4** | 21.6% | 39.7% | 24.5% | L=9: 10.6% |
| **M = 8** | 22.4% | 67.5% | 56.5% | L=25: 17.0% |

(stopband leakage). Hence the default here is `kernel_size = 4*M + 1`, and the inherited `warn_if_worse_than_boxcar` fires if the config is worse than the boxcar. To reproduce the literal 2D BlurPool, pass `kernel_size=3` explicitly and expect the warning.

## Config

| Key | Type | Meaning | Default |
|---|---|---|---|
| `decimation_factor` | int or `(1, M)` | factor `M`. | **required** |
| `kernel_size` | int | kernel length `L`. Aliases: `blur_kernel_size`, `L`. Odd values keep the kernel symmetric about a sample → exactly linear phase, zero net delay. | `4*M + 1` |
| `kernel` | `{'binomial','gaussian','boxcar'}` | kernel family (see below). | `binomial` |
| `sigma` | float | Gaussian std-dev (only for `kernel='gaussian'`). | `kernel_size / 6` |
| shared keys | — | `padding_mode`, `phase`, `gain`, `warn_if_worse_than_boxcar`. | — |

## Kernel families (`_build_kernel`)

| `kernel` | Built by | Notes |
|---|---|---|
| `binomial` | `_binomial_kernel(L)` | Pascal row; standard BlurPool choice. |
| `gaussian` | `_gaussian_kernel(L, sigma)` | no sidelobes at all, slowest roll-off; high in-band droop at default σ. |
| `boxcar` | `np.ones(L)/L` | moving average; `blur_pool_1d(kernel='boxcar', kernel_size=M)` reproduces [`avg_pool_1d`](avg_pool.md), proving the family is **nested**. |

`_build_kernel` reads `kernel` and `kernel_size` via `_get`, sets `self._kind` to the family name, and raises `ValueError` for `kernel_size < 1` or an unknown `kernel`.

## Differences from the base class

Only `_build_kernel` is overridden; padding, subsampling, normalization, the boxcar warning, and all figures of merit are inherited.

## Shapes

Input `(B, D, T)` → output `(B, D, T // M)`. Zero trainable parameters.

## Multichannel variant

`blur_pool_1d_multichannel(config)` wraps `blur_pool_1d` for 4D EEG tensors `(B, D, C, T)`, applying it channel-wise (see [`_multichannel_pool_1d`](support_pooling.md#multichannel-wrapper)). Same `config`; output `(B, D, C, T // M)`.
