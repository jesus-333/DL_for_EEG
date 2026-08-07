# `pooling_1d/avg_pool.py` — `avg_pool_1d` (boxcar baseline)

Source: `src/jade/model/pooling_1d/avg_pool.py`

Standard average pooling, written explicitly as **boxcar filter + subsample**. This is the **baseline** of the anti-aliasing study; expressing it in the same `_fir_decimation_pool_1d` framework as the others makes the comparison airtight — it differs from [`blur_pool_1d`](blur_pool.md) *only* in the choice of `h`.

Subclass of [`_fir_decimation_pool_1d`](support_pooling.md). `_kind = "boxcar"`. Imports `numpy` and the base class.

## DSP idea

The boxcar of length `M` has the closed-form (Dirichlet) response

```
H(w) = (1/M) · sin(wM/2) / sin(w/2) · exp(-j·w·(M-1)/2)
```

Its first zero sits exactly at the post-decimation Nyquist `w = 2π/M` — which is why averaging is not a *disaster* as an anti-aliasing filter. But its sidelobes decay only as `1/w` (first sidelobe ≈ −13 dB), so roughly **20% of its energy lies above `π/M`** and is folded back on subsampling. That residual folding is the phenomenon the study measures against.

## Kernel

`_build_kernel` returns `np.ones(M) / M` — length equals the decimation factor `M`. The length therefore **scales with `M` automatically**, the very property that short fixed binomial kernels lack.

## Config

| Key | Meaning |
|---|---|
| `decimation_factor` (int or `(1, M)`) | **required** |
| shared keys | `padding_mode`, `phase`, `gain`, `warn_if_worse_than_boxcar` — see [base class](support_pooling.md). |

No kernel-specific keys.

## Differences from the base class

- Overrides `_check_against_boxcar()` to a **no-op**: this layer *is* the boxcar, so there is nothing to warn about.

## Equivalences

- With `padding_mode='constant'` and `phase='center'`, numerically identical to `nn.AvgPool1d(M, stride=M)` on the interior (boundary samples differ because `nn.AvgPool1d` does not pad). `self_test()` check #2 verifies this.
- Identical to `blur_pool_1d(kernel='boxcar', kernel_size=M)` — the nesting that makes the family provably consistent.

## Shapes

Input `(B, D, T)` → output `(B, D, T // M)`. Zero trainable parameters.

## Multichannel variant

`avg_pool_1d_multichannel(config)` wraps `avg_pool_1d` for 4D EEG tensors `(B, D, C, T)`, applying it channel-wise (see [`_multichannel_pool_1d`](support_pooling.md#multichannel-wrapper)). Same `config`; output `(B, D, C, T // M)`.
