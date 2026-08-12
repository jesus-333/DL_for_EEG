# `pooling_1d/max_pool.py` — `max_pool_1d` (nonlinear reference)

Source: `src/jade/model/pooling_1d/max_pool.py`

Plain max pooling, included **only** as the *un-anti-aliased nonlinear reference* arm of the study. Unlike the other layers it is **not** a subclass of `_fir_decimation_pool_1d` — it subclasses `nn.Module` directly, because max pooling does not fit the "filter then subsample" decomposition. Imports `torch`, `torch.nn`, and the helpers `_as_decimation_factor`, `_get`.

## Why it is different

The max operator is **nonlinear**: it has no frequency response, and worse, it can *create* new high-frequency content absent from its input (the envelope of a rectified oscillation contains harmonics). So it both fails to suppress out-of-band energy *and* manufactures more of it before subsampling. Several works report average pooling anti-aliases *better* than max pooling for exactly this reason. It is here so the ablation can report the operator most widely used in practice, and so the ordering **max < avg < properly filtered** can be demonstrated rather than assumed.

## Config

| Key | Type | Meaning | Default |
|---|---|---|---|
| `decimation_factor` | int or `(1, M)` | factor `M`. Aliases: `pool_size`, `factor`, `p_kernel`. Missing → `KeyError`. | **required** |
| `ceil_mode` | bool | passed to `nn.MaxPool1d`. Left `False` so output length is `T // M`, matching every other layer. | `False` |

## Construction & forward

- `__init__` resolves the factor via `_get`/`_as_decimation_factor`, then builds `self.pool = nn.MaxPool1d(kernel_size=M, stride=M, ceil_mode=...)`.
- `forward(x)` validates `x` is 3D (`ValueError` otherwise) and returns `self.pool(x)`. Input `(B, D, T)` → output `(B, D, T // M)`.

## `summary(self) -> dict`

Same keys as the base class's `summary` so it slots into the same result table, but the spectral metrics are **undefined (NaN)** because there is no linear frequency response:

| Key | Value |
|---|---|
| `layer` | `"max_pool_1d"` |
| `kind` | `"nonlinear"` |
| `decimation_factor` | `M` |
| `kernel_length` | `M` |
| `stopband_leakage_pct` | `nan` |
| `stopband_attenuation_db` | `nan` |
| `passband_ripple_db` | `nan` |
| `group_delay_samples` | `0.0` |
| `dc_gain` | `1.0` |

## `extra_repr(self) -> str`

`"M={} (nonlinear, no anti-aliasing)"`.

## Notes

- Zero trainable parameters.
- Does **not** implement `effective_kernel`, `stopband_leakage`, etc. — it is not an FIR filter, so those would be meaningless.
- Has a `plot_frequency_response(...)` method (so every pooling layer exposes the same interface), but it **always raises `NotImplementedError`**: a nonlinear operator has no frequency response to plot. The multichannel wrapper delegates to it and therefore raises too.

## Multichannel variant

`max_pool_1d_multichannel(config)` wraps `max_pool_1d` for 4D EEG tensors `(B, D, C, T)`, applying it channel-wise (see [`_multichannel_pool_1d`](support_pooling.md#multichannel-wrapper)). Same `config`; output `(B, D, C, T // M)`.
