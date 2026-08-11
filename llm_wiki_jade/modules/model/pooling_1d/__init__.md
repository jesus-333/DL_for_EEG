# `pooling_1d/__init__.py` — factory, registry, self-test

Source: `src/jade/model/pooling_1d/__init__.py`

Sub-package init. Carries the long **motivation docstring** (the anti-aliasing theory summarized in the [quickstart](0_module_quickstart.md)), re-exports the public API, and hosts the `get_pooling_layer` factory plus `self_test`.

Re-exports from the per-operator modules: the 1D layers `avg_pool_1d`, `blur_pool_1d`, `sinc_pool_1d`, `dwt_pool_1d`, `max_pool_1d`; their channel-wise `{name}_multichannel` wrappers; `WAVELET_FILTERS`; and the internal helpers `_as_decimation_factor`, `_filter_and_subsample`, `_get` (used by the factory and self-test).

## Public API (`__all__`)

The five 1D layers (`avg_pool_1d`, `max_pool_1d`, `blur_pool_1d`, `sinc_pool_1d`, `dwt_pool_1d`), the five multichannel wrappers (`avg_pool_1d_multichannel`, `max_pool_1d_multichannel`, `blur_pool_1d_multichannel`, `sinc_pool_1d_multichannel`, `dwt_pool_1d_multichannel`), `get_pooling_layer`, `POOLING_REGISTRY`, `MULTICHANNEL_POOLING_REGISTRY`, `WAVELET_FILTERS`, `self_test`.

## `POOLING_REGISTRY`

Lowercase alias → pooling class:

| Aliases | Class |
|---|---|
| `avg`, `average`, `boxcar` | `avg_pool_1d` |
| `max` | `max_pool_1d` |
| `blur`, `blurpool`, `binomial`, `gaussian` | `blur_pool_1d` |
| `sinc`, `kaiser` | `sinc_pool_1d` |
| `dwt`, `wavelet` | `dwt_pool_1d` |

## `MULTICHANNEL_POOLING_REGISTRY`

The same aliases mapped to the channel-wise wrappers (`avg` → `avg_pool_1d_multichannel`, `max` → `max_pool_1d_multichannel`, …). These operate on 4D `(B, D, C, T)` EEG tensors and are what [`jade.model.support_functions.get_multichannel_pooling`](../support_functions.md) dispatches through for a model such as EEGNet. See the [quickstart](0_module_quickstart.md#multichannel-4d-variants) and [`support_pooling`](support_pooling.md) (`_multichannel_pool_1d` base).

## `get_pooling_layer(config: pooling_config | dict) -> nn.Module`

Single entry point, designed for declarative ablation sweeps (one config per experimental arm, easy to serialize to YAML/JSON and log with results).

| Parameter | Type | Meaning |
|---|---|---|
| `config` | [`pooling_config`](../../config/model/pooling_1d_cfg.md) or `dict` | A pooling-config dataclass **or** a plain dict (a dataclass is normalized via `to_dict()`, mirroring `get_pooling_config`). Must carry `pooling_type` (an alias above or `'identity'`/`'none'`), plus whatever keys the chosen operator needs (see each operator page). |

**Returns:** the built pooling layer, or `nn.Identity`.

**Behaviour / internals:**
- `pooling_type` is lowercased before lookup.
- **Identity short-circuits:** `pooling_type in ('identity', 'none')` → `nn.Identity()`. Also, a `decimation_factor` (via `_get`, accepting aliases `pool_size`/`factor`/`p_kernel`) that resolves to `1` → `nn.Identity()`. Convenient for ablating a pooling stage away entirely.
- Never mutates the caller's dict — works on `cfg = dict(config)`.
- **Kernel-family convenience:** `pooling_type='gaussian'` sets `cfg.setdefault('kernel','gaussian')` and `'binomial'` sets `cfg.setdefault('kernel','binomial')` — both select `blur_pool_1d` *and* its kernel family in one word.

**Raises:**

| Exception | Condition |
|---|---|
| `KeyError` | no `pooling_type` key. |
| `ValueError` | unknown `pooling_type`. |

**Examples (from docstring):**
```python
get_pooling_layer({'pooling_type' : 'avg',  'decimation_factor' : 4})
get_pooling_layer({'pooling_type' : 'blur', 'decimation_factor' : 4, 'kernel_size' : 17})
get_pooling_layer({'pooling_type' : 'sinc', 'decimation_factor' : 8, 'stopband_attenuation_db' : 60, 'transition_width' : 0.25})
get_pooling_layer({'pooling_type' : 'dwt',  'decimation_factor' : 8, 'wavelet' : 'sym4'})
```

## `self_test(verbose: bool = True) -> None`

Verifies the invariants the sub-package relies on; run once after installing (`assert`s each check; also called by [`__main__`](__main__.md)). Seeds `torch.manual_seed(0)`. The six checks:

| # | Check | What it proves |
|---|---|---|
| 1 | Every layer outputs exactly `T // M` for many `(T, M)` combos (`T` in 1000/1001/1002/999/512/250, `M` in 2/4/8). | shape invariance across the whole family. |
| 2 | `avg_pool_1d` reproduces `nn.AvgPool1d(M, stride=M)` on the interior (`padding_mode='replicate'`, `max|diff| < 1e-5`). | average pooling really is the boxcar special case. |
| 3 | All DC-normalized layers pass a constant signal (value 3.7) unchanged (`err < 1e-4`). | unit DC gain. |
| 4 | The DWT cascade matches its own `effective_kernel()` under circular padding (`max|diff| < 1e-9`), using the documented sampling offset. | the noble identity / effective-kernel collapse is correct. |
| 5 | `blur`/`sinc`/`dwt` are gradient-transparent (finite grad flows), have **0 trainable params**, and an empty `state_dict()`. | no learnable state; kernel is a non-persistent buffer. |
| 6 | Each `{name}_multichannel` wrapper on a 4D `(B, D, C, T)` tensor outputs `(B, D, C, T // M)` and is **bit-identical** to an explicit per-channel loop over the 1D layer (`max|diff| = 0`). | the vectorized `(D, C)`-flatten really is channel-wise pooling. |
