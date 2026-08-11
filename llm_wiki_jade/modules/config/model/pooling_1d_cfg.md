# `config/model/pooling_1d_cfg.py`

Source: `src/jade/config/model/pooling_1d_cfg.py`

The whole configuration family for the [`jade.model.pooling_1d`](../../model/pooling_1d/0_module_quickstart.md) layers, deliberately kept in a **single file** (pooling is reused several times inside a model and the operators share most parameters). Discriminated on `pooling_type` — the exact key the runtime `get_pooling_layer` reads — and the canonical field names match the layer keys, so `config.to_dict()` feeds `get_pooling_layer` directly. As of the dataclass-consistency pass, both `get_pooling_layer` and [`get_multichannel_pooling`](../../model/support_functions.md) also accept a `pooling_config` dataclass directly (they normalize it via `to_dict()` internally), so you no longer have to pre-convert to a dict.

## Module constants

- `BLUR_KERNELS = ("binomial", "gaussian", "boxcar")`.
- `PADDING_MODES = ("reflect", "replicate", "constant", "circular")`, `GAINS = ("dc", "orthonormal", "none")`.

## Class hierarchy

```
pooling_config (base)            pooling_type, decimation_factor  + polymorphic from_dict
└── _fir_pooling_config          + padding_mode, phase, gain, warn_if_worse_than_boxcar
    ├── avg_pooling_config       pooling_type "avg"   (no extra fields)
    ├── blur_pooling_config      pooling_type "blur"  + kernel, kernel_size, sigma
    ├── sinc_pooling_config      pooling_type "sinc"  + cutoff_scale, window, num_taps, beta,
    │                                                    stopband_attenuation_db, transition_width, max_taps
    └── dwt_pooling_config       pooling_type "dwt"   + wavelet   (decimation_factor must be a power of two)
max_pooling_config (base only)   pooling_type "max"   + ceil_mode   (max has no FIR filter)
```

Each concrete config **freezes** its `pooling_type` in `__post_init__` and validates its own parameters (categorical checks for `padding_mode`/`gain`/`kernel`/`phase`; positivity for the optional numeric overrides, which are validated only when not `None` so the layer's `M`-scaled defaults apply). The base validates `decimation_factor` (positive int); `dwt` additionally requires it to be a power of two.

## `pooling_config.from_dict` (polymorphic)

Called on the base class, it reads `pooling_type`, looks it up in `POOLING_CONFIG_REGISTRY` (unknown → `ValueError`), and delegates to the concrete subclass; called on a subclass it uses the standard `generic_config.from_dict`. The registry lives in the same module, so no lazy import is needed.

## Registry + factory

- **`POOLING_CONFIG_REGISTRY`** maps every alias accepted by `get_pooling_layer` to its config class: `avg`/`average`/`boxcar` → `avg_pooling_config`, `max` → `max_pooling_config`, `blur`/`blurpool`/`binomial`/`gaussian` → `blur_pooling_config`, `sinc`/`kaiser` → `sinc_pooling_config`, `dwt`/`wavelet` → `dwt_pooling_config`.
- **`get_pooling_config(config)`** — idempotent (instance → untouched, dict → `pooling_config.from_dict`, else `TypeError`).

## Notes / gotchas

- Optional overrides default to `None` (`kernel_size`, `sigma`, `num_taps`, `window`, `stopband_attenuation_db`, `transition_width`, `max_taps`); on write `to_toml` drops them and on read the pooling layer applies its own `M`-scaled defaults (e.g. blur `4*M+1`, sinc `8*M+1`).
- The pooling layers are **wired into EEGNet**: its `[pooling]` section (`eegnet_pooling_config`) validates each stage's type-specific params by round-tripping them through `get_pooling_config` here, and `EEGNet` builds the layers via `support_functions.get_multichannel_pooling`. This config family is also usable standalone.
