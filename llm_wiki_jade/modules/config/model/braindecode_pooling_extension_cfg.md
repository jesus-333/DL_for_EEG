# `config/model/braindecode_pooling_extension_cfg.py`

Source: `src/jade/config/model/braindecode_pooling_extension_cfg.py`

Defines `braindecode_wrapper_config`, the validated dataclass consumed by [`braindecode_pooling_wrapper`](../../model/braindecode_pooling_extension.md). It says which braindecode model to build, the kwargs to hand it, and which [`pooling_1d`](pooling_1d_cfg.md) operator to swap in for its pooling layers.

Unlike [`EEGNet_config`](EEGNet_cfg.md), the braindecode model hyper-parameters do **not** get their own dataclass: braindecode already exposes a large, per-model argument set and validates it itself, so they travel as a raw `model_kwargs` dict handed straight to the braindecode constructor. What *is* described here (through the shared [`pooling_config`](pooling_1d_cfg.md) family) is the pooling operator to swap in — the point of the wrapper. The config stays **torch-free and braindecode-free** like the rest of `jade.config`; braindecode is imported only by the runtime module.

## Module constants

- `ALLOWED_NAMES = ("braindecode_wrapper", "braindecode", "braindecode_pooling")` — accepted values of `name` (mirror the `jade.model.MODEL_REGISTRY` / `MODEL_CONFIG_REGISTRY` aliases).
- `POOLING_FACTOR_SOURCES = ("stride", "kernel")` — where the decimation factor `M` of each swapped layer is read from on the original braindecode pooling layer.

## Class `braindecode_wrapper_config(model_config)`

`@dataclass`. The defaults form a valid skeleton (a `ShallowFBCSPNet` whose average pooling is re-expressed with the `avg` operator); a real run overrides `braindecode_model_name`, `model_kwargs` and `pooling`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | str | `"braindecode_wrapper"` | Discriminator selecting the wrapper in `get_model`. One of `ALLOWED_NAMES` (case-insensitive). |
| `braindecode_model_name` | str | `"ShallowFBCSPNet"` | braindecode model class name, resolved against `braindecode.models` (e.g. `ShallowFBCSPNet`, `EEGNet`, `Deep4Net`). Must be a non-empty string. |
| `model_kwargs` | dict | `{}` | Keyword args passed straight to the braindecode constructor (e.g. `n_chans`, `n_outputs`, `n_times`). **Validated by braindecode**, not here — so an empty dict only errors at build time. |
| `pooling` | dict | `{"pooling_type": "avg"}` | The pooling operator to substitute, as a [`pooling_config`](pooling_1d_cfg.md) dict (`pooling_type` + operator params, e.g. `{"pooling_type": "sinc", "cutoff_scale": 0.8}`). Its `decimation_factor` is **ignored** (each swapped layer keeps the factor `M` of the original braindecode layer). |
| `keep_pool_nonlin` | bool | `True` | Keep the post-pool nonlinearity (e.g. ShallowFBCSPNet's `safe_log` `pool_nonlin_exp`). `True` changes only the pooling type; `False` replaces that module with `nn.Identity` (an extra ablation arm). |
| `pooling_factor_source` | str | `"stride"` | Which attribute of the original pooling layer defines `M`. `"stride"` (the true downsampling factor, correct for overlapping pooling too) or `"kernel"`. |
| `example_input_shape` | tuple or None | `None` | Optional explicit probe shape (batch included) for the trailing-classifier re-sizing dummy forward; when `None` it is inferred from the model's `n_chans` / `n_times`. |

### Property `target_pooling_config`

Promotes the `pooling` dict to a proper [`pooling_config`](pooling_1d_cfg.md) dataclass via `get_pooling_config`. Its `decimation_factor` is a placeholder — the wrapper overrides it per layer with the factor of the braindecode pooling layer being replaced.

### Validation (`__post_init__`)

- `name` lowercased and checked against `ALLOWED_NAMES`.
- `braindecode_model_name` must be a non-empty string; `model_kwargs` must be a dict (its contents are braindecode's responsibility).
- `pooling` must be a dict and is round-tripped through [`get_pooling_config`](pooling_1d_cfg.md) to validate the `pooling_type` and any operator-specific params early (e.g. dwt power-of-two, sinc knobs).
- `keep_pool_nonlin` must be a genuine `bool`; `pooling_factor_source` lowercased and checked against `POOLING_FACTOR_SOURCES`.
- `example_input_shape`, if given, is coerced list→tuple and must hold positive ints.

## Consumption / gotchas

- Built through the standard path: `get_model(config)` → `braindecode_pooling_wrapper(config)`; a dict with `name = 'braindecode'` (or an alias) is promoted via `get_model_config`.
- **The model kwargs are not validated here** (by design): a missing required kwarg surfaces as braindecode's own error when the wrapper builds the model.
- Registered in [`MODEL_CONFIG_REGISTRY`](__init__.md) under all three aliases and re-exported from `jade.config.model` (and `jade.config`).
