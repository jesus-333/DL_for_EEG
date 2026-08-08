# `config/model/EEGNet_cfg.py`

Source: `src/jade/config/model/EEGNet_cfg.py`

Defines `EEGNet_config`, the validated dataclass consumed by [`EEGNet` and `EEGNet_Classifier`](../../model/EEGNet.md). One config covers both models because the classifier's key set is a superset of the plain extractor's; the `name` field selects which one `jade.model.get_model` builds. Every field has a BCI IV-2a style default, so `EEGNet_config()` is valid on its own.

## Module constants

- `ALLOWED_NAMES = ("eegnet", "eegnet_classifier", "eegnetclassifier")` — the accepted values of `name`.
- `ACTIVATIONS = ("relu", "elu", "selu", "gelu")` — accepted activation names (validated here so the config stays torch-free; the runtime resolution is done by `jade.model.support_functions.get_activation`).

## Class `EEGNet_config(model_config)`

`@dataclass`. Fields:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | str | `"eegnet_classifier"` | Which model to build (`eegnet` or `eegnet_classifier`). Normalized + membership-checked (NOT frozen, since it is a real choice). |
| `use_bias` | bool | `False` | Conv layers use a bias term. |
| `D` | int | `2` | Spatial filters per temporal filter (total = `filter_1 * D`). |
| `activation` | str | `"elu"` | One of `ACTIVATIONS`. |
| `prob_dropout` | float | `0.5` | Dropout probability in `[0, 1]`. |
| `use_dropout_2d` | bool | `False` | `nn.Dropout2d` vs `nn.Dropout`. |
| `C` | int | `22` | EEG channels. |
| `T` | int | `1000` | Time samples. |
| `depth_first_layer` | int | `1` | Input depth (>1 for time-frequency input). |
| `filter_1` | int | `8` | Temporal filters. |
| `c_kernel_1` | tuple | `(1, 64)` | Temporal conv kernel. |
| `c_kernel_2` | tuple | `(22, 1)` | Spatial conv kernel (usually `(C, 1)`). |
| `filter_2` | int | `16` | Separable-conv output filters. |
| `c_kernel_3` | tuple | `(1, 16)` | Separable-block depthwise kernel. |
| `flatten_output` | bool | `False` | Flatten the plain-EEGNet output (ignored by the classifier). |
| `input_size` | tuple | `(1, 1, 22, 1000)` | `(batch, depth, C, T)` dummy input to size the classifier head. |
| `n_classes` | int | `4` | Classifier output classes. |
| `use_softmax` | bool | `True` | `True` → `LogSoftmax` head (log-probs, pair with `nll`); `False` → `Identity` head (logits, pair with `cross_entropy`). |
| `print_var` | bool | `False` | Print param counts / shape tracking at build time. |
| `pooling` | `eegnet_pooling_config` | `eegnet_pooling_config()` | The two pooling stages (the `[pooling]` section). **Replaces** the old `p_kernel_1` / `p_kernel_2`. A dict is accepted and promoted. See below. |

> **Migration note:** the old `p_kernel_1` / `p_kernel_2` fields (2D `AvgPool2d` kernels like `(1, 4)`) are gone. Their behaviour is now the `pooling` section with `pooling_type_* = "default"` and `pooling_factor_1 = 4`, `pooling_factor_2 = 8` (the default), which builds the same `nn.AvgPool2d((1, 4))` / `((1, 8))`.

## The `pooling` section — `eegnet_pooling_config(generic_config)`

`@dataclass`. Configures the two EEGNet pooling stages **independently** (stage 1 lives in the spatial-filter block, stage 2 in the separable-convolution block). Serialized as the `[pooling]` TOML sub-table (declared last in `EEGNet_config` so it follows all scalar keys).

| Field | Type | Default | Meaning |
|---|---|---|---|
| `pooling_type_1` / `pooling_type_2` | str | `"default"` | Operator per stage. `"default"` → classic `nn.AvgPool2d`; a 1D pooling name (`avg`, `max`, `blur`, `sinc`, `dwt`, and aliases) → the channel-wise `*_multichannel` wrapper. Case-insensitive; also accepts `identity`/`none`. |
| `pooling_factor_1` / `pooling_factor_2` | int | `4` / `8` | Decimation factor `M`: new #samples = `T / M`. Must be a non-zero int. **A negative value disables that stage** (a `warnings.warn` fires at instantiation). |
| `params_1` / `params_2` | dict | `{}` | Extra, type-specific params per stage (e.g. `{"wavelet": "sym4"}`, `{"cutoff_scale": 0.8}`). Empty → the layer's own defaults. |

Two derived properties assemble the per-stage dicts the model consumes: `pooling_layer_1_config` / `pooling_layer_2_config` = `{pooling_type, decimation_factor, **params}`. `EEGNet_config` re-exposes them (delegating to `self.pooling`) so `EEGNet.__init__` can read `config.pooling_layer_1_config` and feed it to [`get_multichannel_pooling`](../../model/support_functions.md).

## Validation (`__post_init__`)

- `name` lowercased and checked against `ALLOWED_NAMES`; `activation` lowercased and checked against `ACTIVATIONS`.
- Boolean flags must be genuine `bool`; the sizes (`D`, `filter_1`, `filter_2`, `C`, `T`, `depth_first_layer`) must be positive ints (a `bool` is rejected, since it subclasses `int`); `n_classes` must be an int `>= 2`.
- `prob_dropout` must be a float in `[0, 1]`.
- The conv kernels (`c_kernel_1/2/3`) are coerced list→tuple and must be 2-tuples of positive ints. `input_size` is coerced to a 4-tuple of positive ints.
- `pooling` is promoted from a dict to `eegnet_pooling_config` if needed (base `from_dict` already does this when loading a file). `eegnet_pooling_config.__post_init__` then validates each stage: `pooling_type_*` against the pooling registry + `default`/`identity`/`none`; `pooling_factor_*` as a non-zero int (negative ⇒ warning, stage disabled); `params_*` as a dict; and, for an active non-`default` 1D type, it round-trips `{pooling_type, decimation_factor, **params}` through [`get_pooling_config`](pooling_1d_cfg.md) to validate the type-specific params (e.g. dwt power-of-two, sinc knobs).

Module-level helpers `_is_int`, `_check_positive_int`, `_check_bool`, `_coerce_kernel` back these checks.

## Consumption / gotchas

`EEGNet.__init__` / `EEGNet_Classifier.__init__` call `get_model_config(config)` and then read the attributes directly — so all defaulting and validation happen here, not in the model. A dict is still accepted anywhere a config is (promoted through `get_model_config`), but the canonical key is `name` (the old `model_name` dict key no longer exists).
