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
| `p_kernel_1` | tuple or None | `(1, 4)` | First pooling kernel; `None` disables it. |
| `filter_2` | int | `16` | Separable-conv output filters. |
| `c_kernel_3` | tuple | `(1, 16)` | Separable-block depthwise kernel. |
| `p_kernel_2` | tuple or None | `(1, 8)` | Second pooling kernel; `None` disables it. |
| `flatten_output` | bool | `False` | Flatten the plain-EEGNet output (ignored by the classifier). |
| `input_size` | tuple | `(1, 1, 22, 1000)` | `(batch, depth, C, T)` dummy input to size the classifier head. |
| `n_classes` | int | `4` | Classifier output classes. |
| `use_softmax` | bool | `True` | `True` → `LogSoftmax` head (log-probs, pair with `nll`); `False` → `Identity` head (logits, pair with `cross_entropy`). |
| `print_var` | bool | `False` | Print param counts / shape tracking at build time. |

## Validation (`__post_init__`)

- `name` lowercased and checked against `ALLOWED_NAMES`; `activation` lowercased and checked against `ACTIVATIONS`.
- Boolean flags must be genuine `bool`; the sizes (`D`, `filter_1`, `filter_2`, `C`, `T`, `depth_first_layer`) must be positive ints (a `bool` is rejected, since it subclasses `int`); `n_classes` must be an int `>= 2`.
- `prob_dropout` must be a float in `[0, 1]`.
- The conv kernels are coerced list→tuple and must be 2-tuples of positive ints; `p_kernel_1`/`p_kernel_2` may be `None`. `input_size` is coerced to a 4-tuple of positive ints.

Module-level helpers `_is_int`, `_check_positive_int`, `_check_bool`, `_coerce_kernel` back these checks.

## Consumption / gotchas

`EEGNet.__init__` / `EEGNet_Classifier.__init__` call `get_model_config(config)` and then read the attributes directly — so all defaulting and validation happen here, not in the model. A dict is still accepted anywhere a config is (promoted through `get_model_config`), but the canonical key is `name` (the old `model_name` dict key no longer exists).
