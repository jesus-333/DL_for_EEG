# `config/model/TIDNet_cfg.py`

Source: `src/jade/config/model/TIDNet_cfg.py`

Defines `TIDNet_config`, the validated dataclass consumed by [`TIDNet` and `TIDNet_Classifier`](../../model/TIDNet.md). One config covers both models because the classifier's key set is a superset of the plain extractor's; the `name` field selects which one `jade.model.get_model` builds. Every field has a BCI IV-2a style default that reproduces the original TIDNet architecture (Kostas & Rudzicz, 2020), so `TIDNet_config()` is valid on its own. Torch-free like the rest of `jade.config`.

## Module constants

- `ALLOWED_NAMES = ("tidnet", "tidnet_classifier", "tidnetclassifier")` — the accepted values of `name`.
- `ACTIVATIONS = ("relu", "leaky_relu", "elu", "selu", "gelu")` — accepted activation names (validated here so the config stays torch-free; runtime resolution is done by `jade.model.support_functions.get_activation`). `leaky_relu` is the original choice, and was added to `get_activation` for this model.
- `POOL_MODES = ("max", "mean")` — which pooling the `'default'` type builds.

## Class `TIDNet_config(model_config)`

`@dataclass`. Fields (grouped as in the source):

### Discriminator

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | str | `"tidnet_classifier"` | Which model to build (`tidnet` = feature extractor, or `tidnet_classifier` = extractor + head). Normalized + membership-checked against `ALLOWED_NAMES`. |

### Input geometry

| Field | Type | Default | Meaning |
|---|---|---|---|
| `C` | int | `22` | EEG channels. Sets the height of the channel-collapse kernel `(C, 1)` of the spatial filter. |
| `T` | int | `1000` | Time samples. It also sets the temporal kernel length, `ceil(temp_span * T)`. |
| `depth_first_layer` | int | `1` | Input depth (>1 for time-frequency input). |

### Temporal filter

| Field | Type | Default | Meaning |
|---|---|---|---|
| `t_filters` | int | `32` | Filters of the residual dilated temporal filter. |
| `temp_layers` | int | `2` | Stacked dilated convolutions. The dilation of the `i`-th is `temp_layers - i`, so the first is the most dilated. |
| `temp_span` | float | `0.05` | Temporal kernel length as a fraction of `T` (the kernel covers `ceil(temp_span * T)` samples, rounded up to the next odd value so the convolution preserves the time axis). In `(0, 1]`. |

### Spatial filter

| Field | Type | Default | Meaning |
|---|---|---|---|
| `s_growth` | int | `24` | Growth rate: every dense layer appends `s_growth` feature maps to its input. |
| `spat_layers` | int | `2` | Dense layers. The extractor outputs `t_filters + s_growth * spat_layers` feature maps (80 by default). |
| `bottleneck` | int | `3` | Bottleneck factor of a dense layer (its `1x1` conv expands to `bottleneck * s_growth` maps before the spatial conv reduces them back). |
| `summary` | int | `-1` | Length the time axis is resized to by the adaptive average pooling closing the extractor. `-1` = "keep the pooled length", i.e. `T / pooling_factor`; any other value must be a positive int. |

### Shared options

| Field | Type | Default | Meaning |
|---|---|---|---|
| `activation` | str | `"leaky_relu"` | One of `ACTIVATIONS`. Used in every block; `leaky_relu` is the original. |
| `prob_dropout` | float | `0.4` | Dropout of the temporal and spatial blocks, in `[0, 1]`. |
| `pooling` | `tidnet_pooling_config` | `tidnet_pooling_config()` | The single decimating pooling stage (the `[pooling]` section). A dict is accepted and promoted. See below. |

### Classification head

| Field | Type | Default | Meaning |
|---|---|---|---|
| `n_classes` | int | `4` | Classifier output classes. |
| `use_softmax` | bool | `False` | `False` (default) → `Identity` head (raw logits, pair with `cross_entropy` — the numerically stabler default and the recommended default); `True` → `LogSoftmax` head (log-probs, pair with `nll`). The original TIDNet always ends with a log-softmax, so `True` is the **faithful** setting. |

### Misc

| Field | Type | Default | Meaning |
|---|---|---|---|
| `flatten_output` | bool | `False` | Flatten the plain-`TIDNet` output to `(B, n_features)` (ignored by the classifier). |
| `input_size` | tuple | `(1, 1, 22, 1000)` | `(batch, depth, C, T)` dummy input to size the feedforward layer of the head. |
| `print_var` | bool | `False` | Print param counts / shape tracking at build time. |

> **Faithfulness:** with all defaults a 22×1000 trial flows to `(B, 80, 66)` features, and `TIDNet_Classifier` has **240 404** trainable parameters — exactly the count of `braindecode.models.TIDNet` for the same geometry. The only intentional difference is that the auxiliary subject / run heads of the original are not implemented.

## The `pooling` section — `tidnet_pooling_config(generic_config)`

`@dataclass`. Configures the **single decimating** pooling stage, which closes the temporal block (before the EEG-channel axis is collapsed). Serialized as the `[pooling]` TOML sub-table (declared last in `TIDNet_config` so it follows all scalar keys).

| Field | Type | Default | Meaning |
|---|---|---|---|
| `pooling_type` | str | `"default"` | `"default"` → the original non-overlapping `nn.MaxPool2d((1, pooling_factor))` (or `AvgPool2d`, see `pool_mode`); a 1D pooling name (`avg`, `max`, `blur`, `sinc`, `dwt`, and aliases) → the channel-wise `*_multichannel` wrapper; also accepts `identity`/`none`. Case-insensitive. |
| `pooling_factor` | int | `15` | Decimation factor `M` (new #samples = `T / M`). **Positive** int — unlike EEGNet and Deep4Net a negative value is not the way to disable the stage, because this factor also sets the default `summary` length; use `pooling_type = "identity"` instead. `dwt` additionally requires a power of two, so the original 15 is rejected with it — use 16. |
| `pool_mode` | str | `"max"` | Which pooling the `"default"` type builds: `"max"` (`nn.MaxPool2d`, the original) or `"mean"` (`nn.AvgPool2d`). Ignored by the 1D operators. |
| `params` | dict | `{}` | Extra type-specific params for a 1D operator (e.g. `{"wavelet": "sym4"}`, `{"cutoff_scale": 0.8}`). Empty → the layer's own defaults. Ignored by `"default"`/`"identity"`. |

The derived property `pooling_layer_config` assembles the config the model consumes: a [`pooling_config`](pooling_1d_cfg.md) **dataclass** for an active 1D operator (decimation factor `pooling_factor` + `params`), the plain `{pooling_type, decimation_factor, pool_mode, **params}` **dict** for `default` / `identity` / `none`. `TIDNet_config` re-exposes it (delegating to `self.pooling`) so the model can feed it to [`get_tidnet_pooling`](../../model/TIDNet.md#get_tidnet_poolingconfig---nnmodule).

> **One stage, and one deliberate omission.** The `nn.AdaptiveAvgPool1d(summary)` that closes the feature extractor is **not** configured here: it is an adaptive resize, not a decimation by an integer factor, so it has no anti-aliasing counterpart in `pooling_1d`. The `[pooling]` section covers the one stage where aliasing can actually be introduced.

## Validation (`__post_init__`)

- `name` lowercased and checked against `ALLOWED_NAMES`; `activation` lowercased and checked against `ACTIVATIONS`.
- Boolean flags (`use_softmax`, `flatten_output`, `print_var`) must be genuine `bool`.
- The sizes (`C`, `T`, `depth_first_layer`, `t_filters`, `temp_layers`, `s_growth`, `spat_layers`, `bottleneck`) must be positive ints (a `bool` is rejected); `n_classes` must be an int `>= 2`; `summary` must be positive **unless** it is the `-1` sentinel.
- `prob_dropout` must be a float in `[0, 1]`; `temp_span` must be a float in `(0, 1]` (it is a fraction of the input window, and 0 would give an empty kernel).
- `pooling` is promoted from a dict to `tidnet_pooling_config` if needed. Its `__post_init__` validates `pooling_type` against the pooling registry + `default`/`identity`/`none`, `pool_mode` against `POOL_MODES`, `pooling_factor` as a positive int, `params` as a dict, and — for an active 1D type — round-trips `{pooling_type, decimation_factor, **params}` through [`get_pooling_config`](pooling_1d_cfg.md).
- `input_size` is coerced to a 4-tuple of positive ints, and its time samples must be at least `pooling_factor` (the temporal convolutions are padded, so the pooling is the only step that shrinks the time axis).

Module-level helpers `_is_int`, `_check_positive_int`, `_check_bool`, `_check_probability` back these checks.

## Consumption / gotchas

`TIDNet.__init__` / `TIDNet_Classifier.__init__` call `get_model_config(config)` and then read the attributes directly — all defaulting/validation happens here, not in the model. `jade.training.support_training.cross_check_config` reads `use_softmax` and `n_classes` off this config (via `getattr`), so it plugs into the training pipeline exactly like EEGNet — **but note** that the faithful `use_softmax = True` must be paired with the `nll` loss, or that cross-check will (correctly) complain. A commented TOML skeleton ships as `src/jade/config/default_config/model/TIDNet.toml` (see [`cli/default_config`](../../cli/default_config.md)).
