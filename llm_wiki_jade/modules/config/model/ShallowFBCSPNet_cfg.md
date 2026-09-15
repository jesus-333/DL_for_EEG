# `config/model/ShallowFBCSPNet_cfg.py`

Source: `src/jade/config/model/ShallowFBCSPNet_cfg.py`

Defines `ShallowFBCSPNet_config`, the validated dataclass consumed by [`ShallowFBCSPNet` and `ShallowFBCSPNet_Classifier`](../../model/ShallowFBCSPNet.md). One config covers both models because the classifier's key set is a superset of the plain extractor's; the `name` field selects which one `jade.model.get_model` builds. Every field has a BCI IV-2a style default that reproduces the original Shallow ConvNet architecture (Schirrmeister et al., 2017), so `ShallowFBCSPNet_config()` is valid on its own. Torch-free like the rest of `jade.config`.

## Module constants

- `ALLOWED_NAMES = ("shallowfbcspnet", "shallowfbcspnet_classifier", "shallowfbcspnetclassifier")` — the accepted values of `name`.
- `NONLINEARITIES = ("square", "safe_log", "identity", "relu", "leaky_relu", "elu", "selu", "gelu")` — accepted values of `conv_nonlin` / `pool_nonlin` (validated here so the config stays torch-free; runtime resolution is done by `jade.model.support_functions.get_nonlinearity`).
- `POOL_MODES = ("mean", "max")` — which pooling the `'default'` type builds.

## Class `ShallowFBCSPNet_config(model_config)`

`@dataclass`. Fields (grouped as in the source):

### Discriminator

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | str | `"shallowfbcspnet_classifier"` | Which model to build (`shallowfbcspnet` = feature extractor, or `shallowfbcspnet_classifier` = extractor + head). Normalized + membership-checked against `ALLOWED_NAMES`. |

### Convolutional front end

| Field | Type | Default | Meaning |
|---|---|---|---|
| `C` | int | `22` | EEG channels. Sets the height of the spatial conv kernel `(C, 1)`, which collapses the channel axis to 1. |
| `T` | int | `1000` | Time samples. |
| `depth_first_layer` | int | `1` | Input depth (>1 for time-frequency input). |
| `n_filters_time` | int | `40` | Filters of the temporal convolution. |
| `filter_time_length` | int | `25` | Length of the temporal conv kernel, i.e. the kernel is `(1, filter_time_length)`. Must not exceed `T` (the conv is unpadded). |
| `n_filters_spat` | int | `40` | Filters of the spatial convolution. Ignored when `split_first_layer = False`. |
| `split_first_layer` | bool | `True` | `True` (the original) → temporal `(1, filter_time_length)` then spatial `(C, 1)` conv, with no nonlinearity in between; `False` → a single `(C, filter_time_length)` conv does both at once. |
| `batch_norm` | bool | `True` | Add a `BatchNorm2d` after the front end (the preceding conv then drops its bias, which the normalization would make redundant). |
| `batch_norm_alpha` | float | `0.1` | Momentum of the batch normalization, in `[0, 1]`. Ignored when `batch_norm = False`. |
| `conv_nonlin` | str | `"square"` | Nonlinearity **before** the pooling. One of `NONLINEARITIES`. |
| `pool_nonlin` | str | `"safe_log"` | Nonlinearity **after** the pooling. One of `NONLINEARITIES`. Together, `square` + `safe_log` give the log-variance feature of FBCSP. |
| `prob_dropout` | float | `0.5` | Dropout after the pooling nonlinearity, in `[0, 1]`. |
| `use_dropout_2d` | bool | `False` | Use `nn.Dropout2d` (a whole feature map at once) instead of `nn.Dropout`. |
| `pooling` | `shallowfbcspnet_pooling_config` | `shallowfbcspnet_pooling_config()` | The single pooling stage (the `[pooling]` section). A dict is accepted and promoted. See below. |

### Classification head

| Field | Type | Default | Meaning |
|---|---|---|---|
| `n_classes` | int | `4` | Classifier output classes. |
| `use_softmax` | bool | `False` | `False` (default) → `Identity` head (raw logits, pair with `cross_entropy` — the numerically stabler default and the recommended default); `True` → `LogSoftmax` head (log-probs, pair with `nll`). |

### Misc

| Field | Type | Default | Meaning |
|---|---|---|---|
| `flatten_output` | bool | `False` | Flatten the plain-`ShallowFBCSPNet` output to `(B, n_features)` (ignored by the classifier). |
| `input_size` | tuple | `(1, 1, 22, 1000)` | `(batch, depth, C, T)` dummy input to size the feedforward layer of the head. |
| `print_var` | bool | `False` | Print param counts / shape tracking at build time. |

> **Faithfulness:** with all defaults a 22×1000 trial flows to `(B, 40, 1, 61)` feature maps, and `ShallowFBCSPNet_Classifier` has **46 084** trainable parameters — exactly the count of `braindecode.models.ShallowFBCSPNet` for the same geometry.

## The `pooling` section — `shallowfbcspnet_pooling_config(generic_config)`

`@dataclass`. Configures the **single** pooling stage, which sits between the squaring nonlinearity and the logarithm. Serialized as the `[pooling]` TOML sub-table (declared last in `ShallowFBCSPNet_config` so it follows all scalar keys).

| Field | Type | Default | Meaning |
|---|---|---|---|
| `pooling_type` | str | `"default"` | `"default"` → the original **overlapping** `nn.AvgPool2d((1, pooling_kernel), (1, pooling_stride))` (or `MaxPool2d`, see `pool_mode`); a 1D pooling name (`avg`, `max`, `blur`, `sinc`, `dwt`, and aliases) → the channel-wise `*_multichannel` wrapper with decimation factor `pooling_stride`; also accepts `identity`/`none`. Case-insensitive. |
| `pooling_kernel` | int | `75` | Averaging-window length of the `"default"` overlapping pool. **Ignored** by the 1D operators. Positive int. |
| `pooling_stride` | int | `15` | Stride of the `"default"` pool **and** decimation factor `M` of the 1D operators (new #samples = `T / M`). Positive int. `dwt` additionally requires it to be a power of two, so the original 15 is rejected with it — use 16. |
| `pool_mode` | str | `"mean"` | Which pooling the `"default"` type builds: `"mean"` (`nn.AvgPool2d`, the original) or `"max"` (`nn.MaxPool2d`). Ignored by the 1D operators, where the operator itself is the choice. |
| `params` | dict | `{}` | Extra type-specific params for a 1D operator (e.g. `{"wavelet": "sym4"}`, `{"cutoff_scale": 0.8}`). Empty → the layer's own defaults. Ignored by `"default"`/`"identity"`. |

The derived property `pooling_layer_config` assembles the config the model consumes: for an **active 1D operator** (a type in `POOLING_CONFIG_REGISTRY`) it returns a proper [`pooling_config`](pooling_1d_cfg.md) **dataclass** (built via `get_pooling_config`, decimation factor `pooling_stride` + `params`), so the whole flow is dataclass-based; for the model-only `default` / `identity` / `none` types, which have no config dataclass, it returns the plain `{pooling_type, pooling_kernel, pooling_stride, pool_mode, **params}` **dict**. `ShallowFBCSPNet_config` re-exposes it (delegating to `self.pooling`) so the model can feed it to [`get_shallowfbcspnet_pooling`](../../model/ShallowFBCSPNet.md#get_shallowfbcspnet_poolingconfig---nnmodule).

> **Why this shape (and not the EEGNet `eegnet_pooling_config`).** Like the EEG Conformer's, the original pooling here is *overlapping* (window 75, stride 15), which the `pooling_1d` "output = `T // M`" contract cannot express — so the window (`pooling_kernel`) and the decimation (`pooling_stride`) are separate keys, and it is a copy of [`conformer_pooling_config`](EEGConformer_cfg.md) plus `pool_mode`. There is only **one** stage (vs EEGNet's two, vs [Deep4Net](Deep4Net_cfg.md)'s four).

## Validation (`__post_init__`)

- `name` lowercased and checked against `ALLOWED_NAMES`; `conv_nonlin` and `pool_nonlin` lowercased and checked against `NONLINEARITIES`.
- Boolean flags (`split_first_layer`, `batch_norm`, `use_dropout_2d`, `use_softmax`, `flatten_output`, `print_var`) must be genuine `bool`.
- The sizes (`C`, `T`, `depth_first_layer`, `n_filters_time`, `filter_time_length`, `n_filters_spat`) must be positive ints (a `bool` is rejected); `n_classes` must be an int `>= 2`.
- **`filter_time_length <= T`** (the temporal convolution is unpadded, so it must fit in the window), else `ValueError`.
- `prob_dropout` and `batch_norm_alpha` must be floats in `[0, 1]`.
- `pooling` is promoted from a dict to `shallowfbcspnet_pooling_config` if needed. Its `__post_init__` validates `pooling_type` against the pooling registry + `default`/`identity`/`none`, `pool_mode` against `POOL_MODES`, `pooling_kernel`/`pooling_stride` as positive ints, `params` as a dict, and — for an active 1D type — round-trips `{pooling_type, decimation_factor, **params}` through [`get_pooling_config`](pooling_1d_cfg.md) to validate the type-specific params (e.g. the `dwt` power-of-two decimation factor).
- `input_size` is coerced to a 4-tuple of positive ints.

Module-level helpers `_is_int`, `_check_positive_int`, `_check_bool`, `_check_probability` back these checks.

## Consumption / gotchas

`ShallowFBCSPNet.__init__` / `ShallowFBCSPNet_Classifier.__init__` call `get_model_config(config)` and then read the attributes directly — all defaulting/validation happens here, not in the model. `jade.training.support_training.cross_check_config` reads `use_softmax` and `n_classes` off this config (via `getattr`), so it plugs into the training pipeline exactly like EEGNet. A commented TOML skeleton ships as `src/jade/config/default_config/model/ShallowFBCSPNet.toml` (see [`cli/default_config`](../../cli/default_config.md)).
