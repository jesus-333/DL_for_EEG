# `config/model/Deep4Net_cfg.py`

Source: `src/jade/config/model/Deep4Net_cfg.py`

Defines `Deep4Net_config`, the validated dataclass consumed by [`Deep4Net` and `Deep4Net_Classifier`](../../model/Deep4Net.md). One config covers both models because the classifier's key set is a superset of the plain extractor's; the `name` field selects which one `jade.model.get_model` builds. Every field has a BCI IV-2a style default that reproduces the original Deep ConvNet architecture (Schirrmeister et al., 2017), so `Deep4Net_config()` is valid on its own. Torch-free like the rest of `jade.config`.

## Module constants

- `ALLOWED_NAMES = ("deep4net", "deep4net_classifier", "deep4netclassifier")` — the accepted values of `name`.
- `ACTIVATIONS = ("relu", "leaky_relu", "elu", "selu", "gelu")` — accepted activation names (validated here so the config stays torch-free; runtime resolution is done by `jade.model.support_functions.get_activation`).
- `POOL_NONLINEARITIES = ("identity", "square", "safe_log") + ACTIVATIONS` — accepted values of `pool_nonlin`.
- `POOL_MODES = ("max", "mean")` — which pooling a `'default'` stage builds.
- `POOLING_STAGES = ("1", "2", "3", "4")` — the suffixes of the four per-stage pooling keys, used by the validation loop.

## Class `Deep4Net_config(model_config)`

`@dataclass`. Fields (grouped as in the source):

### Discriminator

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | str | `"deep4net_classifier"` | Which model to build (`deep4net` = feature extractor, or `deep4net_classifier` = extractor + head). Normalized + membership-checked against `ALLOWED_NAMES`. |

### First block (split temporal + spatial convolution)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `C` | int | `22` | EEG channels. Sets the height of the spatial conv kernel `(C, 1)`, which collapses the channel axis to 1. |
| `T` | int | `1000` | Time samples. |
| `depth_first_layer` | int | `1` | Input depth (>1 for time-frequency input). |
| `n_filters_time` | int | `25` | Filters of the temporal convolution of the first block. |
| `filter_time_length` | int | `10` | Length of that temporal conv kernel, i.e. the kernel is `(1, filter_time_length)`. |
| `n_filters_spat` | int | `25` | Filters of the spatial convolution. Ignored when `split_first_layer = False`. |
| `split_first_layer` | bool | `True` | `True` (the original) → temporal `(1, filter_time_length)` then spatial `(C, 1)` conv, with no nonlinearity in between; `False` → a single `(C, filter_time_length)` conv does both at once. |

### Deeper blocks

| Field | Type | Default | Meaning |
|---|---|---|---|
| `n_filters_2` | int | `50` | Filters of the second block. |
| `filter_length_2` | int | `10` | Kernel length of the second block. |
| `n_filters_3` | int | `100` | Filters of the third block. |
| `filter_length_3` | int | `10` | Kernel length of the third block. |
| `n_filters_4` | int | `200` | Filters of the fourth block. |
| `filter_length_4` | int | `10` | Kernel length of the fourth block. |

### Shared block options

| Field | Type | Default | Meaning |
|---|---|---|---|
| `activation` | str | `"elu"` | One of `ACTIVATIONS`. Applied after every convolution. |
| `pool_nonlin` | str | `"identity"` | Nonlinearity after every pooling stage. One of `POOL_NONLINEARITIES`. The original has none, hence `identity`. |
| `batch_norm` | bool | `True` | Add a `BatchNorm2d` after every convolution (the conv then drops its bias, which the normalization would make redundant). |
| `batch_norm_alpha` | float | `0.1` | Momentum of the batch normalization, in `[0, 1]`. Ignored when `batch_norm = False`. |
| `prob_dropout` | float | `0.5` | Dropout at the **beginning** of the second, third and fourth block, in `[0, 1]`. |
| `use_dropout_2d` | bool | `False` | Use `nn.Dropout2d` (a whole feature map at once) instead of `nn.Dropout`. |
| `pooling` | `deep4net_pooling_config` | `deep4net_pooling_config()` | The four pooling stages (the `[pooling]` section). A dict is accepted and promoted. See below. |

### Classification head

| Field | Type | Default | Meaning |
|---|---|---|---|
| `n_classes` | int | `4` | Classifier output classes. |
| `use_softmax` | bool | `False` | `False` (default) → `Identity` head (raw logits, pair with `cross_entropy` — the numerically stabler default and the recommended default); `True` → `LogSoftmax` head (log-probs, pair with `nll`). |

### Misc

| Field | Type | Default | Meaning |
|---|---|---|---|
| `flatten_output` | bool | `False` | Flatten the plain-`Deep4Net` output to `(B, n_features)` (ignored by the classifier). |
| `input_size` | tuple | `(1, 1, 22, 1000)` | `(batch, depth, C, T)` dummy input to size the feedforward layer of the head. |
| `print_var` | bool | `False` | Print param counts / shape tracking at build time. |

> **Faithfulness:** with all defaults a 22×1000 trial flows `1000 → 330 → 107 → 32 → 7` along the time axis, giving `(B, 200, 1, 7)` feature maps, and `Deep4Net_Classifier` has **282 879** trainable parameters — exactly the count of `braindecode.models.Deep4Net` for the same geometry.

## The `pooling` section — `deep4net_pooling_config(generic_config)`

`@dataclass`. Configures the **four** pooling stages, one per convolutional block, each independently — the same per-stage pattern as [`eegnet_pooling_config`](EEGNet_cfg.md), extended from two stages to four and with a `pool_mode_N` key. Serialized as the `[pooling]` TOML sub-table (declared last in `Deep4Net_config` so it follows all scalar keys).

| Field | Type | Default | Meaning |
|---|---|---|---|
| `pooling_type_1` … `pooling_type_4` | str | `"default"` | Pooling operator of that stage. `"default"` → the original non-overlapping `nn.MaxPool2d((1, pooling_factor_N))` (or `AvgPool2d`, see `pool_mode_N`); a 1D pooling name (`avg`, `max`, `blur`, `sinc`, `dwt`, and aliases) → the channel-wise `*_multichannel` wrapper; also accepts `identity`/`none`. Case-insensitive. |
| `pooling_factor_1` … `pooling_factor_4` | int | `3` | Decimation factor `M` of that stage (new #samples = `T / M`). A non-zero int; **negative disables** the stage (a warning is emitted). `dwt` additionally requires a power of two, so the original 3 is rejected with it — use 2 or 4. |
| `pool_mode_1` … `pool_mode_4` | str | `"max"` | Which pooling the `"default"` type builds for that stage: `"max"` (`nn.MaxPool2d`, the original) or `"mean"` (`nn.AvgPool2d`). Ignored by the 1D operators. |
| `params_1` … `params_4` | dict | `{}` | Extra type-specific params for that stage (e.g. `{"wavelet": "sym4"}`, `{"cutoff_scale": 0.8}`). Empty → the layer's own defaults. Ignored by `"default"`/`"identity"`. |

The private helper `_stage_pooling_config(pooling_type, factor, pool_mode, params)` assembles a single stage, and the four properties `pooling_layer_1_config` … `pooling_layer_4_config` expose them. For an **active 1D operator** (a type in `POOLING_CONFIG_REGISTRY`, with a positive factor) the result is a proper [`pooling_config`](pooling_1d_cfg.md) **dataclass** (built via `get_pooling_config`); otherwise it is the plain `{pooling_type, decimation_factor, pool_mode, **params}` **dict**. The model feeds each one to [`get_deep4net_pooling`](../../model/Deep4Net.md#get_deep4net_poolingconfig---nnmodule).

> **Why four independent stages.** Deep4Net's original pooling is *non-overlapping*, so — unlike ShallowFBCSPNet and the EEG Conformer — `"default"` and the 1D operators agree on the output length and the stages are interchangeable with no head re-sizing. Configuring them separately means the ablation can isolate one block: setting only `pooling_type_1` measures anti-aliasing in the first, highest-bandwidth block, which is where aliasing is worst.

## Validation (`__post_init__`)

- `name` lowercased and checked against `ALLOWED_NAMES`; `activation` against `ACTIVATIONS`; `pool_nonlin` against `POOL_NONLINEARITIES`.
- Boolean flags (`split_first_layer`, `batch_norm`, `use_dropout_2d`, `use_softmax`, `flatten_output`, `print_var`) must be genuine `bool`.
- Every size (`C`, `T`, `depth_first_layer`, `n_filters_time`, `filter_time_length`, `n_filters_spat`, `n_filters_2/3/4`, `filter_length_2/3/4`) must be a positive int (a `bool` is rejected); `n_classes` must be an int `>= 2`.
- `prob_dropout` and `batch_norm_alpha` must be floats in `[0, 1]`.
- `pooling` is promoted from a dict to `deep4net_pooling_config` if needed. Its `__post_init__` loops over `POOLING_STAGES` and, per stage, validates the type against the pooling registry + `default`/`identity`/`none`, the mode against `POOL_MODES`, the factor as a non-zero int (warning if negative), the params as a dict, and — for an active 1D type — round-trips `{pooling_type, decimation_factor, **params}` through [`get_pooling_config`](pooling_1d_cfg.md).
- `input_size` is coerced to a 4-tuple of positive ints.
- **The time axis is walked forward** through the four blocks (`n_times = (n_times - filter_length + 1) // factor`, per block): if it ever drops below 1 a clear `ValueError` is raised, instead of letting torch produce a cryptic shape error at the first forward pass. This is the check most likely to fire when adapting the model to a short trial.

Module-level helpers `_is_int`, `_check_positive_int`, `_check_bool`, `_check_probability` back these checks.

## Consumption / gotchas

`Deep4Net.__init__` / `Deep4Net_Classifier.__init__` call `get_model_config(config)` and then read the attributes directly — all defaulting/validation happens here, not in the model. `jade.training.support_training.cross_check_config` reads `use_softmax` and `n_classes` off this config (via `getattr`), so it plugs into the training pipeline exactly like EEGNet. A commented TOML skeleton ships as `src/jade/config/default_config/model/Deep4Net.toml` (see [`cli/default_config`](../../cli/default_config.md)).
