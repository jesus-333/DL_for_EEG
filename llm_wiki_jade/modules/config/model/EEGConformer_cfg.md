# `config/model/EEGConformer_cfg.py`

Source: `src/jade/config/model/EEGConformer_cfg.py`

Defines `EEGConformer_config`, the validated dataclass consumed by [`EEGConformer` and `EEGConformer_Classifier`](../../model/EEGConformer.md). One config covers both models because the classifier's key set is a superset of the plain extractor's; the `name` field selects which one `jade.model.get_model` builds. Every field has a BCI IV-2a style default that reproduces the original EEG Conformer architecture (Song et al., 2023), so `EEGConformer_config()` is valid on its own. Torch-free like the rest of `jade.config`.

## Module constants

- `ALLOWED_NAMES = ("eegconformer", "eegconformer_classifier", "eegconformerclassifier")` — the accepted values of `name`.
- `ACTIVATIONS = ("relu", "elu", "selu", "gelu")` — accepted activation names (validated here so the config stays torch-free; runtime resolution is done by `jade.model.support_functions.get_activation`).

## Class `EEGConformer_config(model_config)`

`@dataclass`. Fields (grouped as in the source):

### Discriminator

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | str | `"eegconformer_classifier"` | Which model to build (`eegconformer` = feature extractor, or `eegconformer_classifier` = extractor + MLP head). Normalized + membership-checked against `ALLOWED_NAMES`. |

### Patch embedding (convolutional front end)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `C` | int | `22` | EEG channels. Sets the height of the spatial conv kernel `(C, 1)`, which collapses the channel axis to 1. |
| `T` | int | `1000` | Time samples. |
| `depth_first_layer` | int | `1` | Input depth (>1 for time-frequency input). |
| `filter_temporal` | int | `40` | Filters of **both** the temporal and the spatial convolution. |
| `c_kernel_temporal` | tuple | `(1, 25)` | Temporal conv kernel (time axis only). |
| `emb_size` | int | `40` | Transformer token embedding size. The `(1,1)` projection conv maps `filter_temporal → emb_size`. Must be divisible by `num_heads`. |
| `activation` | str | `"elu"` | One of `ACTIVATIONS`. Used in the patch embedding and the MLP head. |
| `prob_dropout` | float | `0.5` | Dropout after the pooling stage, in `[0, 1]`. |
| `pooling` | `conformer_pooling_config` | `conformer_pooling_config()` | The single pooling stage (the `[pooling]` section). A dict is accepted and promoted. See below. |

### Transformer encoder

| Field | Type | Default | Meaning |
|---|---|---|---|
| `depth` | int | `6` | Number of stacked transformer encoder blocks. |
| `num_heads` | int | `10` | Self-attention heads. `emb_size` must be divisible by it. |
| `transformer_dropout` | float | `0.5` | Dropout of the attention weights **and** the two post-sublayer dropouts (the reference `drop_p`). |
| `forward_expansion` | int | `4` | Hidden size of the feedforward block = `forward_expansion * emb_size`. |
| `forward_dropout` | float | `0.5` | Dropout inside the feedforward block (the reference `forward_drop_p`). |

### Classification head

| Field | Type | Default | Meaning |
|---|---|---|---|
| `hidden_size_1` | int | `256` | First hidden layer of the MLP head. |
| `hidden_size_2` | int | `32` | Second hidden layer of the MLP head. |
| `head_dropout_1` | float | `0.5` | Dropout after the first hidden layer. |
| `head_dropout_2` | float | `0.3` | Dropout after the second hidden layer. |
| `n_classes` | int | `4` | Classifier output classes. |
| `use_softmax` | bool | `False` | `False` (default) → `Identity` head (raw logits, pair with `cross_entropy` — the numerically stabler default and the recommended default); `True` → `LogSoftmax` head (log-probs, pair with `nll`). |

### Misc

| Field | Type | Default | Meaning |
|---|---|---|---|
| `flatten_output` | bool | `False` | Flatten the plain-`EEGConformer` token sequence to `(B, n_tokens*emb_size)` (ignored by the classifier). |
| `input_size` | tuple | `(1, 1, 22, 1000)` | `(batch, depth, C, T)` dummy input to size the first layer of the MLP head. |
| `print_var` | bool | `False` | Print param counts / shape tracking at build time. |

> **Faithfulness:** with all defaults, a 22×1000 trial flows to **61 tokens** of size 40 (61·40 = 2440, the value hard-coded in the reference `fc`), so the default architecture matches the original.

## The `pooling` section — `conformer_pooling_config(generic_config)`

`@dataclass`. Configures the **single** pooling stage of the patch embedding (which "acts as slicing to obtain the patches along the time dimension", the ViT-style patch splitting). Serialized as the `[pooling]` TOML sub-table (declared last in `EEGConformer_config` so it follows all scalar keys).

| Field | Type | Default | Meaning |
|---|---|---|---|
| `pooling_type` | str | `"default"` | `"default"` → the original **overlapping** `nn.AvgPool2d((1, pooling_kernel), (1, pooling_stride))`; a 1D pooling name (`avg`, `max`, `blur`, `sinc`, `dwt`, and aliases) → the channel-wise `*_multichannel` wrapper with decimation factor `pooling_stride`; also accepts `identity`/`none`. Case-insensitive. |
| `pooling_kernel` | int | `75` | Averaging-window length of the `"default"` overlapping pool. **Ignored** by the 1D operators. Positive int. |
| `pooling_stride` | int | `15` | Stride of the `"default"` pool **and** decimation factor `M` of the 1D operators (new #samples = `T / M`). Positive int. `dwt` additionally requires it to be a power of two. |
| `params` | dict | `{}` | Extra type-specific params for a 1D operator (e.g. `{"wavelet": "sym4"}`, `{"cutoff_scale": 0.8}`). Empty → the layer's own defaults. Ignored by `"default"`/`"identity"`. |

The derived property `pooling_layer_config` assembles the per-stage config the model consumes: for an **active 1D operator** (a type in `POOLING_CONFIG_REGISTRY`) it returns a proper [`pooling_config`](pooling_1d_cfg.md) **dataclass** (built via `get_pooling_config`, decimation factor `pooling_stride` + `params`), so the whole flow is dataclass-based; for the Conformer-only `default` / `identity` / `none` types, which have no config dataclass, it returns the plain `{pooling_type, pooling_kernel, pooling_stride, **params}` **dict**. `EEGConformer_config` re-exposes it (delegating to `self.pooling`) so `EEGConformer` can read `config.pooling_layer_config` and feed it to [`get_conformer_pooling`](../../model/EEGConformer.md#get_conformer_poolingconfig---nnmodule) (which routes `"default"` to an overlapping `AvgPool2d` and a 1D type to [`get_multichannel_pooling`](../../model/support_functions.md)).

> **Why a bespoke section (not the EEGNet `eegnet_pooling_config`).** EEGNet's pooling is a single non-overlapping factor per stage; the Conformer's default pool is *overlapping* (kernel 75, stride 15), which the `pooling_1d` "output = `T // M`" contract cannot express. `conformer_pooling_config` therefore separates the window (`pooling_kernel`) from the decimation (`pooling_stride`), and there is only **one** stage (vs EEGNet's two).

## Validation (`__post_init__`)

- `name` lowercased and checked against `ALLOWED_NAMES`; `activation` lowercased and checked against `ACTIVATIONS`.
- Boolean flags (`use_softmax`, `flatten_output`, `print_var`) must be genuine `bool`.
- The sizes (`C`, `T`, `depth_first_layer`, `filter_temporal`, `emb_size`, `depth`, `num_heads`, `forward_expansion`, `hidden_size_1`, `hidden_size_2`) must be positive ints (a `bool` is rejected); `n_classes` must be an int `>= 2`.
- **`emb_size % num_heads == 0`** (the attention splits the embedding evenly across the heads), else `ValueError`.
- The dropout probabilities (`prob_dropout`, `transformer_dropout`, `forward_dropout`, `head_dropout_1`, `head_dropout_2`) must be floats in `[0, 1]`.
- `c_kernel_temporal` is coerced list→tuple and must be a 2-tuple of positive ints; `input_size` is coerced to a 4-tuple of positive ints.
- `pooling` is promoted from a dict to `conformer_pooling_config` if needed. Its `__post_init__` validates `pooling_type` against the pooling registry + `default`/`identity`/`none`, `pooling_kernel`/`pooling_stride` as positive ints, `params` as a dict, and — for an active 1D type — round-trips `{pooling_type, decimation_factor, **params}` through [`get_pooling_config`](pooling_1d_cfg.md) to validate the type-specific params (e.g. the `dwt` power-of-two decimation factor, the `sinc` knobs).

Module-level helpers `_is_int`, `_check_positive_int`, `_check_bool`, `_check_probability`, `_coerce_kernel` back these checks.

## Consumption / gotchas

`EEGConformer.__init__` / `EEGConformer_Classifier.__init__` call `get_model_config(config)` and then read the attributes directly — all defaulting/validation happens here, not in the model. `jade.training.support_training.cross_check_config` reads `use_softmax` and `n_classes` off this config (via `getattr`), so the Conformer plugs into the training pipeline exactly like EEGNet.
