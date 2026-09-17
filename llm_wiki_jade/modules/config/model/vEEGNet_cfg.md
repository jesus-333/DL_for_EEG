# `config/model/vEEGNet_cfg.py` — `vEEGNet_config`

Source: `src/jade/config/model/vEEGNet_cfg.py`

Configuration of [`vEEGNet`](../../model/vEEGNet.md), the variational autoencoder whose encoder is an EEGNet and whose decoder mirrors it. Registered in [`MODEL_CONFIG_REGISTRY`](__init__.md) under `veegnet` (plus the alias `vae_eegnet`) and consumed by [`get_model`](../../model/__init__.md).

It is the **first generative model config**: it sets `task = "generative"` (see [`generic.md`](generic.md)), which is what tells [`cross_check_config`](../../training/support_training.md) that this model must be paired with the [ELBO](../loss_functions/elbo.md) and not with a classification loss.

Every field has a sensible (BCI IV-2a style) default, so `vEEGNet_config()` with no argument is valid. The config is torch-free like the rest of `jade.config`.

---

## Why the encoder fields are duplicated instead of nested

The obvious design would be to nest an [`EEGNet_config`](EEGNet_cfg.md) under an `encoder` field. **It is not what this config does**, and the reason is worth recording, because the DRY argument points the other way:

- **House style.** `Deep4Net_config`, `TIDNet_config`, `EEGConformer_config` and `ShallowFBCSPNet_config` already re-declare every field they share with EEGNet (`C`, `T`, `depth_first_layer`, `activation`, `prob_dropout`, `input_size`) **and** their own private validation helpers. Nesting here would make vEEGNet the odd one out.
- **Two `name` keys in one file.** A nested config would put `name = "veegnet"` at the top level and `name = "eegnet"` inside `[encoder]`, one of which the user must not edit.
- **The `[pooling]` section would move one level deeper** (`[encoder.pooling]`) than in every other model skeleton, for no gain in a hand-edited TOML.
- **Classifier-only fields would become reachable.** `EEGNet_config` carries `n_classes` and `use_softmax`; nested, they would be readable as `model_config.encoder.n_classes` on a model that has no classifier — exactly the confusion the `task` marker exists to prevent.

The bridge in the other direction is [`to_encoder_config`](#to_encoder_config---eegnet_config), so the encoder is still built in exactly one place and no EEGNet default is restated on the model side.

---

## Fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | `"veegnet"` | Discriminator. One of `ALLOWED_NAMES = ("veegnet", "vae_eegnet")`. Lower-cased, so `"vEEGNet"` in a TOML resolves. |
| `use_bias` | `bool` | `False` | Bias term in the convolutions of **both** the encoder and the decoder. |
| `D` | `int` | `2` | Spatial filters per temporal filter. |
| `activation` | `str` | `"elu"` | One of `ACTIVATIONS = ("relu", "elu", "selu", "gelu")`. |
| `prob_dropout` | `float` | `0.5` | Dropout probability, in `[0, 1]`. |
| `use_dropout_2d` | `bool` | `False` | `nn.Dropout2d` instead of `nn.Dropout`. |
| `C` | `int` | `22` | EEG channels. |
| `T` | `int` | `1000` | Time samples. |
| `depth_first_layer` | `int` | `1` | Input depth (1 for raw EEG). |
| `filter_1` | `int` | `8` | Temporal filters. |
| `c_kernel_1` | `(int, int)` | `(1, 64)` | Temporal convolution kernel. |
| `c_kernel_2` | `(int, int)` | `(22, 1)` | Spatial (depthwise) convolution kernel. Its height must not exceed `C`. |
| `filter_2` | `int` | `16` | Filters of the separable convolution. |
| `c_kernel_3` | `(int, int)` | `(1, 16)` | Depthwise kernel of the separable block. |
| `parameters_map_type` | `str` | `"linear"` | How the encoder output becomes `mu` / `log_var`. One of `PARAMETERS_MAP_TYPES = ("linear", "convolutional")`. |
| `hidden_space_dimension` | `int` | `64` | Latent dimension `d` (the value of the paper). **Used only by the `'linear'` mapping.** |
| `upsample_mode` | `str` | `"nearest"` | Interpolation used by the decoder. One of `UPSAMPLE_MODES = ("nearest", "bilinear", "bicubic", "area")`. |
| `input_size` | `(int, int, int, int)` | `(1, 1, 22, 1000)` | Shape of the dummy input that probes the encoder geometry. |
| `print_var` | `bool` | `False` | Print the parameter counts and the shape tracking at build time. |
| `pooling` | `eegnet_pooling_config` | `eegnet_pooling_config()` | The two encoder pooling stages. Declared **last**, so it serializes as a trailing TOML sub-table. |

**`task`** is a `ClassVar[str] = "generative"` — not a field, so it never appears in `to_dict`, `from_dict` or a TOML.

### What is deliberately absent

- **`n_classes` / `use_softmax` / `flatten_output`.** Their absence is load-bearing: the pre-existing checks in `cross_check_config` are `getattr`-guarded on exactly those names, so a vEEGNet config skips them without any change there.
- **`type_vae`.** A dead field in the original implementation (`0 = normal VAE, 1 = conditional VAE`), threaded through every config and never read by any model.
- **The LoRA latent mapping** (`parameters_map_type = 2` in the original) is not carried over.

### `[pooling]` — the EEGNet section, reused as is

`vEEGNet_config` reuses `eegnet_pooling_config` rather than cloning a bespoke section, which is right here and *not* right for the other models: the encoder **is** an EEGNet, with exactly the same two non-overlapping stages. The decoder does not read the factors at all — see the [model page](../../model/vEEGNet.md#key-fact--how-the-pooling-is-inverted-and-why).

---

## `__post_init__` validation

Same structure as [`EEGNet_cfg.py`](EEGNet_cfg.md) (`# %%%%` sub-headers, locally re-declared `_is_int` / `_check_positive_int` / `_check_bool` / `_coerce_kernel` helpers), plus three checks specific to this model:

- **`c_kernel_2[0] <= C`** — the encoder's spatial convolution is unpadded, so a taller kernel is invalid; and the decoder inverts that convolution to restore `C`, so the constraint belongs to the model geometry rather than to torch.
- **`T` long enough** that at least one time sample survives both pooling stages, so a bad combination fails with a readable message instead of a cryptic torch error.
- **`input_size[1:] == (depth_first_layer, C, T)`.** For a classifier `input_size` is a convenience that only sizes the head; **here it drives the whole decoder geometry** through the build-time probe, so a disagreement is a correctness bug, not a redundancy.

---

## `to_encoder_config() -> EEGNet_config`

Builds the configuration of the EEGNet used as encoder. Two fields are **forced** whatever the user asked for:

- `name = "eegnet"`, so the plain feature extractor is built and not its classifier wrapper;
- `flatten_output = False`, because the sampling layer (and, with the convolutional mapping, the decoder) needs the feature map and not a flat vector.

The classifier-only fields are left at their defaults, since a plain `EEGNet` never reads them. The returned config is **fresh** — the original implementation mutated the caller's dict in place, so building two models from one config was order-dependent — but the `pooling` section is shared **by reference**, so the two objects always describe the same pooling stages.

`test/custom/vEEGNet.py` asserts that every shared field matches, as a guard against drift from `EEGNet_config`.

---

## Skeleton

`src/jade/config/default_config/model/vEEGNet.toml`, copied by `create-default-config`. Flat, with a trailing `[pooling]` sub-table, exactly like `EEGNet.toml`.
