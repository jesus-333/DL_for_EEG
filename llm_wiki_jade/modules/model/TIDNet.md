# `model/TIDNet.py` — `TIDNet` and `TIDNet_Classifier`

Source: `src/jade/model/TIDNet.py`

PyTorch implementation of **TIDNet**, the Thinker Invariance DenseNet (Kostas & Rudzicz, *Thinker invariance: enabling deep neural networks for BCI across more people*, Journal of Neural Engineering 2020, [paper](https://doi.org/10.1088/1741-2552/abb7a7)), as a **feature extractor** (`TIDNet`) plus a **classifier wrapper** (`TIDNet_Classifier`). The network chains a residual dilated temporal filter, a decimating pooling stage, a densely connected spatial filter that collapses the EEG-channel axis, and an adaptive summary of the time axis. Both classes are built via [`get_model`](__init__.md).

This is a jade-native re-implementation (mirroring the [`EEGConformer`](EEGConformer.md) structure), rewritten so that:
- it is driven by a validated [`TIDNet_config`](../config/model/TIDNet_cfg.md) (a dict is accepted and promoted via `get_model_config`): the first line of each `__init__` is `config = get_model_config(config)`, and the class then reads the config **attributes** — all defaulting/validation lives in the config;
- it works on the jade layout `(B, depth, C, T)` directly (the reference expands a `(B, C, T)` tensor internally), so the decimating pooling stage takes a [`pooling_1d`](pooling_1d/0_module_quickstart.md) operator with no axis juggling — see [`get_tidnet_pooling`](#get_tidnet_poolingconfig---nnmodule);
- the **auxiliary subject and run classification heads are dropped** (see the [classifier section](#class-tidnet_classifiernnmodule));
- it depends only on `torch` — no `braindecode`.

Imports `math`, `torch`, `torch.nn`, `weight_norm` (from `torch.nn.utils.parametrizations`, falling back to the deprecated `torch.nn.utils` location), `jade.model.support_functions`, and `jade.config.model.get_model_config`.

---

## `get_tidnet_pooling(config) -> nn.Module`

Module-level factory that builds the decimating pooling layer closing the temporal block, from the config `[pooling]` section (as assembled by `tidnet_pooling_config.pooling_layer_config`).

| `pooling_type` | Built layer |
|---|---|
| `'default'` | The original pooling: a non-overlapping `nn.MaxPool2d((1, decimation_factor))` — or `nn.AvgPool2d` when `pool_mode = 'mean'` (pooling along the time axis only). |
| `'avg'` / `'max'` / `'blur'` / `'sinc'` / `'dwt'` (and aliases) | The matching `*_multichannel` [`pooling_1d`](pooling_1d/0_module_quickstart.md) wrapper, applied channel-wise — resolved through `support_functions.get_multichannel_pooling`. |
| `'identity'` / `'none'` | `nn.Identity` (pooling disabled). |

Unlike the other models, this pooling runs **before** the EEG-channel axis is collapsed, on a `(B, t_filters, C, T)` tensor. That is exactly what the channel-wise wrappers are for: they reshape `(B, D, C, T) → (B, D*C, T)`, filter, decimate the last axis and reshape back, so every EEG channel is decimated with the same kernel.

> **The adaptive pooling is not part of this section.** `nn.AdaptiveAvgPool1d(summary)`, which closes the feature extractor, is an adaptive *resize*, not a decimation by an integer factor, so it has no anti-aliasing counterpart in `pooling_1d` and is left alone. As TIDNet's original pooling is non-overlapping, `'default'` and the 1D operators produce the same length, and the whole 22×1000/4-class model has **240 404** trainable parameters with every pooling type (barring `dwt`, which needs a different factor).

---

## `class TIDNet(nn.Module)`

TIDNet **without** the final classification layer.

### Tensor shapes

- **Input:** 4D `(B, depth_first_layer, C, T)` — batch, input depth (1 for raw EEG; >1 for time-frequency data), EEG channels `C`, time samples `T`.
- **Output:** 3D `(B, t_filters + s_growth * spat_layers, summary)` — the EEG-channel axis is gone, collapsed by the spatial filter — **unless** `flatten_output=True`, in which case `x.flatten(1)` gives `(B, n_features)`. With the defaults, `(B, 80, 66)`.

### Config attributes (read from the `TIDNet_config`)

The exhaustive field list, types, defaults and validation are on the [`TIDNet_config` page](../config/model/TIDNet_cfg.md). What `TIDNet` reads: the geometry (`C`, `T`, `depth_first_layer`), the temporal fields (`t_filters`, `temp_layers`, `temp_span`), the spatial fields (`s_growth`, `spat_layers`, `bottleneck`, `summary`), the shared options (`activation`, `prob_dropout`), the `pooling` section, plus `flatten_output`, `print_var` and `input_size` (for `debug_shape`).

### Architecture

Three stages:

1. **`self.temporal_block`** (an `nn.Sequential`) — `temporal_filter(depth_first_layer, t_filters, temp_layers, temp_len)` → **pooling** (via `get_tidnet_pooling`) → `Dropout2d(prob_dropout)`. The temporal kernel length is `self.temp_len = ceil(temp_span * T)`, so it scales with the input window (25 samples · 2 = 50 for the defaults).
2. **`self.spatial_block`** — `dense_spatial_filter(...)`: a stack of `spat_layers` dense layers that grow the feature axis, then a `(C, 1)` convolution that collapses the EEG-channel axis, squeezed away. `(B, 32, 22, 66) → (B, 80, 66)` with the defaults.
3. **`self.extract_features`** — `nn.AdaptiveAvgPool1d(summary)`, where `summary = T // pooling_factor` when the config's `summary` is the `-1` sentinel. With the default it is a no-op that only *pins* the feature length; an explicit `summary` makes the feature vector independent of `T`.

`self.n_features = (t_filters + s_growth * spat_layers) * summary` records the flattened size.

### `forward(x) -> torch.tensor`

Runs `temporal_block -> spatial_block -> extract_features`, then returns `x.flatten(1)` if `flatten_output` else the 3D tensor.

### `debug_shape(self, input_size)`

Diagnostic (invoked when `print_var=True`, using the config's `input_size`). Creates a random tensor and prints the shape after the temporal block, the spatial block, the adaptive summary and the optional flatten.

### Building-block helper classes (same file, `snake_case` like `pooling_1d`)

| Class | Role |
|---|---|
| `batch_norm_zg` | `nn.BatchNorm2d` whose affine parameters are initialized to **zero** ("zero gamma"), so a residual block starts as an identity. Used only by the residual variant of `conv_block_2d`. |
| `conv_block_2d` | Convolution → `Dropout2d` → activation → batch normalization, in that order. **Faithful quirk:** the normalization comes *after* the activation, not before it, matching the reference implementation. Optional residual addition (which then uses `batch_norm_zg`). |
| `dense_filter` | One DenseNet layer: `BatchNorm2d` → activation → `1x1` bottleneck conv (to `bottleneck * growth_rate` maps) → `BatchNorm2d` → activation → `(filter_len, 1)` conv along the EEG-channel axis (padded, so that axis is preserved) → `Dropout2d`, with the result **concatenated** to the input on the feature axis. |
| `dense_spatial_filter` | A stack of `depth` `dense_filter`s, then a `conv_block_2d` with a `(C, 1)` kernel that collapses the EEG-channel axis, followed by `squeeze(-2)`. |
| `temporal_filter` | A stack of `depth` weight-normalized `(1, temp_len)` convolutions with **decreasing** dilation (the `i`-th one uses `depth - i`), each followed by the activation and `Dropout2d`, padded so the time axis is preserved. The whole stack sits in a `netwise` residual whose `1x1` convolution only adapts the feature count. `temp_len` is rounded up to the next odd value so the dilation-aware padding is symmetric. |

---

## `class TIDNet_Classifier(nn.Module)`

`TIDNet` + a feedforward classification head.

### Construction (`TIDNet_config` or dict)

- `config = get_model_config(config)`.
- Builds `self.tidnet = TIDNet(config)`, then **forces `self.tidnet.flatten_output = True`** — the head needs flat `(B, features)` input, so the wrapped extractor must flatten regardless of the config's `flatten_output`.
- Computes the flattened feature count with `self.compute_number_of_neurons(config.input_size)` (a dummy forward pass).
- Builds `self.classifier = nn.Sequential(Linear(input_neurons, n_classes), LogSoftmax(dim=1) if use_softmax else Identity)`.

> **The auxiliary heads are gone.** The original `TIDNet.forward` returns a dict with `prediction`, `subject`, `run` and `features`, used for its domain-adaptation experiments. This port (like braindecode's) returns a plain tensor, so it plugs into [`jade.training`](../training/0_module_quickstart.md) — which expects a tensor — unchanged. The 22×1000/4-class default has **240 404** trainable parameters, matching `braindecode.models.TIDNet` exactly.

### Methods

- **`forward(x)`** — `x = self.tidnet(x); return self.classifier(x)`.
- **`compute_number_of_neurons(input_size)`** — runs `self.tidnet(torch.rand(input_size))` and returns `len(x.flatten())`.
- **`classify(x, return_as_index=True)`** — calls `forward`; if `return_as_index`, returns `argmax(exp(logits), dim=1)`. Same "do **not** `squeeze` first" comment as `EEGNet_Classifier`.

### Gotchas

- **`use_softmax=True` is the faithful setting.** The original TIDNet always ends with a log-softmax. jade defaults to `False` (raw logits + `cross_entropy`) because that is numerically stabler; set it to `True` only together with the `nll` loss — see [`cross_check_config`](../training/support_training.md).
- **`dwt` cannot use the original factor.** The wavelet operator needs a power-of-two decimation factor, so `pooling_factor = 15` is rejected at config build time; use 16.
- **`pooling_factor` also changes the feature size.** With the default `summary = -1` the adaptive pooling targets `T // pooling_factor`, so changing the factor changes the head width. Set an explicit `summary` to keep the head fixed across an ablation.
- **The temporal kernel scales with `T`.** `temp_len = ceil(temp_span * T)`, so a shorter window automatically gives a shorter kernel; `temp_span` is validated to be in `(0, 1]`.
- **No explicit weight initialization.** The reference applies `xavier_normal_` to the classifier; jade leaves the PyTorch defaults everywhere, and this model follows that convention.
- **Not covered by the braindecode wrapper.** Unlike `ShallowFBCSPNet` and `Deep4Net`, TIDNet was never declared supported by [`braindecode_pooling_wrapper`](braindecode_pooling_extension.md) (its pooling lives inside residual dense blocks, and it also uses adaptive pooling, which the wrapper ignores). This native implementation is the only way to run the pooling ablation on TIDNet.
