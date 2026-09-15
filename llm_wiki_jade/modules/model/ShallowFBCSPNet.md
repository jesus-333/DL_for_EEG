# `model/ShallowFBCSPNet.py` — `ShallowFBCSPNet` and `ShallowFBCSPNet_Classifier`

Source: `src/jade/model/ShallowFBCSPNet.py`

PyTorch implementation of the **Shallow ConvNet** (Schirrmeister et al., *Deep learning with convolutional neural networks for EEG decoding and visualization*, Human Brain Mapping 2017, [paper](https://doi.org/10.1002/hbm.23730)) — universally known by the `ShallowFBCSPNet` name of its reference implementation — as a **feature extractor** (`ShallowFBCSPNet`) plus a **classifier wrapper** (`ShallowFBCSPNet_Classifier`). The network is a learnable version of the FBCSP pipeline: a temporal convolution (a filter bank), a spatial convolution (the CSP filters), then squaring, pooling and a logarithm, which together compute a log-variance feature. Both classes are built via [`get_model`](__init__.md).

This is a jade-native re-implementation (mirroring the [`EEGConformer`](EEGConformer.md) structure), rewritten so that:
- it is driven by a validated [`ShallowFBCSPNet_config`](../config/model/ShallowFBCSPNet_cfg.md) (a dict is accepted and promoted via `get_model_config`): the first line of each `__init__` is `config = get_model_config(config)`, and the class then reads the config **attributes** — all defaulting/validation lives in the config;
- it works on the jade layout `(B, depth, C, T)` with **time last**, instead of the reference `B × F × T × 1` layout, so the [`pooling_1d`](pooling_1d/0_module_quickstart.md) channel-wise wrappers (which always decimate the last axis) drop in with no axis juggling — see [`get_shallowfbcspnet_pooling`](#get_shallowfbcspnet_poolingconfig---nnmodule);
- the head is the usual jade flatten + `Linear`, not the reference convolutional classifier (see the [classifier section](#class-shallowfbcspnet_classifiernnmodule));
- it depends only on `torch` — no `einops`, no `braindecode`.

Imports `torch`, `torch.nn`, `jade.model.support_functions`, and `jade.config.model.get_model_config`.

---

## `get_shallowfbcspnet_pooling(config) -> nn.Module`

Module-level factory that builds the single pooling layer from the config `[pooling]` section (a [`shallowfbcspnet_pooling_config`](../config/model/ShallowFBCSPNet_cfg.md), or a dict promoted to one). Same three-branch shape as [`get_conformer_pooling`](EEGConformer.md#get_conformer_poolingconfig---nnmodule), and it delegates to [`get_multichannel_pooling`](support_functions.md) for the 1D operators.

| `pooling_type` | Built layer |
|---|---|
| `'default'` | The original pooling: an **overlapping** `nn.AvgPool2d((1, pooling_kernel), (1, pooling_stride))` — or `nn.MaxPool2d` when `pool_mode = 'max'` (pooling along the time axis only). |
| `'avg'` / `'max'` / `'blur'` / `'sinc'` / `'dwt'` (and aliases) | The matching `*_multichannel` [`pooling_1d`](pooling_1d/0_module_quickstart.md) wrapper, applied channel-wise, with decimation factor `pooling_stride` — resolved through `support_functions.get_multichannel_pooling(config.pooling_layer_config)`. |
| `'identity'` / `'none'` | `nn.Identity` (pooling disabled). |

The pooling comes after the spatial convolution has collapsed the EEG-channel axis, so it always runs on a `(B, n_filters_conv, 1, T_conv)` tensor (the channel-wise wrappers see a single "EEG channel" and pool along time).

> **`'default'` vs the 1D operators — a genuine difference.** The paper's pooling is *overlapping* (window 75, stride 15), which does not fit the `pooling_1d` "output length = `T // M`" contract. So `'default'` is kept as a faithful overlapping pool, while a 1D operator decimates by `pooling_stride` only. For a 22×1000 trial the default yields **61** pooled samples and a 46 084-parameter classifier (exactly the reference count), a 1D operator yields **65** and 46 724 parameters. The head is sized with a dummy forward, so either works.

---

## `class ShallowFBCSPNet(nn.Module)`

The Shallow ConvNet **without** the final classification layer. It outputs the log-variance feature maps and can optionally flatten them.

### Tensor shapes

- **Input:** 4D `(B, depth_first_layer, C, T)` — batch, input depth (1 for raw EEG; >1 for time-frequency data), EEG channels `C`, time samples `T`. Convolutions are `nn.Conv2d`, so the input must be 4D (same contract as `EEGNet`).
- **Output:** 4D `(B, n_filters_conv, 1, T_pool)` — **unless** `flatten_output=True`, in which case `x.flatten(1)` gives `(B, n_filters_conv * T_pool)`. With the defaults, `(B, 40, 1, 61)`.

### Config attributes (read from the `ShallowFBCSPNet_config`)

The exhaustive field list, types, defaults and validation are on the [`ShallowFBCSPNet_config` page](../config/model/ShallowFBCSPNet_cfg.md). What `ShallowFBCSPNet` reads: the front-end fields (`depth_first_layer`, `n_filters_time`, `filter_time_length`, `n_filters_spat`, `C`, `split_first_layer`, `batch_norm`, `batch_norm_alpha`), the nonlinearities (`conv_nonlin`, `pool_nonlin`), the dropout (`prob_dropout`, `use_dropout_2d`), the `pooling` section, plus `flatten_output`, `print_var` and `input_size` (for `debug_shape`).

### Architecture

Two `nn.Sequential` attributes:

1. **`self.convolutional_filter`** — `Conv2d(depth_first_layer, n_filters_time, (1, filter_time_length))` (temporal conv, bias on) → `Conv2d(n_filters_time, n_filters_spat, (C, 1))` (spatial conv, collapses the EEG-channel axis to 1; bias off when `batch_norm`) → `BatchNorm2d(n_filters_conv, momentum=batch_norm_alpha)` (or `Identity`) → **conv nonlinearity** (`square` by default).
   When `split_first_layer=False` the two convolutions are replaced by one `Conv2d(depth_first_layer, n_filters_time, (C, filter_time_length))` that mixes channels and time at once, and `n_filters_spat` is ignored.
2. **`self.pooling_block`** — **pooling** (via `get_shallowfbcspnet_pooling`) → **pool nonlinearity** (`safe_log` by default) → dropout (via [`get_dropout`](support_functions.md)).

The `square` / `safe_log` nonlinearities live in [`support_functions`](support_functions.md) (not in this file) because [`Deep4Net`](Deep4Net.md) can use them too; they are selected by name through `support_functions.get_nonlinearity`.

### `forward(x) -> torch.tensor`

Runs `convolutional_filter -> pooling_block`, then returns `x.flatten(1)` if `flatten_output` else the 4D tensor.

### `debug_shape(self, input_size)`

Diagnostic (invoked when `print_var=True`, using the config's `input_size`). Creates a random tensor and prints the shape after the convolutional filter, after the pooling block, and after the optional flatten. Purely for interactive inspection.

---

## `class ShallowFBCSPNet_Classifier(nn.Module)`

`ShallowFBCSPNet` + a feedforward classification head.

### Construction (`ShallowFBCSPNet_config` or dict)

- `config = get_model_config(config)`.
- Builds `self.shallowfbcspnet = ShallowFBCSPNet(config)`, then **forces `self.shallowfbcspnet.flatten_output = True`** — the head needs flat `(B, features)` input, so the wrapped extractor must flatten regardless of the config's `flatten_output` (same footgun-avoidance as `EEGNet_Classifier`).
- Computes the flattened feature count with `self.compute_number_of_neurons(config.input_size)` (a dummy forward pass).
- Builds `self.classifier = nn.Sequential(Linear(input_neurons, n_classes), LogSoftmax(dim=1) if use_softmax else Identity)`.

> **Flatten + `Linear` vs the reference `conv_classifier`.** The reference implementation classifies with a `Conv2d(n_filters_conv, n_outputs, (final_conv_length, 1))` spanning the whole remaining time axis, then squeezes. That is the same linear map as flatten + `Linear`, with the same parameter count — so this head is a notational change, not an architectural one. The 22×1000/4-class default has **46 084** trainable parameters, matching `braindecode.models.ShallowFBCSPNet` exactly.

### Methods

- **`forward(x)`** — `x = self.shallowfbcspnet(x); return self.classifier(x)`.
- **`compute_number_of_neurons(input_size)`** — runs `self.shallowfbcspnet(torch.rand(input_size))` and returns `len(x.flatten())` (use a batch dim of 1 in `input_size` so this equals the per-sample feature count).
- **`classify(x, return_as_index=True)`** — calls `forward`; if `return_as_index`, returns `argmax(exp(logits), dim=1)` (class indices), else the raw forward output. Same "do **not** `squeeze` first" comment as `EEGNet_Classifier`. `jade.training.metrics` prefers this method over `argmax(model(x))`.

### Gotchas

- **`dwt` cannot use the original stride.** The wavelet operator needs a power-of-two decimation factor, so `pooling_stride = 15` is rejected at config build time; use 16 (the config raises a clear `ValueError`).
- **`'identity'` pooling makes the head huge.** With no decimation the flattened feature count is ~16× larger (192 484 parameters instead of 46 084). It is there for completeness, not as a sensible ablation point.
- **The head is sized dynamically**, so a change of `C` / `T` / pooling operator / `pooling_stride` needs no manual edit — unlike the reference implementation's `final_conv_length="auto"` machinery.
- **No explicit weight initialization.** The reference applies `xavier_uniform_` / `constant_`; jade leaves the PyTorch defaults everywhere, and this model follows that convention. Expect slightly different training curves from `braindecode`, not a different architecture.
- **`sinc` / `dwt` pooling need optional deps** (`scipy` for `sinc`, `PyWavelets` for `dwt`), imported lazily inside `pooling_1d`; `avg` / `max` / `blur` / `default` need only `torch`/`numpy`.
- **Relation to the braindecode wrapper.** `ShallowFBCSPNet` is also reachable through [`braindecode_pooling_wrapper`](braindecode_pooling_extension.md), which swaps the pooling of the *braindecode* model. That path still works (and is covered by `test/custom/braindecode_pooling.py`), but this native implementation is the one to prefer: no optional dependency, validated hyper-parameters, and no trailing-classifier re-sizing heuristics.
