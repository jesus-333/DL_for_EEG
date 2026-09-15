# `model/Deep4Net.py` — `Deep4Net` and `Deep4Net_Classifier`

Source: `src/jade/model/Deep4Net.py`

PyTorch implementation of the **Deep ConvNet** (Schirrmeister et al., *Deep learning with convolutional neural networks for EEG decoding and visualization*, Human Brain Mapping 2017, [paper](https://doi.org/10.1002/hbm.23730)) — known by the `Deep4Net` name of its reference implementation — as a **feature extractor** (`Deep4Net`) plus a **classifier wrapper** (`Deep4Net_Classifier`). It is the deep counterpart of [`ShallowFBCSPNet`](ShallowFBCSPNet.md): a split temporal + spatial convolution closed by a pooling stage, then three convolution + pooling blocks with 50, 100 and 200 filters. Both classes are built via [`get_model`](__init__.md).

This is a jade-native re-implementation (mirroring the [`EEGConformer`](EEGConformer.md) structure), rewritten so that:
- it is driven by a validated [`Deep4Net_config`](../config/model/Deep4Net_cfg.md) (a dict is accepted and promoted via `get_model_config`): the first line of each `__init__` is `config = get_model_config(config)`, and the class then reads the config **attributes** — all defaulting/validation lives in the config;
- it works on the jade layout `(B, depth, C, T)` with **time last**, instead of the reference `B × F × T × 1` layout, so the [`pooling_1d`](pooling_1d/0_module_quickstart.md) channel-wise wrappers drop in with no axis juggling — see [`get_deep4net_pooling`](#get_deep4net_poolingconfig---nnmodule);
- **all four** pooling stages are configured independently, so the anti-aliasing ablation can be run on one block at a time;
- the head is the usual jade flatten + `Linear`, not the reference convolutional classifier;
- it depends only on `torch` — no `einops`, no `braindecode`.

Imports `torch`, `torch.nn`, `jade.model.support_functions`, and `jade.config.model.get_model_config`.

> **Two reference options deliberately not reproduced.** `stride_before_pool` moves the decimation from the pooling to the convolution and leaves the pooling with stride 1; an overlapping, *non-decimating* pooling has no counterpart in `pooling_1d`, whose operators are decimators by construction, so supporting it would make half the ablation meaningless. The `mapping` dict only exists to load the state dict of an older version of the reference code.

---

## `get_deep4net_pooling(config) -> nn.Module`

Module-level factory that builds the pooling layer of **one** block, from a single stage of the config `[pooling]` section (as assembled by the `pooling_layer_N_config` properties of [`deep4net_pooling_config`](../config/model/Deep4Net_cfg.md)).

| `pooling_type` | Built layer |
|---|---|
| `'default'` | The original pooling: a non-overlapping `nn.MaxPool2d((1, decimation_factor))` — or `nn.AvgPool2d` when `pool_mode_N = 'mean'` (pooling along the time axis only). |
| `'avg'` / `'max'` / `'blur'` / `'sinc'` / `'dwt'` (and aliases) | The matching `*_multichannel` [`pooling_1d`](pooling_1d/0_module_quickstart.md) wrapper, applied channel-wise — resolved through `support_functions.get_multichannel_pooling`. An active 1D operator arrives as an already-validated `pooling_config` dataclass and is handed straight to that factory. |
| `'identity'` / `'none'`, or a negative factor | `nn.Identity` (that stage is disabled). |

> **Unlike ShallowFBCSPNet and the EEG Conformer, nothing is lost here.** Deep4Net's original pooling is already *non-overlapping* (a window of 3 samples every 3), so `'default'` and the 1D operators produce the same output length and the four stages stay perfectly interchangeable. The whole 22×1000/4-class model has **282 879** trainable parameters with every pooling type (barring `dwt`, which needs a different factor).

---

## `class Deep4Net(nn.Module)`

The Deep ConvNet **without** the final classification layer.

### Tensor shapes

- **Input:** 4D `(B, depth_first_layer, C, T)` — batch, input depth (1 for raw EEG; >1 for time-frequency data), EEG channels `C`, time samples `T`.
- **Output:** 4D `(B, n_filters_4, 1, T_4)` — **unless** `flatten_output=True`, in which case `x.flatten(1)` gives `(B, n_filters_4 * T_4)`. With the defaults, `(B, 200, 1, 7)`.

### Config attributes (read from the `Deep4Net_config`)

The exhaustive field list, types, defaults and validation are on the [`Deep4Net_config` page](../config/model/Deep4Net_cfg.md). What `Deep4Net` reads: the first-block fields (`depth_first_layer`, `n_filters_time`, `filter_time_length`, `n_filters_spat`, `C`, `split_first_layer`), the deeper-block sizes (`n_filters_2/3/4`, `filter_length_2/3/4`), the shared options (`activation`, `pool_nonlin`, `batch_norm`, `batch_norm_alpha`, `prob_dropout`, `use_dropout_2d`), the `pooling` section, plus `flatten_output`, `print_var` and `input_size` (for `debug_shape`).

### Architecture

Four block attributes, each shrinking the time axis:

1. **`self.first_block`** (an `nn.Sequential`) — `Conv2d(depth_first_layer, n_filters_time, (1, filter_time_length))` (temporal conv) → `Conv2d(n_filters_time, n_filters_spat, (C, 1))` (spatial conv, collapses the EEG-channel axis to 1) → `BatchNorm2d` (or `Identity`) → activation → **pooling stage 1** → pool nonlinearity. When `split_first_layer=False` the two convolutions become one `Conv2d(depth_first_layer, n_filters_time, (C, filter_time_length))`.
   It is assembled inline rather than reusing `conv_pool_block` because it is the only block **without** a leading dropout.
2. **`self.second_block` / `self.third_block` / `self.fourth_block`** — three `conv_pool_block`s with `(25→50, 10)`, `(50→100, 10)` and `(100→200, 10)`.

### `forward(x) -> torch.tensor`

Runs the four blocks in order, then returns `x.flatten(1)` if `flatten_output` else the 4D tensor.

### `debug_shape(self, input_size)`

Diagnostic (invoked when `print_var=True`, using the config's `input_size`). Creates a random tensor and prints the shape after each of the four blocks and after the optional flatten. For the defaults the time axis goes `1000 → 330 → 107 → 32 → 7`.

### Building-block helper class (same file, `snake_case` like `pooling_1d`)

| Class | Role |
|---|---|
| `conv_pool_block` | One of the three deeper blocks: dropout → `Conv2d(n_filters_in, n_filters_out, (1, filter_length))` → `BatchNorm2d` (or `Identity`) → activation → pooling → pool nonlinearity, in that order (the dropout comes **first**, as in the reference). Takes the whole `config` for the shared options, plus the stage-specific scalars and the already-built pooling layer. |

---

## `class Deep4Net_Classifier(nn.Module)`

`Deep4Net` + a feedforward classification head.

### Construction (`Deep4Net_config` or dict)

- `config = get_model_config(config)`.
- Builds `self.deep4net = Deep4Net(config)`, then **forces `self.deep4net.flatten_output = True`** — the head needs flat `(B, features)` input, so the wrapped extractor must flatten regardless of the config's `flatten_output`.
- Computes the flattened feature count with `self.compute_number_of_neurons(config.input_size)` (a dummy forward pass).
- Builds `self.classifier = nn.Sequential(Linear(input_neurons, n_classes), LogSoftmax(dim=1) if use_softmax else Identity)`.

> **Flatten + `Linear` vs the reference `conv_classifier`.** As for `ShallowFBCSPNet`, the reference classifies with a `Conv2d` spanning the whole remaining time axis: the same linear map with the same parameter count. The 22×1000/4-class default has **282 879** trainable parameters, matching `braindecode.models.Deep4Net` exactly.

### Methods

- **`forward(x)`** — `x = self.deep4net(x); return self.classifier(x)`.
- **`compute_number_of_neurons(input_size)`** — runs `self.deep4net(torch.rand(input_size))` and returns `len(x.flatten())`.
- **`classify(x, return_as_index=True)`** — calls `forward`; if `return_as_index`, returns `argmax(exp(logits), dim=1)`. Same "do **not** `squeeze` first" comment as `EEGNet_Classifier`.

### Gotchas

- **The time axis can collapse.** Four unpadded convolutions plus four poolings shrink `T` fast (`1000 → 7` with the defaults). A short window, a long kernel or a large factor can drive it to zero mid-network; `Deep4Net_config.__post_init__` walks the four blocks and raises a clear `ValueError` before torch produces a cryptic shape error.
- **`dwt` cannot use the original factor.** The wavelet operator needs a power-of-two decimation factor, so `pooling_factor_N = 3` is rejected at config build time; use 2 or 4.
- **Per-stage configuration is the point.** Setting only `pooling_type_1` (and leaving the other three at `'default'`) isolates the effect of anti-aliasing in the first, highest-bandwidth block — which is the interesting ablation.
- **No explicit weight initialization.** The reference applies `xavier_uniform_` / `constant_`; jade leaves the PyTorch defaults everywhere, and this model follows that convention.
- **Relation to the braindecode wrapper.** `Deep4Net` is also reachable through [`braindecode_pooling_wrapper`](braindecode_pooling_extension.md), which swaps all four pooling layers of the *braindecode* model at once. That path still works (and is covered by `test/custom/braindecode_pooling.py`), but this native implementation is the one to prefer: no optional dependency, validated hyper-parameters, and per-stage control the wrapper cannot offer.
