# `model/EEGNet.py` — `EEGNet` and `EEGNet_Classifier`

Source: `src/jade/model/EEGNet.py`

PyTorch implementation of EEGNet (Lawhern et al.) as a **feature extractor** (`EEGNet`) plus a thin **classifier wrapper** (`EEGNet_Classifier`). Variable names follow the original paper's nomenclature where possible. Both are built via [`get_model`](__init__.md).

Both constructors now take an [`EEGNet_config`](../config/model/EEGNet_cfg.md) (a dict is accepted and promoted via `get_model_config`): the first line of each `__init__` is `config = get_model_config(config)`, and the class then reads the config **attributes**. All defaulting/validation lives in the config — the constructors no longer back-fill or mutate anything.

Imports `torch`, `torch.nn`, `jade.model.support_functions`, and `jade.config.model.get_model_config`.

---

## `class EEGNet(nn.Module)`

Classic EEGNet **without** the final feedforward + classification layer of the original paper. It outputs features and can optionally flatten them.

### Tensor shapes

- **Input:** 4D `(B, depth_first_layer, C, T)` — batch, input depth (1 for raw EEG; >1 for time-frequency data), EEG channels `C`, time samples `T`. Convolutions are `nn.Conv2d`, so the input must be 4D.
- **Output:** 4D `(B, filter_2, 1, T')` where `T'` is the time axis after the two pooling stages — **unless** `flatten_output=True`, in which case `x.flatten(1)` gives `(B, filter_2 * T')`.

### Config attributes (read from the `EEGNet_config`)

The exhaustive field list, types, defaults and validation are on the [`EEGNet_config` page](../config/model/EEGNet_cfg.md). What `EEGNet` reads:

| Attribute | Meaning |
|---|---|
| `use_bias` | Whether the conv layers use a bias term. |
| `D` | Spatial filters learned per temporal filter (depth multiplier). Total spatial filters = `filter_1 * D`. |
| `activation` | Activation name → resolved by `support_functions.get_activation` (`relu`/`elu`/`selu`/`gelu`). |
| `prob_dropout` | Dropout probability, in `[0, 1]`. |
| `use_dropout_2d` | `nn.Dropout2d` if True else `nn.Dropout` (via `get_dropout`). |
| `filter_1` | Number of temporal filters (block 1). |
| `c_kernel_1` | Kernel size of the temporal conv. |
| `c_kernel_2` | Kernel size of the spatial (depthwise) conv. |
| `filter_2` | Output filters of the separable conv (block 2). |
| `c_kernel_3` | Kernel size of the separable conv's depthwise part. |
| `pooling_layer_1_config` / `pooling_layer_2_config` | Per-stage pooling config derived from the `pooling` section — a [`pooling_config`](../config/model/pooling_1d_cfg.md) dataclass for an active 1D operator, or a `{pooling_type, decimation_factor, **params}` dict for the EEGNet-only `default`/`identity`/disabled cases → each resolved by `support_functions.get_multichannel_pooling` (which accepts either) into a pooling layer. See the [`EEGNet_config` pooling section](../config/model/EEGNet_cfg.md). |
| `C` | EEG channel count. **Stored but NOT used** (`self.C`). |
| `T` | Time samples. **Stored but NOT used** (`self.T`). |
| `depth_first_layer` | Input depth of the first conv (>1 for time-frequency input). |
| `flatten_output` | Flatten the output to `(B, features)`. |
| `print_var` | Print parameter counts and run `debug_shape(input_size)` at construction. |
| `input_size` | `(batch, depth, C, T)`; drives `debug_shape` (and sizes the classifier head). |

> **Note:** `C` and `T` are read into attributes but never used by the architecture (conv kernels are given explicitly via the `c_kernel_*` attributes; pooling comes from the `pooling` section). Since the config is a validated dataclass, every attribute is always present with a valid value — no back-filling / dict mutation happens anymore.

### Architecture (three `nn.Sequential` blocks)

1. **`self.temporal_filter`** — Block 1, temporal filters:
   - `Conv2d(depth_first_layer, filter_1, kernel_size=c_kernel_1, padding='same', bias=use_bias)`
   - `BatchNorm2d(filter_1)`
2. **`self.spatial_filter`** — Block 1, spatial (depthwise) filters:
   - `Conv2d(filter_1, filter_1*D, kernel_size=c_kernel_2, groups=filter_1, bias=use_bias)` — grouped conv = one spatial filter set per temporal channel; no `padding='same'`, so this conv shrinks the channel axis.
   - `BatchNorm2d(filter_1*D)`
   - `activation`
   - `pooling_layer_1` — `support_functions.get_multichannel_pooling(config.pooling_layer_1_config)` (`nn.AvgPool2d((1, factor))` for the `'default'` type, a channel-wise 1D pooling wrapper otherwise, or `nn.Identity` when disabled)
   - `dropout`
3. **`self.separable_convolution`** — Block 2, separable convolution + feature extraction:
   - depthwise `Conv2d(filter_1*D, filter_1*D, kernel_size=c_kernel_3, groups=filter_1*D, padding='same', bias=use_bias)`
   - pointwise `Conv2d(filter_1*D, filter_2, kernel_size=(1,1), groups=1, bias=use_bias)`
   - `BatchNorm2d(filter_2)`
   - `activation`
   - `pooling_layer_2` — `support_functions.get_multichannel_pooling(config.pooling_layer_2_config)` (same rules as `pooling_layer_1`)
   - `dropout`

> **Design note (pooling):** each pooling stage is chosen by the [`pooling` section](../config/model/EEGNet_cfg.md) of the config. The `'default'` type reproduces the classic stock `nn.AvgPool2d` (pooling along time only); any other type selects one of the [`pooling_1d`](pooling_1d/0_module_quickstart.md) operators applied **channel-wise** via its `*_multichannel` wrapper, resolved by [`get_multichannel_pooling`](support_functions.md). The layout order per block is `BatchNorm -> activation -> pool -> dropout`; this is exactly the "no normalization after pool" scenario the pooling_1d docstring warns about (why the 1D filters default to unit DC gain).

### `forward(x) -> torch.tensor`

Runs `temporal_filter -> spatial_filter -> separable_convolution`, then returns `x.flatten(1)` if `flatten_output` else the 4D tensor.

### `debug_shape(self, input_size)`

Diagnostic (invoked when `print_var=True`, using the config's `input_size`). Creates a random tensor of `input_size` and prints the shape after each named step by indexing into the sequential blocks (e.g. `spatial_filter[0]` conv, `spatial_filter[3]` pool, `separable_convolution[0]`/`[1]`/`[4]`). Purely for interactive inspection; the hard-coded indices assume the block structure above.

---

## `class EEGNet_Classifier(nn.Module)`

`EEGNet` + a linear classification head. (It now has a proper class docstring describing the head and the loss coupling.)

### Construction (`EEGNet_config` or dict)

- `config = get_model_config(config)`.
- Builds `self.eegnet = EEGNet(config)`, then **forces `self.eegnet.flatten_output = True`** — the head needs flat `(B, features)` input, so the wrapped extractor must flatten regardless of the config's `flatten_output` (which only concerns the standalone extractor).
- Computes the flattened feature count with `self.compute_number_of_neurons(config.input_size)` (a dummy forward pass).
- Builds `self.classifier = nn.Sequential(Linear(input_neurons, n_classes), LogSoftmax(dim=1) if use_softmax else Identity())`.

Additional config attributes beyond EEGNet's:

| Attribute | Meaning |
|---|---|
| `input_size` | `(batch, depth, C, T)`, used to size the `Linear` layer via a dummy forward. |
| `n_classes` | Number of output classes. |
| `use_softmax` | `LogSoftmax(dim=1)` head if True, else `Identity` (raw logits). Must match the training loss — see [`cross_check_config`](../training/support_training.md). |

### Methods

- **`forward(x)`** — `x = self.eegnet(x); return self.classifier(x)`.
- **`compute_number_of_neurons(input_size)`** — runs `self.eegnet(torch.rand(input_size))` and returns `len(x.flatten())` (the total element count of the extractor output).
- **`classify(x, return_as_index=True)`** — calls `forward`; if `return_as_index`, returns `argmax(exp(logits), dim=1)` (class indices), else the raw forward output. `exp` is monotonic so it does not change the argmax for either the `LogSoftmax` or `Identity` head. A code comment explicitly warns **not** to `squeeze` first: a batch of size 1 has shape `(1, n_classes)` and squeezing would collapse the batch dim and break `argmax(dim=1)`. `jade.training.metrics` prefers this method over `argmax(model(x))`.

### Gotchas

- **The head needs flat features — now handled automatically.** `compute_number_of_neurons` flattens *all* dims (`len(x.flatten())`), and `forward` feeds the extractor output straight into `Linear`, so the extractor must return a 2D `(B, features)` tensor. `EEGNet_Classifier.__init__` therefore forces `self.eegnet.flatten_output = True`, so the classifier works regardless of the config's `flatten_output` value (this used to be a silent footgun where a `flatten_output=False` config produced a shape mismatch at `forward`). Use a batch dimension of 1 in `input_size` so `input_neurons` equals the per-sample feature count.
