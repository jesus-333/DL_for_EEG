# `model/EEGNet.py` — `EEGNet` and `EEGNet_Classifier`

Source: `src/jade/model/EEGNet.py`

PyTorch implementation of EEGNet (Lawhern et al.) as a **feature extractor** (`EEGNet`) plus a thin **classifier wrapper** (`EEGNet_Classifier`). Variable names follow the original paper's nomenclature where possible. Both are built via [`get_model`](__init__.md).

Imports `torch`, `torch.nn`, and `jade.model.support_functions`.

---

## `class EEGNet(nn.Module)`

Classic EEGNet **without** the final feedforward + classification layer of the original paper. It outputs features and can optionally flatten them.

### Tensor shapes

- **Input:** 4D `(B, depth_first_layer, C, T)` — batch, input depth (1 for raw EEG; >1 for time-frequency data), EEG channels `C`, time samples `T`. Convolutions are `nn.Conv2d`, so the input must be 4D.
- **Output:** 4D `(B, filter_2, 1, T')` where `T'` is the time axis after the two `AvgPool2d` stages — **unless** `flatten_output=True`, in which case `x.flatten(1)` gives `(B, filter_2 * T')`.

### Config keys (all read from the single `config: dict`)

| Key | Type | Meaning | Default |
|---|---|---|---|
| `use_bias` | bool | Whether the conv layers use a bias term. | required |
| `D` | int | Spatial filters learned per temporal filter (depth multiplier). Total spatial filters = `filter_1 * D`. | required |
| `activation` | str | Activation name → resolved by `support_functions.get_activation` (`relu`/`elu`/`selu`/`gelu`, case-insensitive). | required |
| `prob_dropout` | float | Dropout probability, in `[0, 1]`. | required |
| `use_dropout_2d` | bool | `nn.Dropout2d` if True else `nn.Dropout` (via `get_dropout`). | required |
| `filter_1` | int | Number of temporal filters (block 1). | required |
| `c_kernel_1` | tuple | Kernel size of the temporal conv. | required |
| `c_kernel_2` | tuple | Kernel size of the spatial (depthwise) conv. | required |
| `p_kernel_1` | tuple or None | First `AvgPool2d` kernel; `None` → `nn.Identity` (no pooling). | required |
| `filter_2` | int | Output filters of the separable conv (block 2). | required |
| `c_kernel_3` | tuple | Kernel size of the separable conv's depthwise part. | required |
| `p_kernel_2` | tuple or None | Second `AvgPool2d` kernel; `None` → `nn.Identity`. | required |
| `C` | int | EEG channel count. **Stored but NOT used** (`self.C`). | required (read) |
| `T` | int | Time samples. **Stored but NOT used** (`self.T`). | required (read) |
| `depth_first_layer` | int | Input depth of the first conv (>1 for time-frequency input). | 1 |
| `flatten_output` | bool | Flatten the output to `(B, features)`. Back-filled into `config` if absent. | False |
| `print_var` | bool | Print parameter counts (and, if `input_size` present, run `debug_shape`) at construction. Back-filled if absent. | False |
| `input_size` | tuple | Only used when `print_var=True`, to drive `debug_shape`. | optional |

> **Note:** `C` and `T` are read into attributes but never used by the architecture (kernel sizes are given explicitly via the `c_kernel_*` / `p_kernel_*` keys). `get_model` passes a copy of the config, so the back-fills of `flatten_output`/`print_var` do not leak to the caller.

### Architecture (three `nn.Sequential` blocks)

1. **`self.temporal_filter`** — Block 1, temporal filters:
   - `Conv2d(depth_first_layer, filter_1, kernel_size=c_kernel_1, padding='same', bias=use_bias)`
   - `BatchNorm2d(filter_1)`
2. **`self.spatial_filter`** — Block 1, spatial (depthwise) filters:
   - `Conv2d(filter_1, filter_1*D, kernel_size=c_kernel_2, groups=filter_1, bias=use_bias)` — grouped conv = one spatial filter set per temporal channel; no `padding='same'`, so this conv shrinks the channel axis.
   - `BatchNorm2d(filter_1*D)`
   - `activation`
   - `AvgPool2d(p_kernel_1)` or `nn.Identity()` if `p_kernel_1 is None`
   - `dropout`
3. **`self.separable_convolution`** — Block 2, separable convolution + feature extraction:
   - depthwise `Conv2d(filter_1*D, filter_1*D, kernel_size=c_kernel_3, groups=filter_1*D, padding='same', bias=use_bias)`
   - pointwise `Conv2d(filter_1*D, filter_2, kernel_size=(1,1), groups=1, bias=use_bias)`
   - `BatchNorm2d(filter_2)`
   - `activation`
   - `AvgPool2d(p_kernel_2)` or `nn.Identity()` if `p_kernel_2 is None`
   - `dropout`

> **Design note (pooling):** the pooling here is stock `nn.AvgPool2d`, **not** the [`pooling_1d`](pooling_1d/0_module_quickstart.md) layers. The layout order per block is `BatchNorm -> activation -> pool -> dropout`; this ordering is exactly the "no normalization after pool" scenario the pooling_1d docstring warns about (why DC gain must be controlled) — relevant if the anti-aliased layers are ever swapped in.

### `forward(x) -> torch.tensor`

Runs `temporal_filter -> spatial_filter -> separable_convolution`, then returns `x.flatten(1)` if `flatten_output` else the 4D tensor.

### `debug_shape(self, input_size)`

Diagnostic (invoked when `print_var=True` and `input_size` is present). Creates a random tensor of `input_size` and prints the shape after each named step by indexing into the sequential blocks (e.g. `spatial_filter[0]` conv, `spatial_filter[3]` pool, `separable_convolution[0]`/`[1]`/`[4]`). Purely for interactive inspection; the hard-coded indices assume the block structure above.

---

## `class EEGNet_Classifier(nn.Module)`

`EEGNet` + a linear classification head. (The class body has a misplaced docstring — the `"""EEGNet + classifier"""` string sits after `super().__init__()` rather than at the top, so it is not the real `__doc__`, only a no-op string; noted here so future edits can fix it.)

### Construction (`config: dict`)

- Builds `self.eegnet = EEGNet(config)`.
- Computes the flattened feature count with `self.compute_number_of_neurons(config['input_size'])` (a dummy forward pass).
- Builds `self.classifier = nn.Sequential(Linear(input_neurons, n_classes), LogSoftmax(dim=1) if use_softmax else Identity())`.

Additional config keys beyond EEGNet's:

| Key | Type | Meaning |
|---|---|---|
| `input_size` | tuple | Shape of a sample batch, used to size the `Linear` layer via a dummy forward. |
| `n_classes` | int | Number of output classes. |
| `use_softmax` | bool | `LogSoftmax(dim=1)` head if True, else `Identity` (raw logits). |

### Methods

- **`forward(x)`** — `x = self.eegnet(x); return self.classifier(x)`.
- **`compute_number_of_neurons(input_size)`** — runs `self.eegnet(torch.rand(input_size))` and returns `len(x.flatten())` (the total element count of the extractor output).
- **`classify(x, return_as_index=True)`** — calls `forward`; if `return_as_index`, returns `argmax(exp(logits), dim=1)` (class indices), else the raw forward output. `exp` is monotonic so it does not change the argmax for either the `LogSoftmax` or `Identity` head. A code comment explicitly warns **not** to `squeeze` first: a batch of size 1 has shape `(1, n_classes)` and squeezing would collapse the batch dim and break `argmax(dim=1)`. `jade.training.metrics` prefers this method over `argmax(model(x))`.

### Gotchas

- **`flatten_output` must be True for the head to line up.** `compute_number_of_neurons` flattens *all* dims (`len(x.flatten())`), so with an `input_size` whose batch dimension is 1, `input_neurons` equals the per-sample feature count. But `forward` feeds the extractor output straight into `Linear`, so the extractor must already return a 2D `(B, features)` tensor — i.e. the config must set `flatten_output=True`. If `flatten_output` is left False, the extractor returns a 4D tensor and `Linear` will be applied along the last (time) axis with a mismatched size. Ensure `flatten_output=True` (and use a batch size of 1 in `input_size`) when using `EEGNet_Classifier`.
