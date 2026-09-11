# `model/braindecode_pooling_extension.py` — `braindecode_pooling_wrapper`

Source: `src/jade/model/braindecode_pooling_extension.py`

Wraps a model that already exists in [braindecode](https://braindecode.org/) and **substitutes every pooling layer** with a [`pooling_1d`](pooling_1d/0_module_quickstart.md) operator, so the anti-aliasing pooling ablation (so far only runnable on [`EEGNet`](EEGNet.md)) can be run on any braindecode model (`ShallowFBCSPNet`, `EEGNet`, `Deep4Net`, …). Built via [`get_model`](__init__.md) like any other model (aliases `braindecode` / `braindecode_wrapper` / `braindecode_pooling`); its config is [`braindecode_wrapper_config`](../config/model/braindecode_pooling_extension_cfg.md).

The design idea: **keep the decimation factor, change only the pooling type.** A braindecode `AvgPool2d((1, 4))` becomes the chosen jade operator decimating by the same factor 4; the rest of the model is untouched.

> **braindecode is an optional dependency.** It is imported **lazily** (only inside `_get_braindecode_model_class`), so `import jade.model` never requires it. Install it with `pip install braindecode` (or `pip install jade[braindecode]`).

Imports (top level): `inspect`, `warnings`, `torch`, `torch.nn`, `jade.config.model.get_model_config`, `jade.config.model.braindecode_pooling_extension_cfg.braindecode_wrapper_config`, `jade.model.pooling_1d.get_pooling_layer`.

---

## `class braindecode_pooling_wrapper(nn.Module)`

Builds the braindecode model named by the config, swaps its pooling layers, optionally drops the post-pool nonlinearity, and (for overlapping pooling) re-sizes the trailing classifier. It then behaves like any other jade model — an `nn.Module` with `forward` and `classify` — so `jade.training` consumes it unchanged.

### Construction (`braindecode_wrapper_config` or dict)

`__init__(self, config)`:
1. `config = get_model_config(config)` (a dict with `name = 'braindecode_wrapper'` is promoted; a `TypeError` is raised if it is not a `braindecode_wrapper_config`).
2. Resolve the braindecode class from `config.braindecode_model_name` (`_get_braindecode_model_class`) and build `model = ModelClass(**config.model_kwargs)`. **braindecode validates its own kwargs** — there is no jade dataclass for them.
3. `_replace_pooling_layers(model, config.target_pooling_config, config.pooling_factor_source)` swaps the pooling layers (see below), returning `self.swaps` and whether any swap overlapped.
4. If `config.keep_pool_nonlin` is `False`, `_drop_pool_nonlin(model)` replaces every `pool_nonlin*` module with `nn.Identity` (recorded in `self.dropped_pool_nonlin`).
5. If any replaced pooling **overlapped**, `_resize_trailing_classifier(model, _infer_dummy_input(config, model))` fixes the trailing layer size.
6. `self.model = model`.

### Attributes

| Attribute | Meaning |
|---|---|
| `config` | The validated `braindecode_wrapper_config`. |
| `model` | The wrapped braindecode model (with the swapped pooling layers). |
| `swaps` | One dict `{'name', 'type', 'factor', 'overlapping'}` per replaced pooling layer. |
| `dropped_pool_nonlin` | Qualified names of the `pool_nonlin*` modules replaced by identity (empty when `keep_pool_nonlin=True`). |

### Methods

- **`forward(x)`** — delegates to `self.model(x)`. `x` is in the braindecode layout (typically `(B, n_chans, n_times)`); output is typically `(B, n_outputs)`.
- **`classify(x, return_as_index=True)`** — `argmax(model(x), dim=1)` (class indices) by default, else the raw output. Provided for parity with [`EEGNet_Classifier.classify`](EEGNet.md) so `jade.training.metrics` can use it directly.

---

## How the pooling substitution works

### 1. Decimation factor `M` and the pooled axis (`_resolve_factor_and_axis`)

For each pooling layer the geometry is read **from the original layer**, not hard-coded per model:

- **1D pool** (`nn.AvgPool1d` / `nn.MaxPool1d`): the model tensor is `(B, D, T)`, the pooled axis is the last one.
- **2D pool** (`nn.AvgPool2d` / `nn.MaxPool2d`): the decimated axis is the **kernel dimension that is `> 1`** — a kernel `(M, 1)` pools axis 2 (e.g. ShallowFBCSPNet, internal layout `B × F × T × 1`), a kernel `(1, M)` pools axis 3 (e.g. EEGNet, `B × F × C × T`). A kernel that pools **both** spatial axes is rejected with a `ValueError` (these operators decimate along time only) — this is the "check 1D vs 2D / which axis" the feature spec asks for, answered from the kernel itself.
- **Factor `M`** comes from the layer's **stride** (`pooling_factor_source='stride'`, the default — the true downsampling factor) or its kernel size (`'kernel'`). `overlapping = stride != kernel` on the pooled axis.

A layer that does not decimate (resolved `M <= 1`, e.g. a stride-1 sliding max) is **left unchanged** (a warning is emitted) rather than turned into an identity.

### 2. Building the replacement (`_build_replacement`)

The target [`pooling_config`](../config/model/pooling_1d_cfg.md) is cloned and its `decimation_factor` overridden with this layer's `M` (the caller's config is never mutated). Then:

- a **1D** site takes the 1D operator directly: `get_pooling_layer(config_dict)`;
- a **2D** site wraps that 1D operator in `_pool_along_axis_2d(get_pooling_layer(config_dict), axis)`.

### 3. The 2D axis adapter (`class _pool_along_axis_2d(nn.Module)`)

The jade operators are 1D (`(B, D, T) -> (B, D, T // M)`); braindecode decimates time inside a 4D tensor with a 2D pooling layer. The adapter moves the temporal axis last, flattens the other non-batch axes into the depth axis, runs the 1D layer once, and restores the layout:

```
(B, a, b, c) --movedim(axis,-1)--> (B, …, T) --reshape--> (B, D, T) --pool--> (B, D, T//M) --reshape/movedim--> axis decimated
```

This is the same flatten-pool-restore trick as [`_multichannel_pool_1d`](pooling_1d/support_pooling.md), generalized to pool axis 2 **or** 3. For EEGNet's `(1, M)` (time already last) it degenerates to the plain channel-wise path.

### 4. Module surgery (`_set_submodule`)

Each replacement is spliced in with `parent.get_submodule(parent_path)` + `setattr(parent, attr, new)` (the leaf `attr` can be a `nn.Sequential` numeric index like `'3'`). The tree is collected **before** any mutation (never mutate while walking `named_modules()`).

---

## Post-pool nonlinearity (`keep_pool_nonlin`, `_drop_pool_nonlin`)

Some braindecode models wrap the pooling in the classic log-power pipeline `square -> pool -> log`: `conv_nonlin_exp` (a `Square`) before, `pool_nonlin_exp` (a `SafeLog`) after — this is ShallowFBCSPNet's `pool_nonlin_exp`. Deep4Net's `pool_nonlin*` are already `Identity`.

- `keep_pool_nonlin=True` (default) — only the pooling **type** changes; the `square -> pool -> log` pipeline stays intact (the most faithful ablation). `SafeLog` clamps its input, so it stays numerically safe even though sinc/dwt filters (with negative side-lobes) can make a filtered squared signal slightly negative.
- `keep_pool_nonlin=False` — every module whose attribute name starts with `pool_nonlin` (and is not already `Identity`) is replaced with `nn.Identity` (an extra ablation arm that removes the post-pool log).

---

## Overlapping pooling → trailing-classifier re-size

A non-overlapping pool (EEGNet, Deep4Net) keeps the shape-preserving `T // M` output, so every downstream shape stays valid and nothing else changes. An **overlapping** pool (ShallowFBCSPNet: kernel 75, stride 15) has output `(T - kernel) // stride + 1`, but the jade decimator gives `T // stride`; the length differs, which invalidates the trailing classification layer braindecode sized at construction (e.g. a `conv_classifier` `Conv2d` whose temporal kernel equals the pre-classifier length).

`_resize_trailing_classifier` fixes this best-effort:
1. `_find_final_layer` locates the trailing `Conv2d` / `Linear` — first inside a known braindecode block (`final_layer` / `conv_classifier` / `classifier`), else the last conv/linear in the model.
2. `_probe_input_shape` runs a dummy forward that a pre-hook stops at that layer (via the `_ProbeDone` sentinel exception) to capture the tensor shape now entering it.
3. `_rebuild_final_layer` rebuilds the layer at the new size: a `Conv2d` gets its `> 1` kernel dimension matched to the new axis length (so it collapses time to length 1 again); a `Linear` gets `in_features` = the flattened input size. Same in/out channels, stride, padding, bias.

The dummy input comes from `_infer_dummy_input`: `config.example_input_shape` if set, otherwise `(2, n_chans, n_times)` read from the model's `n_chans` / `n_times` attributes (or `model_kwargs`).

---

## braindecode model resolution (`_get_braindecode_model_class`)

`getattr(braindecode.models, model_name)`, accepting the result **only** if it is an `nn.Module` subclass (the `braindecode.models` namespace also exposes lowercase sub-*modules* and the `EEGModuleMixin` base, which are not buildable models). An unknown name raises a `ValueError` listing the available model classes.

> **braindecode class names** (v1.x): the EEGNet variant is `EEGNet` (not `EEGNetv4`). Confirm names against your installed `braindecode.models`.

---

## Gotchas

- **`dwt` needs a power-of-two decimation factor.** Swapping to `dwt` works only when *every* pooling factor of the model is a power of two (EEGNet: 4, 8 ✓; Deep4Net: 3 ✗; ShallowFBCSPNet: 15 ✗). Other factors raise a clear `ValueError` from `dwt_pool_1d`. `avg` / `blur` / `sinc` / `max` accept any factor.
- **Input layout is braindecode-native.** `forward` delegates unchanged, so feed the wrapped model what braindecode expects (`(B, n_chans, n_times)` for the current models) — not the jade `EEG_Dataset` `(B, 1, C, T)` layout. Bridging the two (for a jade training sweep) is future work.
- **The re-size is best-effort.** For an overlapping model with an unusual trailing layer a warning is emitted and the output shape may be wrong; pass `example_input_shape` if the probe cannot infer it.
- **Verification:** `test/custom/braindecode_pooling.py` builds EEGNet / Deep4Net / ShallowFBCSPNet through the wrapper for several pooling types and checks the swaps, shapes and the `keep_pool_nonlin` flag end-to-end (needs braindecode; no dataset).
