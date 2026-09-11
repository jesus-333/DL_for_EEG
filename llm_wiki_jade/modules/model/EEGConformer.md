# `model/EEGConformer.py` — `EEGConformer` and `EEGConformer_Classifier`

Source: `src/jade/model/EEGConformer.py`

PyTorch implementation of the **EEG Conformer** (Song et al., *EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization*, IEEE TNSRE 2023, [paper](https://ieeexplore.ieee.org/document/9991178)) as a **feature extractor** (`EEGConformer`) plus a **classifier wrapper** (`EEGConformer_Classifier`). The model couples a convolutional patch embedding (a shallow EEGNet-like front end) with a transformer encoder and, for the classifier, a flatten + MLP head. Variable names follow the authors' reference implementation where possible. Both are built via [`get_model`](__init__.md).

This is a jade-native re-implementation (mirroring the [`EEGNet`](EEGNet.md) structure), rewritten so that:
- it is driven by a validated [`EEGConformer_config`](../config/model/EEGConformer_cfg.md) (a dict is accepted and promoted via `get_model_config`): the first line of each `__init__` is `config = get_model_config(config)`, and the class then reads the config **attributes** — all defaulting/validation lives in the config;
- the pooling stage of the patch embedding supports the custom [`pooling_1d`](pooling_1d/0_module_quickstart.md) operators, so the anti-aliasing ablation runs on the Conformer too (see [`get_conformer_pooling`](#get_conformer_poolingconfig---nnmodule));
- it depends only on `torch` — the original `einops` `Rearrange`/`rearrange` calls are expressed with plain tensor operations (`flatten` / `transpose` / `view` / `permute`), so `einops` is **not** a dependency.

Imports `torch`, `torch.nn`, `jade.model.support_functions`, and `jade.config.model.get_model_config`.

---

## `get_conformer_pooling(config) -> nn.Module`

Module-level factory that builds the single pooling layer of the patch embedding from the config `[pooling]` section (a [`conformer_pooling_config`](../config/model/EEGConformer_cfg.md), or a dict promoted to one). It is the bridge between the Conformer config and the 1D pooling family, analogous to what [`get_multichannel_pooling`](support_functions.md) does for EEGNet — and it delegates to `get_multichannel_pooling` for the 1D operators.

| `pooling_type` | Built layer |
|---|---|
| `'default'` | The original EEG Conformer pooling: an **overlapping** `nn.AvgPool2d((1, pooling_kernel), (1, pooling_stride))` (average pooling along the time axis only). |
| `'avg'` / `'max'` / `'blur'` / `'sinc'` / `'dwt'` (and aliases) | The matching `*_multichannel` [`pooling_1d`](pooling_1d/0_module_quickstart.md) wrapper, applied channel-wise, with decimation factor `pooling_stride` — resolved through `support_functions.get_multichannel_pooling(config.pooling_layer_config)`. |
| `'identity'` / `'none'` | `nn.Identity` (pooling disabled). |

Because the pooling comes right after the spatial convolution has collapsed the EEG-channel axis, it always runs on a `(B, filter_temporal, 1, T')` tensor (so the channel-wise 1D wrappers see a single "EEG channel" and pool along time). The **decimation factor directly sets the number of transformer tokens**.

> **`'default'` vs the 1D operators — a genuine difference.** The paper's default pooling is *overlapping* (kernel 75, stride 15), which does not fit the `pooling_1d` "output length = `T // M`" contract. So `'default'` is kept as a faithful overlapping `AvgPool2d`, while a 1D operator decimates by `pooling_stride` (non-overlapping). The two therefore produce a slightly different token count — but the classifier head is sized with a dummy forward, so any token count works. For a 22×1000 trial the default yields **61 tokens** (61·40 = 2440, matching the original hard-coded flatten size).

---

## `class EEGConformer(nn.Module)`

The EEG Conformer **without** the final flatten + MLP classification head of the paper. It outputs the transformer token sequence and can optionally flatten it.

### Tensor shapes

- **Input:** 4D `(B, depth_first_layer, C, T)` — batch, input depth (1 for raw EEG; >1 for time-frequency data), EEG channels `C`, time samples `T`. Convolutions are `nn.Conv2d`, so the input must be 4D (same contract as `EEGNet`).
- **Output:** 3D `(B, n_tokens, emb_size)` — the transformer token sequence — **unless** `flatten_output=True`, in which case `x.flatten(1)` gives `(B, n_tokens * emb_size)`.

### Config attributes (read from the `EEGConformer_config`)

The exhaustive field list, types, defaults and validation are on the [`EEGConformer_config` page](../config/model/EEGConformer_cfg.md). What `EEGConformer` reads: the patch-embedding fields (`depth_first_layer`, `filter_temporal`, `c_kernel_temporal`, `C`, `emb_size`, `activation`, `prob_dropout`, `pooling`), the transformer fields (`depth`, `num_heads`, `transformer_dropout`, `forward_expansion`, `forward_dropout`), plus `flatten_output`, `print_var` and `input_size` (for `debug_shape`).

### Architecture

Two sub-modules, both defined as helper classes in the same file:

1. **`self.patch_embedding`** — `patch_embedding(config)`:
   - `shallow_net` = `Conv2d(depth_first_layer, filter_temporal, c_kernel_temporal, (1,1))` (temporal conv) → `Conv2d(filter_temporal, filter_temporal, (C,1), (1,1))` (spatial conv, collapses the EEG-channel axis to 1) → `BatchNorm2d(filter_temporal)` → activation → **pooling** (via `get_conformer_pooling`) → `Dropout(prob_dropout)`.
   - `projection` = `Conv2d(filter_temporal, emb_size, (1,1))` (a linear map to the token embedding size).
   - `forward`: after the convs the tensor is `(B, emb_size, 1, T_pool)`; it is turned into the token sequence `(B, T_pool, emb_size)` with `x.flatten(2).transpose(1, 2)` (the plain-tensor equivalent of einops' `Rearrange('b e h w -> b (h w) e')`, height is 1).
2. **`self.transformer_encoder`** — `transformer_encoder(config)`: a `nn.Sequential` of `config.depth` identical `transformer_encoder_block`s, each = `residual_add(LayerNorm → multi_head_attention → Dropout)` then `residual_add(LayerNorm → feed_forward_block → Dropout)` (pre-norm). Keeps the `(B, n_tokens, emb_size)` shape.

### `forward(x) -> torch.tensor`

Runs `patch_embedding -> transformer_encoder`, then returns `x.flatten(1)` if `flatten_output` else the 3D token tensor.

### `debug_shape(self, input_size)`

Diagnostic (invoked when `print_var=True`, using the config's `input_size`). Creates a random tensor and prints the shape after each step (`shallow_net`, `projection`, the token reshape, the transformer encoder, and the optional flatten). Purely for interactive inspection.

### Building-block helper classes (same file, `snake_case` like `pooling_1d`)

| Class | Role |
|---|---|
| `patch_embedding` | Convolutional front end → token sequence (described above). |
| `multi_head_attention` | Multi-head self-attention. Separate `queries`/`keys`/`values` linear maps, `einsum`-based scaled dot-product attention, `attention_dropout`, output `projection`. **Faithful quirk:** the scaling divides by `sqrt(emb_size)` (not the more common `sqrt(head_dim)`), matching the reference implementation. `emb_size` must be divisible by `num_heads` (checked by the config). |
| `residual_add` | Skip connection `x + fn(x)` (out-of-place). |
| `feed_forward_block` | `Linear(emb_size, expansion*emb_size) → GELU → Dropout → Linear(expansion*emb_size, emb_size)`. |
| `transformer_encoder_block` | One pre-norm block (attention residual + feedforward residual). |
| `transformer_encoder` | Stack of `depth` blocks. |
| `classification_head` | The MLP head used by the classifier (see below). |

---

## `class EEGConformer_Classifier(nn.Module)`

`EEGConformer` + the flatten + MLP classification head of the paper.

### Construction (`EEGConformer_config` or dict)

- `config = get_model_config(config)`.
- Builds `self.eegconformer = EEGConformer(config)`, then **forces `self.eegconformer.flatten_output = True`** — the head needs flat `(B, features)` input, so the wrapped extractor must flatten regardless of the config's `flatten_output` (same footgun-avoidance as `EEGNet_Classifier`).
- Computes the flattened feature count (`n_tokens * emb_size`) with `self.compute_number_of_neurons(config.input_size)` (a dummy forward pass).
- Builds `self.classifier = classification_head(config, input_neurons)`.

### The `classification_head`

An MLP mirroring the reference `fc` head, with configurable sizes/dropouts and the model activation (default ELU):

`Linear(input_neurons, hidden_size_1) → activation → Dropout(head_dropout_1) → Linear(hidden_size_1, hidden_size_2) → activation → Dropout(head_dropout_2) → Linear(hidden_size_2, n_classes) → (LogSoftmax(dim=1) if use_softmax else Identity)`.

Defaults reproduce the original: `hidden_size_1=256`, `hidden_size_2=32`, `head_dropout_1=0.5`, `head_dropout_2=0.3`. The `use_softmax` last layer must match the training loss — see [`cross_check_config`](../training/support_training.md).

### Methods

- **`forward(x)`** — `x = self.eegconformer(x); return self.classifier(x)`.
- **`compute_number_of_neurons(input_size)`** — runs `self.eegconformer(torch.rand(input_size))` and returns `len(x.flatten())` (use a batch dim of 1 in `input_size` so this equals the per-sample feature count).
- **`classify(x, return_as_index=True)`** — calls `forward`; if `return_as_index`, returns `argmax(exp(logits), dim=1)` (class indices), else the raw forward output. Same "do **not** `squeeze` first" comment as `EEGNet_Classifier` (a batch of size 1 has shape `(1, n_classes)`, and squeezing would break `argmax(dim=1)`). `jade.training.metrics` prefers this method over `argmax(model(x))`.

### Gotchas

- **`einops`-free.** The original code uses `einops`; this port uses plain tensor ops, so no new dependency is added to `jade`.
- **The MLP head is sized dynamically**, so a change of `C` / `T` / pooling operator / `pooling_stride` (all of which change `n_tokens`) needs no manual edit — unlike the reference implementation's hard-coded `2440`.
- **`sinc` / `dwt` pooling need optional deps** (`scipy` for `sinc`, sometimes `PyWavelets` for `dwt`), imported lazily inside `pooling_1d`; `avg` / `max` / `blur` / `default` need only `torch`/`numpy`.
- **Do not confuse with braindecode's `EEGConformer`.** braindecode also ships a model called `EEGConformer`; the [`braindecode_pooling_wrapper`](braindecode_pooling_extension.md) cannot swap its (overlapping-pooling → transformer) pooling automatically (its build-time probe raises a clear error, checked in `test/custom/braindecode_pooling.py`). This jade-native `EEGConformer` is the one to use for the pooling ablation on a Conformer.
