# `model/aEEGNet.py` — attention-augmented EEGNet (WORK IN PROGRESS)

Source: `src/jade/model/aEEGNet.py`

Despite the `a` prefix, this file is **not** the anti-aliased EEGNet — its own module docstring reads *"Variant of EEGNet enhanced with attention mechanism."* The `a` stands for **attention**. (The anti-aliasing research lives entirely in [`pooling_1d`](pooling_1d/0_module_quickstart.md), which nothing here imports.) The file is an early scaffold: `aEEGNet` itself is a no-op stub, and the substantive but incomplete piece is `channels_attention_head`.

Imports `torch`, `jade.model.EEGNet` (the module), and `jade.model.support_functions`.

> **Status: not usable as-is.** `aEEGNet` does nothing, and `channels_attention_head` has a class-definition bug (see below). Document reflects the current source; treat as a placeholder for planned work.

---

## `class aEEGNet(torch.nn.Module)`

Stub. Its docstring has an empty `Attributes` section and no parameter documentation.

- **`__init__(self, config: dict)`** — calls `super().__init__()` then `a = 2` (a local no-op; nothing is stored, no layers are built).
- **`forward(self, x)`** — returns `x` unchanged (identity).

So the network currently has no parameters and no effect. It is **not** in `MODEL_REGISTRY`, so `get_model` cannot build it yet.

---

## `class channels_attention_head(torch.nn.module)`

Intended as an attention head specialized for multichannel EEG: it takes a multichannel EEG tensor and outputs a weighted sum of the input channels, with weights produced by a query/key/value attention mechanism.

> **BUG (live):** the class subclasses `torch.nn.module` (lowercase `module`), which is not the base class `torch.nn.Module`. As written this will fail (module object is not a valid base / has no `Module` semantics). Must be `torch.nn.Module`. Flagged as a genuine defect, not intended behaviour.

### `__init__(self, config: dict)`

- `self.q_map, self.k_map, self.v_map = support_functions.create_head(config)` — builds the Linear (or Identity for value) projection layers. See [`create_head`](support_functions.md) for the config keys it needs (`input_length`, `qk_output_length`, `use_head_for_v`, `v_output_length`).
- `self.normalize_qk = config.get('normalize_qk', True)` — whether to scale the QK score by `1/sqrt(Dq)` before softmax.

### `forward(self, x)`

Input `x`: `(B, C, T)` (batch, channels, time). Steps:
1. `attention_weights = self.compute_attention_weights(x)`
2. `v = self.v_map(x)`
3. `output = torch.bmm(attention_weights, v)`

Returns `output` (docstring `Returns` section is empty in source).

### `compute_attention_weights(self, x) -> torch.Tensor`

| Step | Operation |
|---|---|
| Query/Key | `q = self.q_map(x)`, `k = self.k_map(x)` |
| Score | `attention_weights = torch.bmm(q, k.transpose(1, 2))` |
| Normalize | if `normalize_qk`: divide by `q.shape[-1] ** 0.5` |
| Softmax | `torch.softmax(..., dim=-1)` |

Input `x`: `(B, C, T)`; documented return shape is `(B, C)` in the docstring, though the actual `bmm` of `q (B, C, Dq)` with `kᵀ (B, Dq, C)` yields `(B, C, C)` — a per-channel attention matrix. (Docstring shape and code disagree; the code is the authority.)

### Relationship to `support_functions`

This head consumes the projection layers from `create_head`. A standalone functional variant, `compute_attention_values(q, k, v, normalize_qk)`, also exists in [`support_functions`](support_functions.md) and implements the same math with explicit q/k/v arguments (note it transposes `k` internally and expects slightly different shapes) — the two are parallel implementations, not shared code.
