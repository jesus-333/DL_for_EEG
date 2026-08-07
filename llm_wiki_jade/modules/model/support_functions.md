# `model/support_functions.py` — model-creation helpers

Source: `src/jade/model/support_functions.py`

Small stateless helper functions used while building models. Two groups: generic construction helpers (used by [`EEGNet`](EEGNet.md)) and attention helpers (used by [`aEEGNet`](aEEGNet.md), still WIP). Imports `torch`, `torch.nn`.

---

## Construction helpers

### `get_activation(activation_name: str) -> nn.Module`

Maps a string to a fresh PyTorch activation module. Case-insensitive.

| `activation_name` (any case) | Returns |
|---|---|
| `relu` | `nn.ReLU()` |
| `elu` | `nn.ELU()` |
| `selu` | `nn.SELU()` |
| `gelu` | `nn.GELU()` |

**Raises `ValueError`** for any other string (message lists the four supported names).

### `get_dropout(prob_dropout: float, use_droput_2d: bool) -> nn.Module`

Returns `nn.Dropout2d(prob_dropout)` if `use_droput_2d` else `nn.Dropout(prob_dropout)`.

| Param | Type | Meaning |
|---|---|---|
| `prob_dropout` | float | Dropout probability in `[0, 1]`. |
| `use_droput_2d` | bool | Select 2D (channel-wise) vs plain dropout. (Note the parameter name is spelled `use_droput_2d` in the source.) |

### `count_trainable_parameters(torch_module: nn.Module) -> int`

Returns `sum(p.numel() for p in torch_module.parameters() if p.requires_grad)` — the count of trainable parameters. Used by `EEGNet` when `print_var=True` to report per-block parameter counts.

---

## Attention helpers (for `aEEGNet`)

### `create_head(config: dict) -> tuple[nn.Module, nn.Module, nn.Module]`

Builds the query/key/value projection layers for an attention head. Returns `(q_map, k_map, v_map)`.

| Config key | Type | Meaning |
|---|---|---|
| `input_length` | int | Input feature length (in-features of the Linear layers). |
| `qk_output_length` | int | Out-features for the query and key projections. |
| `use_head_for_v` | bool | If True, `v_map` is a `Linear`; otherwise `v_map = nn.Identity()`. |
| `v_output_length` | int | Out-features for the value projection (used only if `use_head_for_v`). |

- `q_map = nn.Linear(input_length, qk_output_length)`
- `k_map = nn.Linear(input_length, qk_output_length)`
- `v_map = nn.Linear(input_length, v_output_length)` if `use_head_for_v` else `nn.Identity()`

### `compute_attention_values(q, k, v, normalize_qk) -> torch.Tensor`

Functional scaled-dot-product attention. Parallel to (not shared with) the method inside `channels_attention_head`.

| Param | Type | Documented shape | Meaning |
|---|---|---|---|
| `q` | `torch.Tensor` | `(B, C, Dq)` | Query array. |
| `k` | `torch.Tensor` | `(B, Dk, C)` | Key array (note the axis order — it is permuted internally). |
| `v` | `torch.Tensor` | `(B, C, Dv)` | Value array. |
| `normalize_qk` | bool | — | If True, divide the score by `sqrt(Dq)` before softmax. |

**Returns** `z` of shape `(B, C, Dv)`.

**Internals:**
1. `k = torch.permute(k, (0, 2, 1))` — transpose the last two axes.
2. `score = torch.bmm(q, k)`.
3. If `normalize_qk`: `score /= torch.sqrt(torch.tensor(q.shape[2]))` (i.e. `sqrt(Dq)`).
4. `score = softmax(score, dim=2)`.
5. `z = torch.bmm(score, v)`.
