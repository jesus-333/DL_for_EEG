# `config/optimizers/lbfgs.py`

Source: `src/jade/config/optimizers/lbfgs.py`

Defines `lbfgs_config`, mirroring `torch.optim.LBFGS`. Every parameter defaults to torch's value. Built by `support_training.get_optimizer`.

**Caveat (from the source):** L-BFGS is a full-batch optimizer that needs a `closure` re-evaluating the model; the standard mini-batch training loop is not tailored for it, so use it only if you know what you are doing.

## Class `lbfgs_config(optimizer_config)`

`@dataclass`. `name` frozen to `"lbfgs"`.

| Field | Type | Default | torch default? | Meaning |
|---|---|---|---|---|
| `name` | `str` | `"lbfgs"` | — | Discriminator (frozen). |
| `lr` | `float` | `1.0` | yes | Learning rate. Must be `> 0`. |
| `max_iter` | `int` | `20` | yes | Max iterations per optimization step. Must be `> 0`. |
| `max_eval` | `int \| None` | `None` | yes | Max function evaluations per step. `None` → torch uses `max_iter * 1.25`. |
| `tolerance_grad` | `float` | `1e-7` | yes | Termination tolerance on first-order optimality. |
| `tolerance_change` | `float` | `1e-9` | yes | Termination tolerance on function/parameter changes. |
| `history_size` | `int` | `100` | yes | Update history size. Must be `> 0`. |
| `line_search_fn` | `str \| None` | `None` | yes | Either `"strong_wolfe"` or `None`. |

## `__post_init__`

- Freeze `self.name = "lbfgs"`.
- `lr > 0`, `max_iter > 0`, `history_size > 0` else `ValueError`.
- `line_search_fn` must be `"strong_wolfe"` or `None` else `ValueError`.

## Consumption

`get_optimizer` forwards `lr`, `max_iter`, `max_eval`, `tolerance_grad`, `tolerance_change`, `history_size`, `line_search_fn` to `torch.optim.LBFGS`.
