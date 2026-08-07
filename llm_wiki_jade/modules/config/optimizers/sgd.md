# `config/optimizers/sgd.py`

Source: `src/jade/config/optimizers/sgd.py`

Defines `sgd_config`, mirroring `torch.optim.SGD`. Every parameter defaults to torch's value. Built by `support_training.get_optimizer`.

## Class `sgd_config(optimizer_config)`

`@dataclass`. `name` frozen to `"sgd"`.

| Field | Type | Default | torch default? | Meaning |
|---|---|---|---|---|
| `name` | `str` | `"sgd"` | — | Discriminator (frozen). |
| `lr` | `float` | `1e-3` | torch requires it (no default); here defaulted to `1e-3` | Learning rate. Must be `> 0`. |
| `momentum` | `float` | `0.0` | yes | Momentum factor. Must be `>= 0`. |
| `dampening` | `float` | `0.0` | yes | Dampening for momentum. |
| `weight_decay` | `float` | `0.0` | yes | L2 penalty. Must be `>= 0`. |
| `nesterov` | `bool` | `False` | yes | Enable Nesterov momentum. Requires positive momentum and zero dampening. |
| `maximize` | `bool` | `False` | yes | Maximize instead of minimize. |

## `__post_init__`

- Freeze `self.name = "sgd"`.
- `lr > 0`, `momentum >= 0`, `weight_decay >= 0` else `ValueError`.
- **Nesterov constraint**: if `nesterov` is `True` and (`momentum <= 0` or `dampening != 0`), raise `ValueError` — this pre-empts the error torch would raise.

## Consumption

`get_optimizer` forwards `lr`, `momentum`, `dampening`, `weight_decay`, `nesterov`, `maximize` to `torch.optim.SGD`.
