# `config/optimizers/rmsprop.py`

Source: `src/jade/config/optimizers/rmsprop.py`

Defines `rmsprop_config`, mirroring `torch.optim.RMSprop`. Every parameter defaults to torch's value. Built by `support_training.get_optimizer`.

## Class `rmsprop_config(optimizer_config)`

`@dataclass`. `name` frozen to `"rmsprop"`.

| Field | Type | Default | torch default? | Meaning |
|---|---|---|---|---|
| `name` | `str` | `"rmsprop"` | — | Discriminator (frozen). |
| `lr` | `float` | `1e-2` | yes | Learning rate. Must be `> 0`. |
| `alpha` | `float` | `0.99` | yes | Smoothing constant. Must be `>= 0`. |
| `eps` | `float` | `1e-8` | yes | Denominator stability term. Must be `> 0`. |
| `weight_decay` | `float` | `0.0` | yes | L2 penalty. Must be `>= 0`. |
| `momentum` | `float` | `0.0` | yes | Momentum factor. Must be `>= 0`. |
| `centered` | `bool` | `False` | yes | If `True`, compute centered RMSprop (gradient normalized by an estimate of its variance). |

## `__post_init__`

- Freeze `self.name = "rmsprop"`.
- `lr > 0`, `eps > 0` else `ValueError`.
- `alpha >= 0`, `weight_decay >= 0`, `momentum >= 0` else `ValueError`.

## Consumption

`get_optimizer` forwards `lr`, `alpha`, `eps`, `weight_decay`, `momentum`, `centered` to `torch.optim.RMSprop`.
