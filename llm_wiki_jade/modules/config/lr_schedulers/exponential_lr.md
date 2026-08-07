# `config/lr_schedulers/exponential_lr.py`

Source: `src/jade/config/lr_schedulers/exponential_lr.py`

Defines `exponential_lr_config`, mirroring `torch.optim.lr_scheduler.ExponentialLR`. At every step the learning rate is multiplied by `gamma` (`lr = lr * gamma`), giving geometric/exponential decay. Built by `support_training.get_lr_scheduler`.

## Class `exponential_lr_config(lr_scheduler_config)`

`@dataclass`. `name` frozen to `"exponential_lr"`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | `"exponential_lr"` | Discriminator (frozen). |
| `gamma` | `float` | `0.99` | Multiplicative decay factor applied every step. Must be in `(0, 1]` (`1` = no decay). |

## `__post_init__`

- Freeze `self.name = "exponential_lr"`.
- `0.0 < gamma <= 1.0` else `ValueError`.

## Consumption

`get_lr_scheduler` calls `ExponentialLR(optimizer, gamma=config.gamma)`. Note torch's `ExponentialLR` has no default `gamma`; this config supplies `0.99`.
