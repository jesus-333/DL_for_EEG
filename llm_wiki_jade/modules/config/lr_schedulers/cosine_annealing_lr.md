# `config/lr_schedulers/cosine_annealing_lr.py`

Source: `src/jade/config/lr_schedulers/cosine_annealing_lr.py`

Defines `cosine_annealing_lr_config`, mirroring `torch.optim.lr_scheduler.CosineAnnealingLR`. The learning rate follows a cosine curve from its initial value down to `eta_min` over `T_max` steps. Built by `support_training.get_lr_scheduler`.

## Class `cosine_annealing_lr_config(lr_scheduler_config)`

`@dataclass`. `name` frozen to `"cosine_annealing_lr"`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | `"cosine_annealing_lr"` | Discriminator (frozen). |
| `T_max` | `int` | `10` | Maximum number of iterations (half a cosine period). Must be a positive `int`. |
| `eta_min` | `float` | `0.0` | Minimum learning rate. Must be `>= 0`. |

## `__post_init__`

- Freeze `self.name = "cosine_annealing_lr"`.
- `T_max` must be a positive `int` (`bool` rejected) else `ValueError`.
- `eta_min >= 0` else `ValueError`.

## Consumption

`get_lr_scheduler` calls `CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)`. Note the field keeps torch's exact capitalization `T_max`.
