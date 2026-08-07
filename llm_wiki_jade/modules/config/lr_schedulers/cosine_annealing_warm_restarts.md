# `config/lr_schedulers/cosine_annealing_warm_restarts.py`

Source: `src/jade/config/lr_schedulers/cosine_annealing_warm_restarts.py`

Defines `cosine_annealing_warm_restarts_config`, mirroring `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`. The learning rate follows a cosine curve that restarts every `T_0` steps, with each restart period multiplied by `T_mult`. Built by `support_training.get_lr_scheduler`.

## Class `cosine_annealing_warm_restarts_config(lr_scheduler_config)`

`@dataclass`. `name` frozen to `"cosine_annealing_warm_restarts"`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | `"cosine_annealing_warm_restarts"` | Discriminator (frozen). |
| `T_0` | `int` | `10` | Iterations before the first restart. Must be a positive `int`. |
| `T_mult` | `int` | `1` | Factor multiplying the restart period after each restart. Must be an `int >= 1`. |
| `eta_min` | `float` | `0.0` | Minimum learning rate. Must be `>= 0`. |

## `__post_init__`

- Freeze `self.name = "cosine_annealing_warm_restarts"`.
- `T_0` must be a positive `int` (`bool` rejected) else `ValueError`.
- `T_mult` must be an `int >= 1` (`bool` rejected) else `ValueError`.
- `eta_min >= 0` else `ValueError`.

## Consumption

`get_lr_scheduler` calls `CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0, T_mult=config.T_mult, eta_min=config.eta_min)`. Fields keep torch's exact capitalization (`T_0`, `T_mult`).
