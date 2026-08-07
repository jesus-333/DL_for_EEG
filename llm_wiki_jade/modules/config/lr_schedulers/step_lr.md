# `config/lr_schedulers/step_lr.py`

Source: `src/jade/config/lr_schedulers/step_lr.py`

Defines `step_lr_config`, mirroring `torch.optim.lr_scheduler.StepLR`. Every `step_size` steps the learning rate is multiplied by `gamma`. Built by `support_training.get_lr_scheduler`.

## Class `step_lr_config(lr_scheduler_config)`

`@dataclass`. `name` frozen to `"step_lr"`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | `"step_lr"` | Discriminator (frozen). |
| `step_size` | `int` | `10` | Period of decay, in scheduler steps. Must be a positive `int`. |
| `gamma` | `float` | `0.1` | Multiplicative decay factor. Must be in `(0, 1]`. |

## `__post_init__`

- Freeze `self.name = "step_lr"`.
- `step_size` must be a positive `int` (`bool` rejected) else `ValueError`.
- `0.0 < gamma <= 1.0` else `ValueError`.

## Consumption

`get_lr_scheduler` calls `StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)`.
