# `config/lr_schedulers/cyclic_lr.py`

Source: `src/jade/config/lr_schedulers/cyclic_lr.py`

Defines `cyclic_lr_config`, mirroring `torch.optim.lr_scheduler.CyclicLR`. The learning rate cycles between `base_lr` and `max_lr` following the selected `mode`. Built by `support_training.get_lr_scheduler`.

## Module constant

| Constant | Value | Meaning |
|---|---|---|
| `CYCLIC_MODES` | `("triangular", "triangular2", "exp_range")` | Allowed cycling policies (torch). |

## Class `cyclic_lr_config(lr_scheduler_config)`

`@dataclass`. `name` frozen to `"cyclic_lr"`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | `"cyclic_lr"` | Discriminator (frozen). |
| `base_lr` | `float` | `1e-4` | Lower lr boundary of the cycle. Must be `> 0`. |
| `max_lr` | `float` | `1e-2` | Upper lr boundary. Must be `> base_lr`. |
| `step_size_up` | `int` | `2000` | Iterations in the increasing half of a cycle. Must be a positive `int`. |
| `step_size_down` | `int \| None` | `None` | Iterations in the decreasing half. `None` = equal to `step_size_up`. If set, positive `int`. |
| `mode` | `str` | `"triangular"` | One of `CYCLIC_MODES`. Lower-cased in `__post_init__`. |
| `gamma` | `float` | `1.0` | Constant in the `"exp_range"` scaling function. Must be `> 0`. |
| `cycle_momentum` | `bool` | `True` | Cycle momentum inversely to the lr. Only valid for optimizers exposing a momentum. |

## `__post_init__`

- Freeze `self.name = "cyclic_lr"`.
- `base_lr > 0`; `max_lr > base_lr` else `ValueError`.
- `step_size_up` positive `int` (`bool` rejected); if `step_size_down` is not `None`, it must be a positive `int` too.
- `mode` lower-cased and checked against `CYCLIC_MODES` else `ValueError`.
- `gamma > 0` else `ValueError`.

## Consumption / gotchas

`get_lr_scheduler` forwards `base_lr`, `max_lr`, `step_size_up`, `step_size_down`, `mode`, `gamma`, `cycle_momentum` to `CyclicLR`. **Gotcha:** `cycle_momentum=True` (torch default) only works with optimizers that expose a `momentum` (SGD, RMSprop); set it to `False` when cycling the lr of Adam/AdamW.
