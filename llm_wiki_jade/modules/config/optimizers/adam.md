# `config/optimizers/adam.py`

Source: `src/jade/config/optimizers/adam.py`

Defines `adam_config`, mirroring `torch.optim.Adam`. Only commonly-tuned parameters are exposed; each defaults to torch's value, so `adam_config()` ≈ `torch.optim.Adam(model.parameters())`. Built into the real optimizer by `support_training.get_optimizer`.

## Class `adam_config(optimizer_config)`

`@dataclass`. `name` is frozen to `"adam"`.

| Field | Type | Default | torch default? | Meaning |
|---|---|---|---|---|
| `name` | `str` | `"adam"` | — | Discriminator (frozen in `__post_init__`). |
| `lr` | `float` | `1e-3` | yes | Learning rate. Must be `> 0`. |
| `betas` | `tuple[float,float]` | `(0.9, 0.999)` | yes | Running-average coefficients. Each in `[0, 1)`. |
| `eps` | `float` | `1e-8` | yes | Denominator stability term. Must be `> 0`. |
| `weight_decay` | `float` | `0.0` | yes | L2 penalty. Must be `>= 0`. |
| `decoupled_weight_decay` | `bool` | `False` | yes | Use AdamW-style decoupled weight decay. |
| `amsgrad` | `bool` | `False` | yes | Use the AMSGrad variant. |
| `maximize` | `bool` | `False` | yes | Maximize instead of minimize the objective. |

## `__post_init__`

- Freezes `self.name = "adam"`.
- `lr > 0` else `ValueError`.
- `self.betas = tuple(self.betas)` (normalizes a list from a file back to a tuple); must have length 2; each beta in `[0, 1)` else `ValueError`.
- `eps > 0`, `weight_decay >= 0` else `ValueError`.

## Consumption / gotchas

`get_optimizer` forwards all eight fields to `torch.optim.Adam` and wraps `betas` in `tuple(...)` again (defensive, since a JSON/TOML round-trip yields a list). All torch defaults are preserved; nothing is silently changed.
