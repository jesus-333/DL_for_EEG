# `config/optimizers/adamw.py`

Source: `src/jade/config/optimizers/adamw.py`

Defines `adamw_config`, mirroring `torch.optim.AdamW`. Almost identical to [`adam_config`](adam.md); the meaningful differences are the default `weight_decay=1e-2` (torch's AdamW default) and that the weight decay is **always decoupled** (so there is no `decoupled_weight_decay` field). Built by `support_training.get_optimizer`.

## Class `adamw_config(optimizer_config)`

`@dataclass`. `name` frozen to `"adamw"`.

| Field | Type | Default | torch default? | Meaning |
|---|---|---|---|---|
| `name` | `str` | `"adamw"` | — | Discriminator (frozen). |
| `lr` | `float` | `1e-3` | yes | Learning rate. Must be `> 0`. |
| `betas` | `tuple[float,float]` | `(0.9, 0.999)` | yes | Running-average coefficients. Each in `[0, 1)`. |
| `eps` | `float` | `1e-8` | yes | Denominator stability term. Must be `> 0`. |
| `weight_decay` | `float` | `1e-2` | yes | Decoupled weight decay. Must be `>= 0`. |
| `amsgrad` | `bool` | `False` | yes | Use the AMSGrad variant. |
| `maximize` | `bool` | `False` | yes | Maximize instead of minimize. |

## `__post_init__`

Identical validation to Adam: freeze name; `lr > 0`; `betas` normalized to a length-2 tuple with each beta in `[0, 1)`; `eps > 0`; `weight_decay >= 0`. Any violation → `ValueError`.

## Consumption / gotchas

`get_optimizer` forwards `lr`, `betas` (wrapped in `tuple(...)`), `eps`, `weight_decay`, `amsgrad`, `maximize` to `torch.optim.AdamW`. No `decoupled_weight_decay` is passed (AdamW is decoupled by definition).
