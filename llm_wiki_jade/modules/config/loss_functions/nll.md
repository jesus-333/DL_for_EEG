# `config/loss_functions/nll.py`

Source: `src/jade/config/loss_functions/nll.py`

Defines `nll_config`, mirroring `torch.nn.NLLLoss`. This loss expects **log-probabilities** as input (the model should end with a `LogSoftmax` layer). Use it with a model whose head applies `LogSoftmax` (e.g. `EEGNet_Classifier` with `use_softmax=True`, which is also what its `classify` method assumes). If the model outputs raw logits, use [`cross_entropy_config`](cross_entropy.md) instead. Built by `support_training.get_loss_function`.

## Module constant

| Constant | Value | Meaning |
|---|---|---|
| `REDUCTIONS` | `("none", "mean", "sum")` | Allowed reduction modes (torch). |

## Class `nll_config(loss_config)`

`@dataclass`. `name` frozen to `"nll"`. Same fields as [`cross_entropy_config`](cross_entropy.md) minus `label_smoothing` (NLLLoss has none).

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | `"nll"` | Discriminator (frozen). |
| `weight` | `list[float] \| None` | `None` | Per-class rescaling weights. Stored as a plain list; `get_loss_function` turns it into a tensor. `None` = equal weighting. |
| `ignore_index` | `int` | `-100` | Target value ignored (no gradient contribution). |
| `reduction` | `str` | `"mean"` | One of `REDUCTIONS`. Lower-cased in `__post_init__`. |

## `__post_init__`

- Freeze `self.name = "nll"`.
- If `weight` is not `None`: must be a list/tuple, coerced element-wise to `float` and re-stored as a list; else `ValueError`.
- `reduction` lower-cased and checked against `REDUCTIONS` else `ValueError`.

## Consumption

`get_loss_function` builds `torch.nn.NLLLoss(weight=<tensor or None>, ignore_index=config.ignore_index, reduction=config.reduction)`, converting `weight` via `torch.tensor(config.weight, dtype=torch.float)`.
