# `config/loss_functions/cross_entropy.py`

Source: `src/jade/config/loss_functions/cross_entropy.py`

Defines `cross_entropy_config`, mirroring `torch.nn.CrossEntropyLoss`. This loss expects **raw, un-normalized scores (logits)** — it applies `log_softmax` internally. Use it with a model whose head does NOT apply a softmax (e.g. `EEGNet_Classifier` built with `use_softmax=False`). If the model already outputs log-probabilities, use [`nll_config`](nll.md) instead to avoid applying `log_softmax` twice. Built by `support_training.get_loss_function`.

## Module constant

| Constant | Value | Meaning |
|---|---|---|
| `REDUCTIONS` | `("none", "mean", "sum")` | Allowed reduction modes (torch). |

## Class `cross_entropy_config(loss_config)`

`@dataclass`. `name` frozen to `"cross_entropy"`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | `"cross_entropy"` | Discriminator (frozen). |
| `weight` | `list[float] \| None` | `None` | Per-class rescaling weights (length `n_classes`). Stored as a plain list; `get_loss_function` turns it into a tensor. `None` = equal weighting. |
| `ignore_index` | `int` | `-100` | Target value ignored (no gradient contribution). |
| `reduction` | `str` | `"mean"` | One of `REDUCTIONS`. Lower-cased in `__post_init__`. |
| `label_smoothing` | `float` | `0.0` | Label smoothing in `[0, 1)`. |

## `__post_init__`

- Freeze `self.name = "cross_entropy"`.
- If `weight` is not `None`: must be a list/tuple, coerced element-wise to `float` and re-stored as a list; else `ValueError`.
- `reduction` lower-cased and checked against `REDUCTIONS` else `ValueError`.
- `0.0 <= label_smoothing < 1.0` else `ValueError`.

## Consumption

`get_loss_function` builds `torch.nn.CrossEntropyLoss(weight=<tensor or None>, ignore_index=config.ignore_index, reduction=config.reduction, label_smoothing=config.label_smoothing)`, converting `weight` via `torch.tensor(config.weight, dtype=torch.float)`.
