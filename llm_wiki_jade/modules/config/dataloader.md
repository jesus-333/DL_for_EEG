# `config/dataloader.py`

Source: `src/jade/config/dataloader.py`

Defines `dataloader_config`, the parameters forwarded to `torch.utils.data.DataLoader` when the training loop wraps a dataset (see `jade.training.support_training.get_dataloader`). **`batch_size` is deliberately NOT here** — it lives in [`training_config`](training.md), since it is a training hyper-parameter shared by every loader. In `training_config` this nested config is **optional**: leaving it `None` (the default) creates a default `dataloader_config` — i.e. these torch-based defaults.

Unlike the other configs, some inconsistent combinations are **gently fixed with a logged warning** rather than raising, mirroring the old code's behavior.

## Module-level

- `logger = logging.getLogger(__name__)` — used for the warnings below.

## Class `dataloader_config(generic_config)`

`@dataclass`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `num_workers` | `int` | `0` | Data-loading subprocesses; `0` = load in the main process. |
| `prefetch_factor` | `int \| None` | `None` | Batches prefetched per worker. Must be `None` when `num_workers=0` (torch requirement); set to `2` when workers exist and it was left `None`. |
| `pin_memory` | `bool` | `False` | Copy tensors into pinned (page-locked) memory before returning. |
| `persistent_workers` | `bool` | `False` | Keep worker processes alive between epochs. Only valid when `num_workers>0`. |
| `shuffle` | `bool` | `True` | Reshuffle the data every epoch. |

## `__post_init__` — validation & auto-fix

Three blocks:

1. **`num_workers`**: must be a non-negative `int` (`bool` rejected). Otherwise `ValueError`.
2. **`prefetch_factor`** (depends on `num_workers`):
   - `num_workers == 0`: if `prefetch_factor` is not `None`, **log a warning and force it to `None`** (torch forbids a prefetch factor without workers).
   - `num_workers > 0`: if `prefetch_factor` is `None`, set it to `2` (torch's usual default); if `<= 0`, raise `ValueError`.
3. **`persistent_workers`**: if `True` while `num_workers == 0`, **log a warning and force it to `False`**.

## Gotchas

- The warn-and-fix policy (blocks 2 & 3) is intentional and different from the raise-on-bad-value policy used elsewhere in the module. Inconsistent worker/prefetch/persistence combinations are silently corrected, not rejected.
- `batch_size` absence is by design; `get_dataloader(dataset, batch_size, dataloader_config_to_use)` receives it separately from `training_config.batch_size`.
