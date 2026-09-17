# `jade.training.custom_loss_function` — Subpackage Quickstart

> READ FIRST. Overview of the custom-loss subpackage. Field-level detail is on the per-file pages.

## Purpose

The losses that have **no `torch.nn` counterpart** and must therefore be implemented inside `jade`. Currently a single member, the **ELBO** used to train a variational autoencoder ([`vEEGNet`](../../model/vEEGNet.md)).

## The split between a custom loss and its config

A custom loss lives in two places, and the split is not arbitrary:

| Half | Where | Why |
|---|---|---|
| The **implementation** (`elbo_loss`) | here, in `jade.training` | it needs torch |
| The **config** (`elbo_config`) | [`jade.config.loss_functions`](../../config/loss_functions/elbo.md) | it is the `[loss]` section of a `training_config`, so it must be in `LOSS_CONFIG_REGISTRY` to be loadable from a TOML — and `jade.config` is deliberately torch-free |

Putting the config here instead would invert the package's one-way `training → config` dependency and make the loss unreachable from a config file. See the [config page](../../config/loss_functions/elbo.md#why-the-config-lives-here-and-not-next-to-its-implementation).

## Members

| File | Holds |
|---|---|
| [`support_loss.py`](support_loss.md) | The stateless pieces: the three reconstruction terms + their registry, the closed-form Gaussian KL, the warm-up schedule. Pure functions, testable without a model. |
| [`elbo.py`](elbo.md) | `elbo_loss` (an `nn.Module`) and the `elbo_terms` named tuple. |
| [`__init__.py`](__init__.md) | `CUSTOM_LOSS_REGISTRY` + `get_custom_loss_function`, the usual registry + factory pair. |

## How it is reached

Users never build these directly. [`support_training.get_loss_function`](../support_training.md) dispatches here when the loss name is not a classification one, with a **function-local import** (to keep the module import light and to avoid a cycle, since this subpackage lives under `jade.training`, whose `__init__` imports `support_training`):

```
training_config.loss (elbo_config)
      -> support_training.get_loss_function
      -> custom_loss_function.get_custom_loss_function
      -> elbo_loss
```

## Two design decisions worth knowing

**`elbo_loss` is an `nn.Module`**, not a plain class, for three concrete reasons: the training cycle calls `loss_function.to(device)`; `wandb.watch(criterion = ...)` expects a module; and a future reconstruction term carrying buffers (a learned output variance, a DTW kernel) needs no change.

**It returns every term, not a scalar.** `elbo_terms` is a named tuple `(total, reconstruction, kl, kl_weight, total_full_kl)`. This keeps the loss a *pure function of its inputs*: stashing the terms on the module (`self.last_reconstruction_loss`) would break the moment the loss is called for the training set and then for the validation set before being read. `total` is the field to backpropagate; `total_full_kl` is the same objective at the **final** KL weight, which is the only one comparable across epochs while a warm-up is running — see [`elbo.md`](elbo.md#why-total_full_kl-exists).

## Dependencies

- **External:** `torch` (and `torch.nn.functional` for the reconstruction terms).
- **Internal:** `jade.config.loss_functions.get_loss_config` only. It does **not** import `jade.model` nor the rest of `jade.training`.
