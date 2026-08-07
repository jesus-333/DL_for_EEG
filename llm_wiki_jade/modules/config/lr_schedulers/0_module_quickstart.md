# `jade.config.lr_schedulers` — Subpackage Quickstart

> READ FIRST. Overview of the learning-rate-scheduler-config family. Field-level detail is on the per-file pages.

## Purpose

A **family** of interchangeable scheduler configs, one per `torch.optim.lr_scheduler` class, selected by a string `name` discriminator. Each config only *describes* a scheduler; `jade.training.support_training.get_lr_scheduler(config, optimizer)` builds the real scheduler, dispatching on `config.name`. In [`training_config`](../training.md) the scheduler is **optional** (`lr_scheduler=None` disables it) and, when present, is **stepped once per epoch**.

## Members

| Config | torch class | Key params (defaults) |
|---|---|---|
| [`exponential_lr_config`](exponential_lr.md) | `ExponentialLR` | `gamma=0.99` |
| [`step_lr_config`](step_lr.md) | `StepLR` | `step_size=10`, `gamma=0.1` |
| [`cosine_annealing_lr_config`](cosine_annealing_lr.md) | `CosineAnnealingLR` | `T_max=10`, `eta_min=0.0` |
| [`cosine_annealing_warm_restarts_config`](cosine_annealing_warm_restarts.md) | `CosineAnnealingWarmRestarts` | `T_0=10`, `T_mult=1`, `eta_min=0.0` |
| [`cyclic_lr_config`](cyclic_lr.md) | `CyclicLR` | `base_lr=1e-4`, `max_lr=1e-2`, `step_size_up=2000`, `mode="triangular"` |

## Registry + factory pattern

Defined in [`__init__.py`](__init__.md), identical in shape to the optimizer subpackage:

- **`lr_scheduler_config`** ([generic.md](generic.md)) — base carrying only `name`; overrides `from_dict` for polymorphic deserialization (lazy registry import to dodge a circular import).
- **`LR_SCHEDULER_CONFIG_REGISTRY`** — `{"exponential_lr", "step_lr", "cosine_annealing_lr", "cosine_annealing_warm_restarts", "cyclic_lr"}` → their classes.
- **`get_lr_scheduler_config(config)`** — returns an already-built config untouched, promotes a `dict` via the base `from_dict`, else `TypeError`.

Each concrete config **freezes** its `name` in `__post_init__`.

## Gotchas

- **Some `name`s differ from the field naming** you might expect: e.g. `T_max`, `T_0`, `T_mult`, `eta_min` keep torch's exact capitalization.
- `cyclic_lr_config.cycle_momentum` defaults to `True` (torch), but only works with optimizers exposing a `momentum` (SGD, RMSprop). Set it to `False` when cycling Adam/AdamW's lr.
- All the step-count fields reject `bool` (subclass of `int`) and require positivity.

Same pattern as [optimizers](../optimizers/0_module_quickstart.md) and [loss_functions](../loss_functions/0_module_quickstart.md).
