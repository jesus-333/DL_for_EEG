# `jade.config.optimizers` — Subpackage Quickstart

> READ FIRST. Overview of the optimizer-config family. Field-level detail is on the per-file pages.

## Purpose

A **family** of interchangeable optimizer configs, one per `torch.optim` optimizer, selected by a string `name` discriminator. Each config only *describes* an optimizer; `jade.training.support_training.get_optimizer(config, model)` turns it into the real `torch.optim.*` instance, dispatching on `config.name`.

## Members

| Config | torch class | Notable default(s) |
|---|---|---|
| [`adam_config`](adam.md) | `torch.optim.Adam` | `lr=1e-3`; exposes `decoupled_weight_decay` |
| [`adamw_config`](adamw.md) | `torch.optim.AdamW` | `lr=1e-3`, `weight_decay=1e-2` (decoupled always) |
| [`sgd_config`](sgd.md) | `torch.optim.SGD` | `lr=1e-3`, `momentum=0` |
| [`rmsprop_config`](rmsprop.md) | `torch.optim.RMSprop` | `lr=1e-2`, `alpha=0.99` |
| [`lbfgs_config`](lbfgs.md) | `torch.optim.LBFGS` | `lr=1.0`; full-batch, needs a closure |

Every exposed field defaults to the value torch uses, so a bare `adam_config()` ≈ `torch.optim.Adam(model.parameters())`. Only commonly-tuned parameters are exposed.

## The registry + factory pattern

Defined in [`__init__.py`](__init__.md):

- **`optimizer_config`** ([generic.md](generic.md)) — base class carrying only `name`; overrides `from_dict` for **polymorphic** deserialization.
- **`OPTIMIZER_CONFIG_REGISTRY`** — `{"adam": adam_config, "adamw": adamw_config, "sgd": sgd_config, "rmsprop": rmsprop_config, "lbfgs": lbfgs_config}`.
- **`get_optimizer_config(config)`** — returns a config untouched if already an `optimizer_config`, promotes a `dict` (keyed on `name`) via `optimizer_config.from_dict`, else raises `TypeError`.

### How polymorphic load works
`optimizer_config.from_dict(data)`, when called **on the base class**, reads `data["name"].lower()`, looks it up in the registry (unknown → `ValueError`), and delegates to the concrete subclass's `from_dict`. Called on a concrete subclass it just runs the standard `generic_config.from_dict`. The registry is imported **lazily inside the method** to avoid a circular import with `__init__`.

Each concrete config **freezes** its `name` in `__post_init__` (e.g. `self.name = "adam"`), so it always round-trips as the right optimizer.

## Design notes

- `betas` (Adam/AdamW) are stored as a tuple but normalized from a list in `__post_init__`, because a TOML/JSON round-trip turns tuples into lists; `get_optimizer` also wraps them in `tuple(...)`.
- Validation is strict (`ValueError` on bad values) and mostly checks positivity/ranges that torch itself would reject.
- Mirrors the `MODEL_REGISTRY`/`get_model` convention elsewhere in `jade`, and the identical patterns in [lr_schedulers](../lr_schedulers/0_module_quickstart.md) and [loss_functions](../loss_functions/0_module_quickstart.md).
