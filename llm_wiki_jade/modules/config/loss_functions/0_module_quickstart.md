# `jade.config.loss_functions` — Subpackage Quickstart

> READ FIRST. Overview of the loss-function-config family. Field-level detail is on the per-file pages.

## Purpose

A **family** of interchangeable classification-loss configs, one per `torch.nn` loss, selected by a string `name` discriminator. Each config only *describes* a loss; `jade.training.support_training.get_loss_function(config)` builds the real `torch.nn.*` module, dispatching on `config.name`. In [`training_config`](../training.md) the default is `cross_entropy_config()`.

## Members

| Config | torch class | Input expected | Pair with model |
|---|---|---|---|
| [`cross_entropy_config`](cross_entropy.md) | `torch.nn.CrossEntropyLoss` | raw logits (applies `log_softmax` internally) | `use_softmax = False` |
| [`nll_config`](nll.md) | `torch.nn.NLLLoss` | log-probabilities (model ends with `LogSoftmax`) | `use_softmax = True` |

### Critical gotcha — loss ↔ model coupling
Choosing the loss that does not match the model's output head silently corrupts training:
- CrossEntropy on already-`LogSoftmax`'d output ⇒ `log_softmax` applied **twice**.
- NLL on raw logits ⇒ `log_softmax` **never** applied.
This is the single most important thing to get right when configuring a run.

## Registry + factory pattern

Defined in [`__init__.py`](__init__.md), identical in shape to the optimizer/scheduler subpackages:

- **`loss_config`** ([generic.md](generic.md)) — base carrying only `name`; overrides `from_dict` for polymorphic deserialization (lazy registry import).
- **`LOSS_CONFIG_REGISTRY`** — `{"cross_entropy": cross_entropy_config, "nll": nll_config}`.
- **`get_loss_config(config)`** — returns a built config untouched, promotes a `dict` via the base `from_dict`, else `TypeError`.

Each concrete config **freezes** its `name` in `__post_init__`.

## Shared design

- Both configs share the module-level `REDUCTIONS = ("none", "mean", "sum")` constant and a `weight`/`ignore_index`/`reduction` field set.
- `weight` is stored as a **plain list** (or `None`); `get_loss_function` converts it to a `torch.tensor(..., dtype=torch.float)`. This keeps the config JSON/TOML-serializable (no tensors on disk).

Same pattern as [optimizers](../optimizers/0_module_quickstart.md) and [lr_schedulers](../lr_schedulers/0_module_quickstart.md).
