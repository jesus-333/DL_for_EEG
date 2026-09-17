# `jade.config.loss_functions` — Subpackage Quickstart

> READ FIRST. Overview of the loss-function-config family. Field-level detail is on the per-file pages.

## Purpose

A **family** of interchangeable loss configs, selected by a string `name` discriminator. Each config only *describes* a loss; `jade.training.support_training.get_loss_function(config)` builds the real module, dispatching on `config.name`. In [`training_config`](../training.md) the default is `cross_entropy_config()`.

Most members mirror a `torch.nn` loss. One, [`elbo_config`](elbo.md), describes a **custom** loss instead — one with no `torch.nn` counterpart, implemented in [`jade.training.custom_loss_function`](../../training/custom_loss_function/0_module_quickstart.md). Its config lives here anyway, because it is the `[loss]` section of a `training_config` and must be in the registry below to be loadable from a TOML; see [its page](elbo.md#why-the-config-lives-here-and-not-next-to-its-implementation).

## Members

| Config | torch class | Input expected | Pair with model |
|---|---|---|---|
| [`cross_entropy_config`](cross_entropy.md) | `torch.nn.CrossEntropyLoss` | raw logits (applies `log_softmax` internally) | `use_softmax = False` |
| [`nll_config`](nll.md) | `torch.nn.NLLLoss` | log-probabilities (model ends with `LogSoftmax`) | `use_softmax = True` |
| [`elbo_config`](elbo.md) | `custom_loss_function.elbo_loss` | the model output tuple `(x_reconstructed, mu, log_var, …)` and the input itself | a **generative** model (`task = 'generative'`), e.g. `vEEGNet` |

### Critical gotcha — loss ↔ model coupling
Choosing the loss that does not match the model's output head silently corrupts training:
- CrossEntropy on already-`LogSoftmax`'d output ⇒ `log_softmax` applied **twice**.
- NLL on raw logits ⇒ `log_softmax` **never** applied.
This is the single most important thing to get right when configuring a run.

## Registry + factory pattern

Defined in [`__init__.py`](__init__.md), identical in shape to the optimizer/scheduler subpackages:

- **`loss_config`** ([generic.md](generic.md)) — base carrying only `name`; overrides `from_dict` for polymorphic deserialization (lazy registry import).
- **`LOSS_CONFIG_REGISTRY`** — `{"cross_entropy": cross_entropy_config, "nll": nll_config, "elbo": elbo_config}`.
- **`get_loss_config(config)`** — returns a built config untouched, promotes a `dict` via the base `from_dict`, else `TypeError`.

Each concrete config **freezes** its `name` in `__post_init__`.

## Shared design

- The two **classification** configs share the module-level `REDUCTIONS = ("none", "mean", "sum")` constant and a `class_weight`/`ignore_index`/`reduction` field set. `elbo_config` has neither `class_weight` nor `ignore_index` (a per-class rescaling is meaningless for a reconstruction objective) and its `REDUCTIONS` excludes `"none"` (a non-reduced ELBO is not a scalar and cannot be backpropagated).
- `class_weight` is stored as a **plain list** (or `None`); `get_loss_function` converts it to a `torch.tensor(..., dtype=torch.float)`. This keeps the config JSON/TOML-serializable (no tensors on disk).

### The `class_weight` name

The field maps to torch's `weight` argument, but it is **not** called `weight` in the config: these are *per-class* coefficients used to counter a class imbalance, and the name `weight` reads as the trainable weights of a model. They are a property of the dataset's label distribution, not of the network.

Because `generic_config.from_dict` silently ignores unknown keys, an old config carrying `weight = [...]` would have trained **without** its class weights and without any warning. `loss_config.from_dict` therefore rejects the legacy key with an explicit message.

Same pattern as [optimizers](../optimizers/0_module_quickstart.md) and [lr_schedulers](../lr_schedulers/0_module_quickstart.md).
