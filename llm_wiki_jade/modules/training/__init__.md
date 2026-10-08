# `jade.training.__init__`

Source: `src/jade/training/__init__.py`

Package initializer for the `training` subpackage. It performs three jobs: aggregate the public API, attach a `NullHandler` to the package logger, and set the intra-package import order so that leaf modules are loaded before the modules that depend on them.

## Import order (intentional)

The imports are ordered **leaf-first** so that dependent modules find their dependencies already loaded:

```
logging_utils  →  wandb_utils  →  custom_loss_function  →  support_training  →  metrics
               →  classifier_epoch  →  vae_epoch  →  classifier  →  vae  →  test_functions
```

- Every submodule is bound as a module object (`support_training`, `metrics`, `classifier`, `classifier_epoch`, `vae`, `vae_epoch`, `custom_loss_function`, `wandb_utils`), so callers can do `jade.training.vae.train_model(...)` or `jade.training.support_training.get_optimizer(...)`. **This module-qualified form is the preferred style for new code.**
- `logging_utils` is torch-free (stdlib only), so it is listed first; `wandb_utils` has no internal import at all and `custom_loss_function` reaches only `jade.config`, so both are true leaves. `support_training` reaches `custom_loss_function` through a *function-local* import, so there is no ordering constraint between them, but listing leaves first keeps the stated convention honest.

## Package logger

```python
logging.getLogger(__name__).addHandler(logging.NullHandler())
```

Following the standard library-logging best practice, a `logging.NullHandler` is attached to the `jade.training` logger. This means importing the package never emits a "No handlers could be found" warning and never imposes any logging configuration on the user's application. Each module logs through its own `logging.getLogger(__name__)`; the application decides whether/where those records are emitted.

## Public API

`__all__` exports:

| Name | Origin | Kind |
|---|---|---|
| `train` | `classifier` | function — builds the model from a `model_config`, cross-checks, delegates to `train_model` |
| `train_model` | `classifier` | function — full training cycle on an already-built model |
| `wandb_train` | `classifier` | function — wandb-wrapped `train` (builds the model from a `model_config`) |
| `wandb_train_model` | `classifier` | function — wandb-wrapped `train_model` (already-built model; logs optional `model_config`/`dataset_config`) |
| `train_epoch_function` | `classifier_epoch` | function — one training epoch |
| `validation_epoch_function` | `classifier_epoch` | function — one validation epoch |
| `vae_train`, `vae_train_model`, `vae_wandb_train`, `vae_wandb_train_model` | `vae` | the same four entry points for a generative model |
| `vae_train_epoch_function`, `vae_validation_epoch_function` | `vae_epoch` | one epoch each, for a generative model |
| `test` | `test_functions` | function — generic evaluation |
| `setup_logging` | `logging_utils` | function — opt-in root-logger setup (console + optional file) |
| `support_training` | submodule | factories + helpers |
| `metrics` | submodule | classification and reconstruction metrics |
| `classifier`, `classifier_epoch`, `vae`, `vae_epoch`, `custom_loss_function`, `wandb_utils` | submodules | the two training cycles, the custom losses and the wandb plumbing |

### The naming decision

Both families define `train`, `train_model`, `wandb_train`, `wandb_train_model`, `train_epoch_function` and `validation_epoch_function`. The **classification** ones keep the plain names they always had — so every existing `from jade.training import train_model` keeps resolving to the classifier, with no deprecation and no breakage — and the **generative** ones are exported with a `vae_` prefix.

## Notes / gotchas

- The module docstring states the current scope: the **classification** and the **generative** objectives are both implemented, each with its own cycle; a federated-learning submodule is planned but absent.
- `from __future__ import annotations` is used here (and in every module of this package) so all annotations are lazy strings.
