# `jade.training.__init__`

Source: `src/jade/training/__init__.py`

Package initializer for the `training` subpackage. It performs three jobs: aggregate the public API, attach a `NullHandler` to the package logger, and set the intra-package import order so that leaf modules are loaded before the modules that depend on them.

## Import order (intentional)

The imports are ordered **leaf-first** so that dependent modules find their dependencies already loaded:

```
support_training  →  metrics  →  classifier_epoch  →  classifier  →  test_functions
```

- `support_training` and `metrics` are bound as submodule objects (`from jade.training import support_training`, `... metrics`), so callers can do `jade.training.support_training.get_optimizer(...)` and `jade.training.metrics.compute_metrics(...)`.
- `train_epoch_function`, `validation_epoch_function` (from `classifier_epoch`), `train`, `train_model`, `wandb_train` (from `classifier`) and `test` (from `test_functions`) are imported as names into the package namespace.

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
| `wandb_train` | `classifier` | function — wandb-wrapped training |
| `train_epoch_function` | `classifier_epoch` | function — one training epoch |
| `validation_epoch_function` | `classifier_epoch` | function — one validation epoch |
| `test` | `test_functions` | function — generic evaluation |
| `support_training` | submodule | factories + helpers |
| `metrics` | submodule | classification metrics |

## Notes / gotchas

- The module docstring explicitly states the current scope: **only the classification (finetuning) objective** is implemented; a reconstruction/pretraining objective and a federated-learning submodule are planned but absent.
- `from __future__ import annotations` is used here (and in every module of this package) so all annotations are lazy strings.
