# `model/__init__.py` — Model factory (`get_model` / `MODEL_REGISTRY`)

Source: `src/jade/model/__init__.py`

Package init for `jade.model`. Hosts the top-level model factory: a name→class registry and a single `get_model(config)` entry point that builds any model from a config dict. Mirrors the `get_pooling_layer` / `POOLING_REGISTRY` pattern used one level down in [`pooling_1d`](pooling_1d/__init__.md), but for full models.

## Public API (`__all__`)

`EEGNet`, `EEGNet_Classifier`, `MODEL_REGISTRY`, `get_model`.

The 1D pooling layers are **intentionally not** re-exported here: they are building blocks (with their own factory in `jade.model.pooling_1d`), not full models.

## `MODEL_REGISTRY`

A dict mapping lowercase string aliases → model class. Every registered class takes a single `config: dict` argument, so the factory can build all of them uniformly.

| Alias | Class |
|---|---|
| `eegnet` | `EEGNet` |
| `eegnet_classifier` | `EEGNet_Classifier` |
| `eegnetclassifier` | `EEGNet_Classifier` |

## `get_model(config) -> nn.Module`

The single entry point for model creation.

| Parameter | Type | Meaning |
|---|---|---|
| `config` | `dict` | Must contain `model_name` (one of `MODEL_REGISTRY`, case-insensitive) plus every key the chosen model's constructor requires. |

**Returns:** an instance of the requested `torch.nn.Module`.

**Behaviour / internals:**
- `config["model_name"]` is coerced with `str(...).lower()` before lookup, so casing and non-string names are tolerated.
- The registry class is called with `dict(config)` — a **shallow copy** — so a model constructor can freely mutate its config (EEGNet does: it back-fills `flatten_output`, `print_var`) without the caller's dict changing.

**Raises:**

| Exception | Condition |
|---|---|
| `KeyError` | `config` has no `model_name` key. |
| `ValueError` | `model_name` is present but not in `MODEL_REGISTRY`. |

Both messages list the sorted set of valid aliases.

**Examples (from the docstring):**
```python
get_model({'model_name' : 'eegnet', 'use_bias' : False, 'D' : 2})
get_model({'model_name' : 'eegnet_classifier', 'n_classes' : 4})
```
(These are illustrative; a real EEGNet config needs all the keys listed on the [`EEGNet`](EEGNet.md) page.)

## Notes / gotchas

- This is the intended construction path for models used by `jade.training`; training code treats the result as an opaque `nn.Module`.
- Adding a model is a two-line change: import the class and add its alias(es) to `MODEL_REGISTRY`.
- There is currently no `jade.config` sub-package for models, so the `config` dict is assembled by the caller (unlike optimizers/loss/schedulers, which have config builders).
