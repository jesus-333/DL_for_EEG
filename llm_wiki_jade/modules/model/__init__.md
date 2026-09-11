# `model/__init__.py` — Model factory (`get_model` / `MODEL_REGISTRY`)

Source: `src/jade/model/__init__.py`

Package init for `jade.model`. Hosts the top-level model factory: a name→class registry and a single `get_model(config)` entry point that builds any model from a **model config** (an instance of a `jade.config.model.model_config` subclass, or a dict promoted to one). Mirrors the `get_pooling_layer` / `POOLING_REGISTRY` pattern used one level down in [`pooling_1d`](pooling_1d/__init__.md), but for full models. Imports `get_model_config` from `jade.config.model` (no import cycle: `config` never imports `model`).

## Public API (`__all__`)

`EEGNet`, `EEGNet_Classifier`, `EEGConformer`, `EEGConformer_Classifier`, `braindecode_pooling_wrapper`, `MODEL_REGISTRY`, `get_model`.

The 1D pooling layers are **intentionally not** re-exported here: they are building blocks (with their own factory in `jade.model.pooling_1d`), not full models.

## `MODEL_REGISTRY`

A dict mapping lowercase string aliases → model class. Every registered class takes a single model-config argument, so the factory can build all of them uniformly.

| Alias | Class |
|---|---|
| `eegnet` | `EEGNet` |
| `eegnet_classifier` | `EEGNet_Classifier` |
| `eegnetclassifier` | `EEGNet_Classifier` |
| `eegconformer` | `EEGConformer` |
| `eegconformer_classifier` | `EEGConformer_Classifier` |
| `eegconformerclassifier` | `EEGConformer_Classifier` |
| `braindecode_wrapper` | `braindecode_pooling_wrapper` |
| `braindecode` | `braindecode_pooling_wrapper` |
| `braindecode_pooling` | `braindecode_pooling_wrapper` |

## `get_model(config) -> nn.Module`

The single entry point for model creation.

| Parameter | Type | Meaning |
|---|---|---|
| `config` | `model_config` or `dict` | A model config instance (e.g. `EEGNet_config`) — or a dict with at least the key `name`, promoted via `get_model_config`. |

**Returns:** an instance of the requested `torch.nn.Module`.

**Behaviour / internals:**
- `config = get_model_config(config)` first (a dict is promoted through the polymorphic `model_config.from_dict`; a built config passes through untouched).
- `config.name` is coerced with `str(...).lower()` before the `MODEL_REGISTRY` lookup, then the class is called with the config instance. The model reads validated attributes off the dataclass — no dict mutation / back-filling anymore (validation and defaulting live in `EEGNet_config`).

**Raises:** `ValueError` if `config.name` is not in `MODEL_REGISTRY` (message lists the sorted valid aliases). A malformed dict raises from `get_model_config` / the config's own validation.

**Examples (from the docstring):**
```python
from jade.config.model import EEGNet_config
get_model(EEGNet_config(name = 'eegnet_classifier', n_classes = 4))
get_model({'name' : 'eegnet', 'use_bias' : False, 'D' : 2})
```

## Notes / gotchas

- This is the intended construction path for models used by `jade.training` (its `train` calls `get_model` internally); training code treats the result as an opaque `nn.Module`.
- Adding a model is a two-line change here: import the class and add its alias(es) to `MODEL_REGISTRY` — plus a matching `{model}_cfg.py` config in [`jade.config.model`](../config/model/0_module_quickstart.md) and an entry in `MODEL_CONFIG_REGISTRY`.
- The discriminator is `config.name` (the old `model_name` dict key is gone).
