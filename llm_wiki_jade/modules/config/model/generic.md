# `config/model/generic.py`

Source: `src/jade/config/model/generic.py`

Defines `model_config`, the base class every model configuration inherits from. It carries only the `name` discriminator and the polymorphic `from_dict`; concrete fields and validation live on the subclasses (e.g. [`EEGNet_config`](EEGNet_cfg.md)). Same pattern as [`optimizer_config`](../optimizers/generic.md).

## Class `model_config(generic_config)`

`@dataclass`. Single field:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | `""` | Discriminator with the concrete model name. Also the key `jade.model.get_model` uses to pick the model class from `MODEL_REGISTRY`. Concrete subclasses validate/normalize it in `__post_init__`. |

## `from_dict(data)` (classmethod, polymorphic)

- When called **on the base class** `model_config`: reads `data["name"].lower()`, looks it up in `MODEL_CONFIG_REGISTRY` (unknown → `ValueError`), and delegates to the concrete subclass's `from_dict`.
- When called **on a concrete subclass**: falls back to the standard `generic_config.from_dict`.

The registry is imported **lazily inside the method** (`from jade.config.model import MODEL_CONFIG_REGISTRY`) to avoid a circular import with the package `__init__`.

This makes model configs participate in the recursive (de)serialization of `generic_config` just like the optimizer/scheduler/loss families.
