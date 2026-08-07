# `config/optimizers/generic.py`

Source: `src/jade/config/optimizers/generic.py`

Defines `optimizer_config`, the base class for every optimizer config. Inherits from [`generic_config`](../generic.md), adds the single `name` discriminator, and overrides `from_dict` to enable **polymorphic deserialization** of nested optimizer configs.

## Class `optimizer_config(generic_config)`

`@dataclass`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | `""` | Discriminator naming the concrete optimizer (`"adam"`, `"sgd"`, …). Concrete subclasses set and freeze it in their `__post_init__`, so it normally is not passed by hand. |

### `from_dict(data: dict) -> optimizer_config` (classmethod, overridden)

Two behaviors depending on which class it is called on:

- **Called on the base `optimizer_config`** (as happens when [`training_config`](../training.md) rebuilds its `optimizer` field from a dict): reads `name = str(data.get("name","")).lower()`, checks it against `OPTIMIZER_CONFIG_REGISTRY` (unknown → `ValueError` listing the known names), and delegates to `REGISTRY[name].from_dict(data)`.
- **Called on a concrete subclass** (`cls is not optimizer_config`): falls back to `super().from_dict(data)` — the standard [`generic_config.from_dict`](../generic.md).

Key implementation detail: the registry is imported **inside the method** (`from jade.config.optimizers import OPTIMIZER_CONFIG_REGISTRY`), not at module top, to avoid a **circular import** (the package `__init__` imports this module while building the registry).

## Why this design

It lets the recursive (de)serialization of `generic_config` handle a *polymorphic* nested field for free: when a parent config recurses into its `optimizer` field via `nested_cls.from_dict`, `nested_cls` is `optimizer_config`, and this override picks the right concrete subclass from `name`. Same pattern in [lr_schedulers/generic](../lr_schedulers/generic.md) and [loss_functions/generic](../loss_functions/generic.md).
