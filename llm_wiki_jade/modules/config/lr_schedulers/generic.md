# `config/lr_schedulers/generic.py`

Source: `src/jade/config/lr_schedulers/generic.py`

Defines `lr_scheduler_config`, the base class for every scheduler config. Inherits from [`generic_config`](../generic.md), adds the `name` discriminator, and overrides `from_dict` for polymorphic deserialization. Structurally identical to [optimizers/generic](../optimizers/generic.md).

## Class `lr_scheduler_config(generic_config)`

`@dataclass`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | `""` | Discriminator naming the concrete scheduler (`"exponential_lr"`, …). Concrete subclasses set and freeze it in `__post_init__`. |

### `from_dict(data: dict) -> lr_scheduler_config` (classmethod, overridden)

- **Called on the base class**: reads `name = str(data.get("name","")).lower()`, checks `LR_SCHEDULER_CONFIG_REGISTRY` (unknown → `ValueError` listing known names), delegates to `REGISTRY[name].from_dict(data)`.
- **Called on a concrete subclass**: `super().from_dict(data)` (standard [`generic_config.from_dict`](../generic.md)).

The registry is imported **lazily inside the method** to avoid a circular import with the package `__init__`.

The scheduler itself is later built from the config by `support_training.get_lr_scheduler`.
