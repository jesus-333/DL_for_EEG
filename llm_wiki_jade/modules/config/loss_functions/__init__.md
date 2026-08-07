# `config/loss_functions/__init__.py`

Source: `src/jade/config/loss_functions/__init__.py`

Aggregator for the loss-config family. Imports the base and the two concrete configs, defines the registry and factory, declares `__all__`.

## `LOSS_CONFIG_REGISTRY`

```python
{"cross_entropy": cross_entropy_config, "nll": nll_config}
```

Maps the lowercase loss name to its config class; used by `loss_config.from_dict` for polymorphic dispatch (see [generic.md](generic.md)).

## `get_loss_config(config) -> loss_config`

| Input | Behavior |
|---|---|
| a `loss_config` instance | returned untouched |
| a `dict` | promoted via `loss_config.from_dict(config)` (dispatches on `name`) |
| anything else | raises `TypeError` |

Called by [`training_config.__post_init__`](../training.md) to promote its `loss` field, and by `support_training.get_loss_function`.

## `__all__`

`loss_config`, `cross_entropy_config`, `nll_config`, `LOSS_CONFIG_REGISTRY`, `get_loss_config`.
