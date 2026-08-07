# `config/optimizers/__init__.py`

Source: `src/jade/config/optimizers/__init__.py`

Aggregator for the optimizer-config family. Imports the base class and the five concrete configs, defines the name→class registry and the promotion factory, and declares `__all__`.

## `OPTIMIZER_CONFIG_REGISTRY`

```python
{"adam": adam_config, "adamw": adamw_config, "sgd": sgd_config,
 "rmsprop": rmsprop_config, "lbfgs": lbfgs_config}
```

Maps the lowercase optimizer name to its config class. Used by `optimizer_config.from_dict` for polymorphic dispatch (see [generic.md](generic.md)). Adding an optimizer = add a file + one entry here.

## `get_optimizer_config(config) -> optimizer_config`

Factory that normalizes any input to a concrete `optimizer_config` instance.

| Input | Behavior |
|---|---|
| an `optimizer_config` instance | returned untouched |
| a `dict` | promoted via `optimizer_config.from_dict(config)` (dispatches on `name`) |
| anything else | raises `TypeError` |

This is what [`training_config.__post_init__`](../training.md) calls to promote its `optimizer` field, and what `support_training.get_optimizer` calls before building the torch optimizer.

## `__all__`

`optimizer_config`, `adam_config`, `adamw_config`, `sgd_config`, `rmsprop_config`, `lbfgs_config`, `OPTIMIZER_CONFIG_REGISTRY`, `get_optimizer_config`.
