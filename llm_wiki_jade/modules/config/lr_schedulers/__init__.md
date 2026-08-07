# `config/lr_schedulers/__init__.py`

Source: `src/jade/config/lr_schedulers/__init__.py`

Aggregator for the scheduler-config family. Imports the base and the five concrete configs, defines the registry and factory, declares `__all__`.

## `LR_SCHEDULER_CONFIG_REGISTRY`

```python
{"exponential_lr": exponential_lr_config,
 "step_lr": step_lr_config,
 "cosine_annealing_lr": cosine_annealing_lr_config,
 "cosine_annealing_warm_restarts": cosine_annealing_warm_restarts_config,
 "cyclic_lr": cyclic_lr_config}
```

Maps the lowercase scheduler name to its config class; used by `lr_scheduler_config.from_dict` for polymorphic dispatch (see [generic.md](generic.md)).

## `get_lr_scheduler_config(config) -> lr_scheduler_config`

| Input | Behavior |
|---|---|
| an `lr_scheduler_config` instance | returned untouched |
| a `dict` | promoted via `lr_scheduler_config.from_dict(config)` (dispatches on `name`) |
| anything else | raises `TypeError` |

Called by [`training_config.__post_init__`](../training.md) (only when `lr_scheduler is not None`) and by `support_training.get_lr_scheduler`.

## `__all__`

`lr_scheduler_config`, `exponential_lr_config`, `step_lr_config`, `cosine_annealing_lr_config`, `cosine_annealing_warm_restarts_config`, `cyclic_lr_config`, `LR_SCHEDULER_CONFIG_REGISTRY`, `get_lr_scheduler_config`.
