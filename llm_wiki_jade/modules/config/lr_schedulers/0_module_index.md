# `jade.config.lr_schedulers` — Subpackage Index

Table of contents. Start with the [subpackage quickstart](0_module_quickstart.md). Parent: [config index](../0_module_index.md).

| File | Page | Description |
|---|---|---|
| `__init__.py` | [__init__.md](__init__.md) | `LR_SCHEDULER_CONFIG_REGISTRY` + `get_lr_scheduler_config` factory + re-exports. |
| `generic.py` | [generic.md](generic.md) | `lr_scheduler_config` base class + polymorphic `from_dict`. |
| `exponential_lr.py` | [exponential_lr.md](exponential_lr.md) | `exponential_lr_config` → `torch.optim.lr_scheduler.ExponentialLR`. |
| `step_lr.py` | [step_lr.md](step_lr.md) | `step_lr_config` → `StepLR`. |
| `cosine_annealing_lr.py` | [cosine_annealing_lr.md](cosine_annealing_lr.md) | `cosine_annealing_lr_config` → `CosineAnnealingLR`. |
| `cosine_annealing_warm_restarts.py` | [cosine_annealing_warm_restarts.md](cosine_annealing_warm_restarts.md) | `cosine_annealing_warm_restarts_config` → `CosineAnnealingWarmRestarts`. |
| `cyclic_lr.py` | [cyclic_lr.md](cyclic_lr.md) | `cyclic_lr_config` → `CyclicLR`. |
