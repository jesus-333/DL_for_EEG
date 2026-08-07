# `jade.config.optimizers` — Subpackage Index

Table of contents. Start with the [subpackage quickstart](0_module_quickstart.md). Parent: [config index](../0_module_index.md).

| File | Page | Description |
|---|---|---|
| `__init__.py` | [__init__.md](__init__.md) | `OPTIMIZER_CONFIG_REGISTRY` + `get_optimizer_config` factory + re-exports. |
| `generic.py` | [generic.md](generic.md) | `optimizer_config` base class + polymorphic `from_dict`. |
| `adam.py` | [adam.md](adam.md) | `adam_config` → `torch.optim.Adam`. |
| `adamw.py` | [adamw.md](adamw.md) | `adamw_config` → `torch.optim.AdamW`. |
| `sgd.py` | [sgd.md](sgd.md) | `sgd_config` → `torch.optim.SGD`. |
| `rmsprop.py` | [rmsprop.md](rmsprop.md) | `rmsprop_config` → `torch.optim.RMSprop`. |
| `lbfgs.py` | [lbfgs.md](lbfgs.md) | `lbfgs_config` → `torch.optim.LBFGS`. |
