# `config/__init__.py`

Source: `src/jade/config/__init__.py`

Package aggregator. It imports and re-exports every configuration class, registry and factory from the module's files and subpackages, giving a single **flat** import surface (`from jade.config import training_config, adam_config, get_optimizer_config, ...`). It defines no classes or functions of its own.

## What it re-exports

Grouped exactly as in `__all__`:

| Group | Names |
|---|---|
| Base & dataset-related | `generic_config`, `filter_config`, `dataset_config`, `dataloader_config`, `wandb_training_config`, `training_config` |
| Optimizers | `optimizer_config`, `adam_config`, `adamw_config`, `sgd_config`, `rmsprop_config`, `lbfgs_config`, `OPTIMIZER_CONFIG_REGISTRY`, `get_optimizer_config` |
| LR schedulers | `lr_scheduler_config`, `exponential_lr_config`, `step_lr_config`, `cosine_annealing_lr_config`, `cosine_annealing_warm_restarts_config`, `cyclic_lr_config`, `LR_SCHEDULER_CONFIG_REGISTRY`, `get_lr_scheduler_config` |
| Loss functions | `loss_config`, `cross_entropy_config`, `nll_config`, `elbo_config`, `LOSS_CONFIG_REGISTRY`, `get_loss_config` |
| Model configs | `model_config`, `EEGNet_config`, `eegnet_pooling_config`, `EEGConformer_config`, `conformer_pooling_config`, `ShallowFBCSPNet_config`, `shallowfbcspnet_pooling_config`, `Deep4Net_config`, `deep4net_pooling_config`, `TIDNet_config`, `tidnet_pooling_config`, `vEEGNet_config`, `braindecode_wrapper_config`, `MODEL_CONFIG_REGISTRY`, `get_model_config` |
| Pooling configs | `pooling_config`, `avg_pooling_config`, `blur_pooling_config`, `sinc_pooling_config`, `dwt_pooling_config`, `max_pooling_config`, `POOLING_CONFIG_REGISTRY`, `get_pooling_config` |

## Notes

- The module docstring is the canonical high-level description of the package: it lists one file/subpackage per concern, **including the `model/` subpackage** (model + pooling configs).
- Import order matters for the family subpackages: importing `jade.config.optimizers` (etc.) triggers building of its `*_CONFIG_REGISTRY`. The base classes' `from_dict` methods import the registry **lazily** (inside the method) to avoid a circular import — see [optimizers/generic.md](optimizers/generic.md).
- `__all__` is explicit, so `from jade.config import *` yields exactly the names above.

See the [module quickstart](0_module_quickstart.md) for the design rationale.
