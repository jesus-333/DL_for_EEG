# `jade.config` — Module Index

Compact table of contents for the `config` module. Start with the [module quickstart](0_module_quickstart.md).

## Root files

| File | Page | One-line description |
|---|---|---|
| `__init__.py` | [__init__.md](__init__.md) | Package aggregator: re-exports every config, registry and factory; declares `__all__`. |
| `generic.py` | [generic.md](generic.md) | `generic_config` base class + (de)serialization helpers (`to_dict`/`from_dict`/`to_toml`/`from_toml`/`to_json`/`from_json`) and private helpers. |
| `filter.py` | [filter.md](filter.md) | `filter_config`: parameters of one moabb/MNE filtering operation (bandpass/bandstop/lowpass/highpass/notch). |
| `dataset.py` | [dataset.md](dataset.md) | `dataset_config`: moabb dataset download + preprocessing; holds a nested `filter_config`. |
| `dataloader.py` | [dataloader.md](dataloader.md) | `dataloader_config`: parameters forwarded to `torch.utils.data.DataLoader` (no `batch_size`). |
| `wandb_training.py` | [wandb_training.md](wandb_training.md) | `wandb_training_config`: Weights & Biases logging parameters. |
| `training.py` | [training.md](training.md) | `training_config`: training hyper-parameters; holds the nested optimizer/scheduler/loss/dataloader/wandb configs. |

## Subpackage `optimizers/`

See the [optimizers quickstart](optimizers/0_module_quickstart.md) and [index](optimizers/0_module_index.md).

| File | Page | Description |
|---|---|---|
| `optimizers/__init__.py` | [optimizers/__init__.md](optimizers/__init__.md) | Registry `OPTIMIZER_CONFIG_REGISTRY` + factory `get_optimizer_config`. |
| `optimizers/generic.py` | [optimizers/generic.md](optimizers/generic.md) | `optimizer_config` base + polymorphic `from_dict`. |
| `optimizers/adam.py` | [optimizers/adam.md](optimizers/adam.md) | `adam_config` → `torch.optim.Adam`. |
| `optimizers/adamw.py` | [optimizers/adamw.md](optimizers/adamw.md) | `adamw_config` → `torch.optim.AdamW`. |
| `optimizers/sgd.py` | [optimizers/sgd.md](optimizers/sgd.md) | `sgd_config` → `torch.optim.SGD`. |
| `optimizers/rmsprop.py` | [optimizers/rmsprop.md](optimizers/rmsprop.md) | `rmsprop_config` → `torch.optim.RMSprop`. |
| `optimizers/lbfgs.py` | [optimizers/lbfgs.md](optimizers/lbfgs.md) | `lbfgs_config` → `torch.optim.LBFGS`. |

## Subpackage `lr_schedulers/`

See the [lr_schedulers quickstart](lr_schedulers/0_module_quickstart.md) and [index](lr_schedulers/0_module_index.md).

| File | Page | Description |
|---|---|---|
| `lr_schedulers/__init__.py` | [lr_schedulers/__init__.md](lr_schedulers/__init__.md) | Registry `LR_SCHEDULER_CONFIG_REGISTRY` + factory `get_lr_scheduler_config`. |
| `lr_schedulers/generic.py` | [lr_schedulers/generic.md](lr_schedulers/generic.md) | `lr_scheduler_config` base + polymorphic `from_dict`. |
| `lr_schedulers/exponential_lr.py` | [lr_schedulers/exponential_lr.md](lr_schedulers/exponential_lr.md) | `exponential_lr_config` → `ExponentialLR`. |
| `lr_schedulers/step_lr.py` | [lr_schedulers/step_lr.md](lr_schedulers/step_lr.md) | `step_lr_config` → `StepLR`. |
| `lr_schedulers/cosine_annealing_lr.py` | [lr_schedulers/cosine_annealing_lr.md](lr_schedulers/cosine_annealing_lr.md) | `cosine_annealing_lr_config` → `CosineAnnealingLR`. |
| `lr_schedulers/cosine_annealing_warm_restarts.py` | [lr_schedulers/cosine_annealing_warm_restarts.md](lr_schedulers/cosine_annealing_warm_restarts.md) | `cosine_annealing_warm_restarts_config` → `CosineAnnealingWarmRestarts`. |
| `lr_schedulers/cyclic_lr.py` | [lr_schedulers/cyclic_lr.md](lr_schedulers/cyclic_lr.md) | `cyclic_lr_config` → `CyclicLR`. |

## Subpackage `loss_functions/`

See the [loss_functions quickstart](loss_functions/0_module_quickstart.md) and [index](loss_functions/0_module_index.md).

| File | Page | Description |
|---|---|---|
| `loss_functions/__init__.py` | [loss_functions/__init__.md](loss_functions/__init__.md) | Registry `LOSS_CONFIG_REGISTRY` + factory `get_loss_config`. |
| `loss_functions/generic.py` | [loss_functions/generic.md](loss_functions/generic.md) | `loss_config` base + polymorphic `from_dict`. |
| `loss_functions/cross_entropy.py` | [loss_functions/cross_entropy.md](loss_functions/cross_entropy.md) | `cross_entropy_config` → `torch.nn.CrossEntropyLoss` (logits). |
| `loss_functions/nll.py` | [loss_functions/nll.md](loss_functions/nll.md) | `nll_config` → `torch.nn.NLLLoss` (log-probabilities). |
