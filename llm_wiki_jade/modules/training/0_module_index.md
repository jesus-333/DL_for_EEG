# `jade.training` — Module Index

Table of contents for the `training` subpackage wiki. Start with the [quickstart](0_module_quickstart.md).

| Page | Source file | What it covers |
|---|---|---|
| [0_module_quickstart.md](0_module_quickstart.md) | — | Read-first overview: purpose, end-to-end flow, config-driven design, logging, device, checkpointing, metrics, dependencies, connections. |
| [\_\_init\_\_.md](__init__.md) | `src/jade/training/__init__.py` | Package logger `NullHandler`, leaf-first import order, public API (`__all__`) aggregation. |
| [classifier.md](classifier.md) | `src/jade/training/classifier.py` | `train` / `train_model` (full training cycle) and their wandb counterparts `wandb_train` / `wandb_train_model`. |
| [classifier_epoch.md](classifier_epoch.md) | `src/jade/training/classifier_epoch.py` | `train_epoch_function` and `validation_epoch_function` (one epoch each). |
| [vae.md](vae.md) | `src/jade/training/vae.py` | The same four entry points for a **generative** model (e.g. `vEEGNet`), with the same signatures. |
| [vae_epoch.md](vae_epoch.md) | `src/jade/training/vae_epoch.py` | The single-epoch functions of a generative model; they return the dict of loss terms instead of a float. |
| [wandb_utils.md](wandb_utils.md) | `src/jade/training/wandb_utils.py` | Everything about a wandb run (run/artifact setup, `watch`, `log`), shared by the two training cycles. |
| [test_functions.md](test_functions.md) | `src/jade/training/test_functions.py` | `test` — generic loss+metrics evaluation on any dataset. |
| [support_training.md](support_training.md) | `src/jade/training/support_training.py` | Config→torch factories (`get_dataloader`, `get_optimizer`, `get_lr_scheduler`, `get_loss_function`), the cross-config checks (`cross_check_config`, `require_config_instance`), training helpers (`backup_every_epoch`, `update_log_dict_metrics`, `set_seed`), memory-profiling utilities. |
| [metrics.md](metrics.md) | `src/jade/training/metrics.py` | torchmetrics-based classification metrics and confusion matrix, plus the reconstruction metrics of a generative model. |
| [logging_utils.md](logging_utils.md) | `src/jade/training/logging_utils.py` | `setup_logging` — opt-in root-logger configuration (console + optional file) for users who do not set up their own logger. |

## Subpackage `custom_loss_function/` (`src/jade/training/custom_loss_function/`)

The losses with no `torch.nn` counterpart. Their configs live in [`jade.config.loss_functions`](../config/loss_functions/0_module_index.md), not here.

| Page | Source | One-line description |
|---|---|---|
| [custom_loss_function/0_module_quickstart.md](custom_loss_function/0_module_quickstart.md) | — | Read-first overview of the subpackage. |
| [custom_loss_function/0_module_index.md](custom_loss_function/0_module_index.md) | — | Subpackage table of contents. |
| [custom_loss_function/__init__.md](custom_loss_function/__init__.md) | `custom_loss_function/__init__.py` | `CUSTOM_LOSS_REGISTRY` / `get_custom_loss_function` factory. |
| [custom_loss_function/elbo.md](custom_loss_function/elbo.md) | `custom_loss_function/elbo.py` | `elbo_loss` (the VAE objective) and the `elbo_terms` named tuple. |
| [custom_loss_function/support_loss.md](custom_loss_function/support_loss.md) | `custom_loss_function/support_loss.py` | The reconstruction terms + registry, `kl_divergence_normal`, `compute_kl_weight`. |
