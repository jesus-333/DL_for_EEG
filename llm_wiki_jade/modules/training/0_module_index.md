# `jade.training` — Module Index

Table of contents for the `training` subpackage wiki. Start with the [quickstart](0_module_quickstart.md).

| Page | Source file | What it covers |
|---|---|---|
| [0_module_quickstart.md](0_module_quickstart.md) | — | Read-first overview: purpose, end-to-end flow, config-driven design, logging, device, checkpointing, metrics, dependencies, connections. |
| [\_\_init\_\_.md](__init__.md) | `src/jade/training/__init__.py` | Package logger `NullHandler`, leaf-first import order, public API (`__all__`) aggregation. |
| [classifier.md](classifier.md) | `src/jade/training/classifier.py` | `train` / `train_model` (full training cycle) and their wandb counterparts `wandb_train` / `wandb_train_model`. |
| [classifier_epoch.md](classifier_epoch.md) | `src/jade/training/classifier_epoch.py` | `train_epoch_function` and `validation_epoch_function` (one epoch each). |
| [test_functions.md](test_functions.md) | `src/jade/training/test_functions.py` | `test` — generic loss+metrics evaluation on any dataset. |
| [support_training.md](support_training.md) | `src/jade/training/support_training.py` | Config→torch factories (`get_dataloader`, `get_optimizer`, `get_lr_scheduler`, `get_loss_function`), training helpers (`backup_every_epoch`, `update_log_dict_metrics`, `set_seed`), memory-profiling utilities. |
| [metrics.md](metrics.md) | `src/jade/training/metrics.py` | torchmetrics-based classification metrics and confusion matrix. |
| [logging_utils.md](logging_utils.md) | `src/jade/training/logging_utils.py` | `setup_logging` — opt-in root-logger configuration (console + optional file) for users who do not set up their own logger. |
