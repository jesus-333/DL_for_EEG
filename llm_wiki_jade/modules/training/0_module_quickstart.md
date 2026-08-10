# `jade.training` — Module Quickstart

**READ THIS FIRST.** After reading only this page you should understand the whole `training` module without opening the source. Depth over length; no code dumps.

## Purpose

`jade.training` is the subpackage that trains a model. Today it implements a single objective: **classification (finetuning)** of an EEG classifier. A reconstruction/pretraining cycle and a federated-learning submodule are anticipated by the docstrings but **not yet implemented** — do not assume they exist.

The module is a thin, config-driven orchestration layer on top of PyTorch. It does not define models or datasets. The standard entry point `train` takes a `model_config` (from `jade.config.model`) and builds the model itself via `jade.model.get_model`; the lower-level `train_model` takes an already-built `torch.nn.Module`. Both also take one or two `torch.utils.data.Dataset` objects and a `training_config` (from `jade.config`), and produce a trained model plus a dictionary of per-epoch metrics.

## End-to-end flow

```
train(training_config, model_config, …)                 wandb_train(training_config, model, …)
   │  cross_check_config + get_model(model_config)          │  opens wandb run + optional model Artifact
   └───────────────► train_model(training_config, model, …) ◄┘
                                                                 │
                       ┌─────────────────────────────────────────┤ per epoch
                       ▼                                          ▼
             train_epoch_function                    validation_epoch_function (optional)
               (forward + backward)                     (forward only, no grad)
                       │                                          │
                       └──────────────► metrics.compute_metrics ◄─┘  (optional, both sets)
```

1. **`train(training_config, model_config, train_dataset, validation_dataset=None, wandb_model_artifact=None)`** is the standard entry point. It runs `support_training.cross_check_config(training_config, model_config)` (fail fast on e.g. a loss ↔ `use_softmax` mismatch), builds the model with `jade.model.get_model(model_config)`, and delegates to `train_model`, forwarding every argument.
2. **`train_model(training_config, model, train_dataset, validation_dataset=None, wandb_model_artifact=None)`** runs the full cycle on an already-built model:
   - Promotes a plain `dict` config to a real `training_config`; reads `device` and whether wandb is requested.
   - Optionally seeds numpy/torch (`support_training.set_seed`).
   - Builds dataloaders, loss, optimizer and (optional) LR scheduler from the config via the `support_training` factories.
   - Moves model+loss to `device`; optionally resumes from past weights/optimizer state (`torch.load` + `load_state_dict`).
   - Creates the checkpoint folder; if wandb is on, calls `wandb.watch`.
   - Loops `epochs` times: one training epoch (`train_epoch_function`), optional validation epoch (`validation_epoch_function`), optional metric computation on both sets (`metrics.compute_metrics`), LR logging, `lr_scheduler.step()`, INFO logging, and wandb logging.
   - Returns `(model, computed_metrics_during_training)` — the trained model and a dict mapping each tracked quantity to its list of per-epoch values.
3. **`wandb_train(...)`** is a thin wrapper: it opens a `wandb.init(...)` run (job_type `"train"`, resume `"allow"`, run id derived from the seed), builds an optional model `wandb.Artifact`, calls `train_model` inside the run context (it already holds a built model), logs the artifact, and returns the result.
4. **`test(training_config, model, test_dataset, label="test")`** is the generic evaluator: it reuses `validation_epoch_function` for the loss and `metrics.compute_metrics` for the metrics, and returns `(test_loss, log_dict)` where every key is suffixed with `label`. It does **not** log to wandb itself — the caller does.

## How the `training_config` drives everything

`training_config` (`jade.config.training`) is a validated dataclass holding every knob. Its nested configs are **torch-free** by design (lightweight, serializable to TOML); `support_training` is the single place where they are *materialized* into real torch objects:

| Config field | Factory in `support_training` | Produces |
|---|---|---|
| `dataloader` (+ `batch_size`) | `get_dataloader` | `torch.utils.data.DataLoader` |
| `optimizer` (`name`: adam/adamw/sgd/rmsprop/lbfgs) | `get_optimizer` | `torch.optim.Optimizer` |
| `lr_scheduler` (`name`: exponential_lr/step_lr/cosine_annealing_lr/cosine_annealing_warm_restarts/cyclic_lr) or `None` | `get_lr_scheduler` | `torch.optim.lr_scheduler.*` |
| `loss` (`name`: cross_entropy/nll) | `get_loss_function` | `torch.nn.Module` |

Other important config fields: `epochs`, `device`, `path_to_save_model`, `epoch_to_save_model` (period of periodic checkpoints; a non-positive value is rewritten to `epochs + 1` so periodic saves never fire), `backup_model_every_epoch`, `measure_metrics_during_training`, `print_var`, `seed`, `path_past_weights`, `path_past_optimizer`, and the optional nested `wandb` (`wandb_training_config`). Every function accepts a plain dict for the config and promotes it via `training_config.from_dict`.

The **model** config is *not* nested in `training_config`: `train` takes it as a separate `model_config` argument. Before building the model, `train` calls `support_training.cross_check_config(training_config, model_config)`, which enforces the cross-config constraints — the loss ↔ `use_softmax` coupling (`nll` needs `use_softmax=True`, `cross_entropy` needs `use_softmax=False`) and the class-weight length ↔ `n_classes` match — raising `ValueError` on a mismatch.

## Logging approach

Library best-practice: **no `print`**. Each module owns `logger = logging.getLogger(__name__)`, and the package `__init__` attaches a `logging.NullHandler` to the `jade.training` logger so importing never forces logging config on the user. Per-epoch summaries are emitted at INFO level, gated by `training_config.print_var`. `logging`'s lazy `%`-formatting is used throughout (`logger.info("... %s", value)`).

For users who do not want to set up their own logger, [`logging_utils.setup_logging(level=logging.INFO, logfile=None)`](logging_utils.md) is an **opt-in** convenience: it configures the root logger with a `stdout` handler (and an optional file handler), a readable `asctime | level | name | message` format, and `force=True` so it can be re-called in a notebook without stacking duplicate handlers. It is torch-free (stdlib only).

## Device handling

`device` is a plain string from the config (`"cpu"`, `"cuda"`, `"cuda:0"`, …). Models and loss are moved with `.to(device)`; each epoch function moves the batch data and labels to `device` before the forward pass. In `metrics`, predictions and labels are pulled back to **CPU** before being concatenated and handed to torchmetrics.

## Checkpointing / model-artifact behavior

`train` writes several `.pth` files (state dicts, via `torch.save`) into `path_to_save_model`:

- `model_{n}.pth` + `optimizer_{n}.pth` every `epoch_to_save_model` epochs (periodic checkpoints).
- `model_BEST.pth` + `optimizer_BEST.pth` whenever a new **minimum validation loss** is reached (only when a validation set is given).
- `model_END.pth` once at the end of training (always).
- If `backup_model_every_epoch` is set, an alternating `model_backup_A.pth` / `model_backup_B.pth` (and optimizer/scheduler) is overwritten each epoch — a "backup of the backup" so an HPC job interrupted mid-write still leaves one intact file.

When wandb is active and `wandb.log_model_artifact` is true, periodic checkpoints, the final model, and the best model are added to the passed `wandb.Artifact` and uploaded via `wandb.save`.

## Metrics

`metrics.compute_metrics` returns a dict of five **scalar** floats: `accuracy` (micro-averaged), `cohen_kappa`, `sensitivity` (weighted recall), `specificity` (weighted), `f1` (weighted), computed with the `torchmetrics` multiclass **functional** API (no scikit-learn). Predicted labels come from `model.classify(x)` if the model exposes it (as `EEGNet_Classifier` does), otherwise `argmax(model(x))`. `num_classes` is inferred as `max(label)+1` (floored to 2) when not given. The confusion matrix is provided separately (`compute_confusion_matrix`) because it is a matrix, not a scalar, and is not logged among the scalar metrics.

## Dependencies

- **External:** `torch` (core), `torchmetrics` (metrics), `numpy` (seeding), stdlib `logging`/`os`/`sys`/`gc`/`time`/`threading`. `wandb` is an **optional** extra — imported in a `try/except`; a clear `ImportError` is raised only if wandb logging is actually requested without it installed. `psutil` is imported lazily inside the CPU memory profiler.
- **Internal:** `jade.config.training.training_config` (+ nested `dataloader`, `optimizers`, `lr_schedulers`, `loss_functions`, `wandb_training` configs and their `get_*_config` promoters); `jade.config.model.get_model_config` (for `cross_check_config`); `jade.model.get_model` (imported by `classifier` so `train` can build the model).

## How it connects to the rest of `jade`

- **Consumes `jade.config`** — the `training_config` and all nested configs live there; `support_training` turns them into torch objects. Configs stay torch-free so they can be serialized to TOML with one section per nested config. `train` also takes a `model_config` (from `jade.config.model`).
- **Builds / consumes a model** — `train` builds the model from a `model_config` via `jade.model.get_model`; `train_model`/`wandb_train`/`test` take an already-built `torch.nn.Module`. Metrics prefer the model's `classify` method (`jade.model.EEGNet.EEGNet_Classifier.classify`), falling back to `argmax`.
- **Consumes datasets** — any `torch.utils.data.Dataset` whose items are `(data, label)` pairs.
- **`wandb_train`** additionally logs the training config (and an optional `model_config`, promoted to a dict via `to_dict`, purely for reproducibility) to the wandb run.

## Public API (from `jade.training`)

`train`, `train_model`, `wandb_train`, `train_epoch_function`, `validation_epoch_function`, `test`, `setup_logging`, plus the submodules `support_training` (which also exposes `cross_check_config`) and `metrics`.

See the [module index](0_module_index.md) for per-file pages.
