# `jade.training` — Module Quickstart

**READ THIS FIRST.** After reading only this page you should understand the whole `training` module without opening the source. Depth over length; no code dumps.

## Purpose

`jade.training` is the subpackage that trains a model. It implements **two objectives**, each with its own training cycle and its own single-epoch functions:

| Objective | Cycle | Epoch functions | Loss | Model |
|---|---|---|---|---|
| **classification** (finetuning) | [`classifier`](classifier.md) | [`classifier_epoch`](classifier_epoch.md) | `cross_entropy` / `nll` | any `*_Classifier` |
| **generative** (variational autoencoder) | [`vae`](vae.md) | [`vae_epoch`](vae_epoch.md) | [`elbo`](custom_loss_function/elbo.md) | [`vEEGNet`](../model/vEEGNet.md) |

The two cycles expose the **same four entry points with the same signatures**, so moving a run from one to the other only means importing another module. A federated-learning submodule is anticipated by the docstrings but **not yet implemented** — do not assume it exists.

The module is a thin, config-driven orchestration layer on top of PyTorch. It does not define models or datasets. The standard entry point `train` takes a `model_config` (from `jade.config.model`) and builds the model itself via `jade.model.get_model`; the lower-level `train_model` takes an already-built `torch.nn.Module`. Both also take one or two `torch.utils.data.Dataset` objects and a `training_config` (from `jade.config`), and produce a trained model plus a dictionary of per-epoch metrics.

## End-to-end flow

The shape below is the same in `classifier` and in `vae`; only the epoch functions and the metric function differ.

```
build-the-model entry points          already-built-model workers
─────────────────────────────          ───────────────────────────
train(training_config, model_config, …)      ──►  train_model(training_config, model, …)
   │  cross_check + get_model                          ▲
   │                                                   │ (calls inside the run)
wandb_train(training_config, model_config, …) ──► wandb_train_model(training_config, model, …)
   │  cross_check + get_model                          │  wandb_utils.run_training_with_wandb
                                                       ▼  opens wandb run + optional model Artifact
                       ┌───────────────────────── per epoch (in train_model) ─────────┐
                       ▼                                                               ▼
             train_epoch_function                                  validation_epoch_function (optional)
               (forward + backward)                                   (forward only, no grad)
                       │                                                               │
                       └────────────────►  metrics function  ◄──────────────────────────┘  (optional, both sets)

                 classifier: classifier_epoch.*  +  metrics.compute_metrics
                 vae       : vae_epoch.*         +  metrics.compute_reconstruction_metrics
                             (+ loss_function.set_epoch(epoch) at the top of the epoch)
```

1. **`train(training_config, model_config, train_dataset, validation_dataset=None, wandb_model_artifact=None)`** is the standard entry point. It checks both configs are dataclass instances, runs `support_training.cross_check_config(training_config, model_config)` (fail fast on e.g. a loss ↔ `use_softmax` mismatch), builds the model with `jade.model.get_model(model_config)`, and delegates to `train_model`, forwarding every argument.
2. **`train_model(training_config, model, train_dataset, validation_dataset=None, wandb_model_artifact=None)`** runs the full cycle on an already-built model:
   - Checks `training_config` is a real `training_config` instance (**no dict promotion**); reads `device` and whether wandb is requested.
   - Optionally seeds numpy/torch (`support_training.set_seed`).
   - Builds dataloaders, loss, optimizer and (optional) LR scheduler from the config via the `support_training` factories.
   - Moves model+loss to `device`; optionally resumes from past weights/optimizer state (`torch.load` + `load_state_dict`).
   - Creates the checkpoint folder; if wandb is on, calls `wandb.watch`.
   - Loops `epochs` times: one training epoch (`train_epoch_function`), optional validation epoch (`validation_epoch_function`), optional metric computation on both sets (`metrics.compute_metrics`), LR logging, `lr_scheduler.step()`, INFO logging, and wandb logging.
   - Returns `(model, computed_metrics_during_training)` — the trained model and a dict mapping each tracked quantity to its list of per-epoch values.
3. **`wandb_train(training_config, model_config, …, dataset_config=None)`** mirrors `train` for a wandb-tracked run: it checks the configs, cross-checks, builds the model, and delegates to `wandb_train_model` (forwarding `model_config` and `dataset_config` for logging). **`wandb_train_model(training_config, model, …, model_config=None, dataset_config=None)`** mirrors `train_model` (already-built model) and delegates the whole run to [`wandb_utils.run_training_with_wandb`](wandb_utils.md), which opens a `wandb.init(...)` run (job_type `"train"`, resume `"allow"`, run id derived from the seed), logs the `training_config` (plus the optional `model_config` and `dataset_config`, each `to_dict`-ed, **for reproducibility only**), builds an optional model `wandb.Artifact`, calls the cycle's own `train_model` inside the run context, logs the artifact, and returns the result. Since `run_training_with_wandb` takes the training function as its first argument, **the same code serves both cycles** and the public signatures are untouched.
4. **`test(training_config, model, test_dataset, label="test")`** is the generic evaluator: it reuses `validation_epoch_function` for the loss and `metrics.compute_metrics` for the metrics, and returns `(test_loss, log_dict)` where every key is suffixed with `label`. It does **not** log to wandb itself — the caller does.

## How the `training_config` drives everything

`training_config` (`jade.config.training`) is a validated dataclass holding every knob. Its nested configs are **torch-free** by design (lightweight, serializable to TOML); `support_training` is the single place where they are *materialized* into real torch objects:

| Config field | Factory in `support_training` | Produces |
|---|---|---|
| `dataloader` (+ `batch_size`) | `get_dataloader` | `torch.utils.data.DataLoader` |
| `optimizer` (`name`: adam/adamw/sgd/rmsprop/lbfgs) | `get_optimizer` | `torch.optim.Optimizer` |
| `lr_scheduler` (`name`: exponential_lr/step_lr/cosine_annealing_lr/cosine_annealing_warm_restarts/cyclic_lr) or `None` | `get_lr_scheduler` | `torch.optim.lr_scheduler.*` |
| `loss` (`name`: cross_entropy/nll/**elbo**) | `get_loss_function` | `torch.nn.Module` — for `elbo` it dispatches to [`custom_loss_function`](custom_loss_function/0_module_quickstart.md) |

Other important config fields: `epochs`, `device`, `path_to_save_model`, `epoch_to_save_model` (period of periodic checkpoints; a non-positive value is rewritten to `epochs + 1` so periodic saves never fire), `backup_model_every_epoch`, `measure_metrics_during_training`, `print_var`, `seed`, `path_past_weights`, `path_past_optimizer`, and the optional nested `wandb` (`wandb_training_config`). All four entry points accept only config **dataclass instances** (no dict promotion) — a wrong type raises `TypeError`; build the config with `training_config.from_dict(...)` first if you have a dict.

The **model** config is *not* nested in `training_config`: `train` takes it as a separate `model_config` argument. Before building the model, `train` calls `support_training.cross_check_config(training_config, model_config)`, which enforces the cross-config constraints — the loss ↔ `use_softmax` coupling (`nll` needs `use_softmax=True`, `cross_entropy` needs `use_softmax=False`), the class-weight length ↔ `n_classes` match, and the **training objective** (a generative model must be paired with a generative loss, and the other way around) — raising `ValueError` on a mismatch.

## Logging approach

Library best-practice: **no `print`**. Each module owns `logger = logging.getLogger(__name__)`, and the package `__init__` attaches a `logging.NullHandler` to the `jade.training` logger so importing never forces logging config on the user. Per-epoch summaries are emitted at INFO level, gated by `training_config.print_var`. `logging`'s lazy `%`-formatting is used throughout (`logger.info("... %s", value)`).

For users who do not want to set up their own logger, [`logging_utils.setup_logging(level=logging.INFO, logfile=None, format=None, datefmt=None)`](logging_utils.md) is an **opt-in** convenience: it configures the root logger with a `stdout` handler (and an optional file handler), a readable `asctime | level | name | message` format (overridable via `format`/`datefmt`, which default to `DEFAULT_LOG_FORMAT`/`DEFAULT_DATE_FORMAT` when left `None`), and `force=True` so it can be re-called in a notebook without stacking duplicate handlers. It is torch-free (stdlib only).

## Device handling

`device` is a plain string from the config (`"cpu"`, `"cuda"`, `"cuda:0"`, …). Models and loss are moved with `.to(device)`; each epoch function moves the batch data and labels to `device` before the forward pass. In `metrics`, predictions and labels are pulled back to **CPU** before being concatenated and handed to torchmetrics.

## Checkpointing / model-artifact behavior

`train_model` writes several `.pth` files (state dicts, via `torch.save`) into `path_to_save_model` (`train` itself only builds the model and delegates to `train_model`, which owns the entire epoch loop and all checkpointing):

- `model_{n}.pth` + `optimizer_{n}.pth` every `epoch_to_save_model` epochs (periodic checkpoints).
- `model_BEST.pth` + `optimizer_BEST.pth` whenever a new **minimum validation loss** is reached (only when a validation set is given).
- `model_END.pth` once at the end of training (always).
- If `backup_model_every_epoch` is set, an alternating `model_backup_A.pth` / `model_backup_B.pth` (and optimizer/scheduler) is overwritten each epoch — a "backup of the backup" so an HPC job interrupted mid-write still leaves one intact file.

When wandb is active and `wandb.log_model_artifact` is true, periodic checkpoints, the final model, and the best model are added to the passed `wandb.Artifact` and uploaded via `wandb.save`.

## Metrics

For a **generative** model these do not apply (there is no label to compare): `metrics.compute_reconstruction_metrics` returns `reconstruction_mse` and `reconstruction_mae` instead, preferring `model.reconstruct(x)` and falling back to `model(x)[0]` — the same fallback idiom used for `classify`. It is gated by the same `measure_metrics_during_training` flag.

`metrics.compute_metrics` returns a dict of five **scalar** floats: `accuracy` (micro-averaged), `cohen_kappa`, `sensitivity` (weighted recall), `specificity` (weighted), `f1` (weighted), computed with the `torchmetrics` multiclass **functional** API (no scikit-learn). Predicted labels come from `model.classify(x)` if the model exposes it (as `EEGNet_Classifier` does), otherwise `argmax(model(x))`. `num_classes` is inferred as `max(label)+1` (floored to 2) when not given. The confusion matrix is provided separately (`compute_confusion_matrix`) because it is a matrix, not a scalar, and is not logged among the scalar metrics.

## Dependencies

- **External:** `torch` (core), `torchmetrics` (metrics), `numpy` (seeding), stdlib `logging`/`os`/`sys`/`gc`/`time`/`threading`. `wandb` is an **optional** extra — imported in a `try/except` **inside [`wandb_utils`](wandb_utils.md) only**, so no training cycle imports it directly; a clear `ImportError` is raised only if wandb logging is actually requested without it installed. `psutil` is imported lazily inside the CPU memory profiler.
- **Internal:** `jade.config.training.training_config` (+ nested `dataloader`, `optimizers`, `lr_schedulers`, `loss_functions`, `wandb_training` configs and their `get_*_config` promoters); `jade.config.model.get_model_config` (for `cross_check_config`); `jade.model.get_model` (imported by `classifier` so `train` can build the model).

## How it connects to the rest of `jade`

- **Consumes `jade.config`** — the `training_config` and all nested configs live there; `support_training` turns them into torch objects. Configs stay torch-free so they can be serialized to TOML with one section per nested config. `train` also takes a `model_config` (from `jade.config.model`).
- **Builds / consumes a model** — `train`/`wandb_train` build the model from a `model_config` via `jade.model.get_model`; `train_model`/`wandb_train_model`/`test` take an already-built `torch.nn.Module`. Classification metrics prefer the model's `classify` method (`jade.model.EEGNet.EEGNet_Classifier.classify`), falling back to `argmax`; reconstruction metrics prefer `reconstruct` (`jade.model.vEEGNet.vEEGNet.reconstruct`), falling back to `model(x)[0]`.
- **Consumes datasets** — any `torch.utils.data.Dataset` whose items are `(data, label)` pairs.
- **`wandb_train` / `wandb_train_model`** additionally log the training config (and the optional `model_config` and `dataset_config`, each serialized via `to_dict`, purely for reproducibility — they are not used during training) to the wandb run.

## Public API (from `jade.training`)

The **classification** entry points keep their plain names (`train`, `train_model`, `wandb_train`, `wandb_train_model`, `train_epoch_function`, `validation_epoch_function`), and the **generative** twins are exported with a `vae_` prefix (`vae_train`, `vae_train_model`, `vae_wandb_train`, `vae_wandb_train_model`, `vae_train_epoch_function`, `vae_validation_epoch_function`), so nothing that already imported the former changes meaning. Plus `test`, `setup_logging`, and the submodules `classifier`, `classifier_epoch`, `vae`, `vae_epoch`, `custom_loss_function`, `wandb_utils`, `support_training` (which also exposes `cross_check_config`) and `metrics`.

**For new code the preferred style is the module-qualified call** — `from jade.training import vae` then `vae.train_model(...)` — which is already how the scripts under `training/EEGNet/` import `classifier`.

See the [module index](0_module_index.md) for per-file pages.
