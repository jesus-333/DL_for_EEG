# `jade.training.classifier`

Source: `src/jade/training/classifier.py`

The full classification training cycle. `train` runs the entire loop (dataloaders, optimizer/scheduler/loss creation, epoch loop, checkpointing, optional metrics, optional wandb logging) and returns the trained model plus the per-epoch metrics; `wandb_train` is a thin wrapper that opens a wandb run around a call to `train`. Only the classification (finetuning) objective exists for now.

## Module-level facts

- `wandb` is imported inside a `try/except ImportError` and set to `None` if absent (optional extra).
- `training_config` from `jade.config.training` is imported under the alias `training_config_class` (used for `isinstance` checks and `from_dict` promotion).
- Module logger: `logger = logging.getLogger(__name__)`.

---

## `train(training_config, model, train_dataset, validation_dataset=None, wandb_model_artifact=None)`

Trains a classifier and returns it together with the metrics collected during training.

### Parameters

| Name | Type | Default | Meaning |
|---|---|---|---|
| `training_config` | `training_config` or `dict` | — | Full run configuration. A dict is promoted via `training_config_class.from_dict`. Holds every hyper-parameter and the nested optimizer/scheduler/loss/dataloader/wandb configs. |
| `model` | `torch.nn.Module` | — | Model to train. For metrics it should expose a `classify` method (as `jade.model.EEGNet.EEGNet_Classifier` does). |
| `train_dataset` | `torch.utils.data.Dataset` | — | Training data; each item is a `(data, label)` pair. |
| `validation_dataset` | `torch.utils.data.Dataset` | `None` | Validation data. If `None`, no validation is performed and **no `model_BEST.pth`** is saved. |
| `wandb_model_artifact` | `wandb.Artifact` | `None` | Artifact that checkpoints are added to. Normally created and passed by `wandb_train`. |

### Returns

`(model, computed_metrics_during_training)`:

- `model : torch.nn.Module` — the trained model (mutated in place; also returned for convenience).
- `computed_metrics_during_training : dict` — maps each tracked quantity to the **list of its per-epoch values**. Keys: `"train_loss"`, `"validation_loss"` (only if validating), and, when `measure_metrics_during_training` is on, `"{metric}_train"` and `"{metric}_validation"` for each metric in the metrics dict (`accuracy`, `cohen_kappa`, `sensitivity`, `specificity`, `f1`).

### Flow, step by step

1. **Config check & setup** — promote dict → `training_config`; alias `device = training_config.device`, `use_wandb = training_config.wandb is not None`. If `use_wandb` but `wandb is None`, raise `ImportError`. If `training_config.seed is not None`, call `support_training.set_seed`.
2. **Dataloaders** — `support_training.get_dataloader(train_dataset, batch_size, dataloader)`; same for validation if present, else `validation_loader = None`.
3. **Loss / optimizer / scheduler** — `model.to(device)`; optionally load `path_past_weights` into the model (`torch.load(..., map_location=device)` + `load_state_dict`); build loss via `get_loss_function` and move to device (matters when the loss carries class weights); build optimizer via `get_optimizer`; optionally load `path_past_optimizer` state; build LR scheduler via `get_lr_scheduler` if `lr_scheduler` config is set, else `None`.
4. **Training preparation** — `os.makedirs(path_to_save_model, exist_ok=True)`; if `use_wandb`, `wandb.watch(model, criterion=loss_function, log="all", log_freq=wandb.log_freq, log_graph=True)`; `best_loss_val = sys.maxsize`; init `log_dict = {}` and `computed_metrics_during_training = dict()`.
5. **Epoch loop** (`for epoch in range(epochs)`):
   - **Training epoch** (mandatory): `train_loss = train_epoch_function(model, train_loader, loss_function, optimizer, device, log_dict, print_var)`.
   - **Periodic checkpoint**: if `(epoch + 1) % epoch_to_save_model == 0`, save `model_{epoch+1}.pth` and `optimizer_{epoch+1}.pth`.
   - **A/B backup** (optional): if `backup_model_every_epoch`, call `support_training.backup_every_epoch`.
   - Accumulate `train_loss` into the list (create on epoch 0, append after).
   - **Validation epoch** (optional): if `validation_loader`, `validation_loss = validation_epoch_function(...)`. If `validation_loss < best_loss_val`, update `best_loss_val` and save `model_BEST.pth` + `optimizer_BEST.pth`. Accumulate `validation_loss`.
   - **Metrics** (optional): if `measure_metrics_during_training`, compute `metrics.compute_metrics` on train (and validation) loaders and accumulate under `"{metric}_train"` / `"{metric}_validation"`.
   - **LR logging & scheduler step**: if `use_wandb`, store `log_dict["learning_rate"] = optimizer.param_groups[0]["lr"]` **before** stepping; then `lr_scheduler.step()` if a scheduler exists.
   - **INFO logging** (optional): if `print_var`, log epoch index, train loss, validation loss, learning rate, and train/validation accuracy (whichever apply).
   - **wandb logging** (optional): set `log_dict["train_loss"]` (and `"validation_loss"`), merge metrics via `support_training.update_log_dict_metrics(..., "train"/"validation")`, add the periodic checkpoint file to the artifact (guarded by `wandb.log_model_artifact` and a non-`None` artifact), and `wandb.log(log_dict)`.
6. **End of training** — always save `model_END.pth`. If wandb + `log_model_artifact` + artifact present, add `model_END.pth` (and `model_BEST.pth` when validating) to the artifact and `wandb.save` them.
7. Return `(model, computed_metrics_during_training)`.

### Side effects

- **Filesystem writes** (via `torch.save`, state dicts): `model_{n}.pth`, `optimizer_{n}.pth`, `model_BEST.pth`, `optimizer_BEST.pth`, `model_END.pth`, plus A/B backups. Creates `path_to_save_model`.
- **In-place model mutation** — trained weights, and `model.to(device)`.
- **Logging** — INFO records when `print_var`.
- **wandb** — `wandb.watch`, `wandb.log`, `wandb.save`, artifact `add_file` when enabled.

### Raised errors

- `ImportError` if wandb logging is requested but wandb is not installed.
- Propagates any error from the factories (`support_training.get_*`) or torch (e.g. a bad `state_dict` load).

### Gotchas / design notes

- **`log_dict` is never reset** between epochs: the same keys are overwritten each epoch, which is functionally equivalent to resetting for wandb's per-step logging.
- **Two dicts, two purposes:** `computed_metrics_during_training` stores per-epoch **lists** and is returned; `log_dict` holds a **single** epoch's values and feeds wandb.
- **`epoch_to_save_model` semantics** live in the config: a non-positive value is rewritten to `epochs + 1` in `training_config.__post_init__`, so the modulo test never fires and only `model_END.pth`/`model_BEST.pth` are written.
- **Off-by-one naming:** at loop index `epoch == n` the model has been trained for `n + 1` epochs; checkpoint files use `epoch + 1`.
- The learning rate is logged **before** `lr_scheduler.step()`, so the value logged for an epoch is the rate actually used during it.

---

## `wandb_train(training_config, model, train_dataset, validation_dataset=None, model_config=None)`

Trains while tracking the run on wandb. Opens a `wandb.init` run (and optional model `wandb.Artifact`), then calls `train` inside it. See [classifier_epoch.md](classifier_epoch.md) and this page's `train` for the training details.

### Parameters

| Name | Type | Default | Meaning |
|---|---|---|---|
| `training_config` | `training_config` or `dict` | — | Config; its `wandb` field (a `wandb_training_config`) **must** be set. Dict is promoted. |
| `model` | `torch.nn.Module` | — | Model to train. |
| `train_dataset` | `torch.utils.data.Dataset` | — | Training data. |
| `validation_dataset` | `torch.utils.data.Dataset` | `None` | Validation data. |
| `model_config` | `dict` | `None` | Not used during training — only logged to wandb (with the training config) for reproducibility. |

### Returns

`(model, computed_metrics_during_training)` — exactly what `train` returns.

### Flow

1. Promote dict → config.
2. Raise `ImportError` if `wandb is None`; raise `ValueError` if `training_config.wandb is None`.
3. Build `config_to_log = {"training_config": training_config.to_dict()}`, adding `"model_config"` when provided.
4. `run_id = str(training_config.seed)` if a seed is set, else `None` — so a seeded run can be **resumed** under the same id.
5. `with wandb.init(project=..., entity=..., name=..., job_type="train", config=config_to_log, notes=..., id=run_id, resume="allow") as run:`
   - Artifact name = `wandb.model_artifact_name + "_trained"`. If `log_model_artifact`, create a `wandb.Artifact(type="model", description=..., metadata=config_to_log)`, else `None`.
   - Call `train(training_config, model, train_dataset, validation_dataset, wandb_model_artifact)`.
   - If `log_model_artifact`, `run.log_artifact(wandb_model_artifact)`.
6. Return `train`'s result.

### Side effects / errors

- Opens (and closes) a wandb run; logs config, watches the model, logs metrics/artifacts (through `train`).
- `ImportError` (wandb missing) / `ValueError` (`wandb` config unset). Any error inside the `with` still lets the run context close.

### Related pages

- Single-epoch logic: [classifier_epoch.md](classifier_epoch.md)
- Factories/helpers: [support_training.md](support_training.md)
- Metrics: [metrics.md](metrics.md)
