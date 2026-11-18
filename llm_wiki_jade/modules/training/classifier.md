# `jade.training.classifier`

Source: `src/jade/training/classifier.py`

The full classification training cycle, as **two mirrored pairs**: `train` / `wandb_train` build the model from a `model_config`, while `train_model` / `wandb_train_model` train an already-built model.

> The **generative** counterpart of this module is [`vae.py`](vae.md), which offers the same four entry points with the same signatures.

- `train` is the standard entry point: it cross-checks the training/model configs, builds the model from a `model_config` with `jade.model.get_model`, and delegates to `train_model`.
- `train_model` runs the entire loop on an already-built model (dataloaders, optimizer/scheduler/loss creation, epoch loop, checkpointing, optional metrics, optional wandb logging) and returns the trained model plus the per-epoch metrics.
- `wandb_train` is the wandb counterpart of `train` (it cross-checks, builds the model, delegates to `wandb_train_model`).
- `wandb_train_model` is the wandb counterpart of `train_model`: after the config checks it delegates the whole run to [`wandb_utils.run_training_with_wandb`](wandb_utils.md), passing its own `train_model` as the function to call inside the run. Its signature and return value are unchanged, which matters because `training/EEGNet/train_EEGNet.py` and `train_pooling.py` call it with six positional arguments.

**Config contract:** all four functions accept only config **dataclass instances** (no dict promotion), consistently with the rest of the codebase — a wrong type raises `TypeError` via the shared `_require_config_instance` helper.

## Module-level facts

- `wandb` is **not** imported here any more: every wandb call goes through [`wandb_utils`](wandb_utils.md), which owns the optional import.
- The config classes are imported under aliases for the `isinstance` checks: `training_config` → `training_config_class`, `model_config` (base) → `model_config_class`, `dataset_config` → `dataset_config_class`.
- `get_model` is imported from `jade.model` (used by `train`/`wandb_train` to build the model from its config).
- `_require_config_instance(value, expected_class, name)` — now a one-line **alias** for [`support_training.require_config_instance`](support_training.md), which moved there because the generative cycle needs the identical check. The private name is kept so anything that reached for it still resolves. Raises `TypeError` unless `value` is an instance of `expected_class`; used by every entry point to reject dicts/other types.
- Module logger: `logger = logging.getLogger(__name__)`.

---

## `train(training_config, model_config, train_dataset, validation_dataset=None, wandb_model_artifact=None)`

The standard entry point. It builds the model from `model_config` and delegates the actual training to `train_model`. Returns exactly what `train_model` returns.

### Parameters

| Name | Type | Default | Meaning |
|---|---|---|---|
| `training_config` | `training_config` | — | Full run configuration (a dataclass instance; **no dict**). |
| `model_config` | `model_config` | — | Model configuration instance (e.g. `EEGNet_config`). The model is built from it. |
| `train_dataset` | `torch.utils.data.Dataset` | — | Training data; each item is a `(data, label)` pair. |
| `validation_dataset` | `torch.utils.data.Dataset` | `None` | Validation data (forwarded to `train_model`). |
| `wandb_model_artifact` | `wandb.Artifact` | `None` | Artifact for checkpoints (forwarded to `train_model`; normally created by `wandb_train_model`). |

### Flow

1. `_require_config_instance` on `training_config` and `model_config` — `TypeError` on a dict/wrong type (dataclass-only).
2. `support_training.cross_check_config(training_config, model_config)` — fail fast on a training ↔ model mismatch (e.g. loss vs `use_softmax`).
3. `model = get_model(model_config)` — the config `name` selects the class.
4. `return train_model(training_config, model, train_dataset, validation_dataset, wandb_model_artifact)` — every argument forwarded.

---

## `train_model(training_config, model, train_dataset, validation_dataset=None, wandb_model_artifact=None)`

Trains an **already-built** classifier and returns it together with the metrics collected during training. This is the former `train` body (renamed); `train` and `wandb_train` both delegate here.

### Parameters

| Name | Type | Default | Meaning |
|---|---|---|---|
| `training_config` | `training_config` | — | Full run configuration (a dataclass instance; **no dict** — `TypeError` otherwise). Holds every hyper-parameter and the nested optimizer/scheduler/loss/dataloader/wandb configs. |
| `model` | `torch.nn.Module` | — | Model to train. For metrics it should expose a `classify` method (as `jade.model.EEGNet.EEGNet_Classifier` does). |
| `train_dataset` | `torch.utils.data.Dataset` | — | Training data; each item is a `(data, label)` pair. |
| `validation_dataset` | `torch.utils.data.Dataset` | `None` | Validation data. If `None`, no validation is performed and **no `model_BEST.pth`** is saved. |
| `wandb_model_artifact` | `wandb.Artifact` | `None` | Artifact that checkpoints are added to. Normally created and passed by `wandb_train_model`. |

### Returns

`(model, computed_metrics_during_training)`:

- `model : torch.nn.Module` — the trained model (mutated in place; also returned for convenience).
- `computed_metrics_during_training : dict` — maps each tracked quantity to the **list of its per-epoch values**. Keys: `"train_loss"`, `"validation_loss"` (only if validating), and, when `measure_metrics_during_training` is on, `"{metric}_train"` and `"{metric}_validation"` for each metric in the metrics dict (`accuracy`, `cohen_kappa`, `sensitivity`, `specificity`, `f1`).

### Flow, step by step

1. **Config check & setup** — `_require_config_instance(training_config, training_config_class, ...)` (`TypeError` on a dict/wrong type); alias `device = training_config.device`, `use_wandb = training_config.wandb is not None`. If `use_wandb` but `wandb is None`, raise `ImportError`. If `training_config.seed is not None`, call `support_training.set_seed`.
2. **Dataloaders** — `support_training.get_dataloader(train_dataset, batch_size, dataloader)`; same for validation if present, else `validation_loader = None`.
3. **Loss / optimizer / scheduler** — `model.to(device)`; optionally load `path_past_weights` into the model (`torch.load(..., map_location=device)` + `load_state_dict`); build loss via `get_loss_function` and move to device (matters when the loss carries class weights); build optimizer via `get_optimizer`; optionally load `path_past_optimizer` state; build LR scheduler via `get_lr_scheduler` if `lr_scheduler` config is set, else `None`.
4. **Training preparation** — `os.makedirs(path_to_save_model, exist_ok=True)`; if `use_wandb`, `wandb_utils.watch_model(model, loss_function, training_config.wandb)` (which itself calls `wandb.watch(model, criterion=loss_function, log="all", log_freq=wandb_config.log_freq, log_graph=True)`); `best_loss_val = sys.maxsize`; init `log_dict = {}` and `computed_metrics_during_training = dict()`.
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

## `wandb_train(training_config, model_config, train_dataset, validation_dataset=None, dataset_config=None)`

The wandb counterpart of `train`: it cross-checks the configs, builds the model from `model_config` (`get_model`), and delegates to `wandb_train_model` (which opens the run). Use it for a wandb-tracked run when you have configs (not a built model).

### Parameters

| Name | Type | Default | Meaning |
|---|---|---|---|
| `training_config` | `training_config` | — | Config instance; its `wandb` field (a `wandb_training_config`) **must** be set. |
| `model_config` | `model_config` | — | Model configuration instance. The model is built from it, and it is logged to wandb for reproducibility. |
| `train_dataset` | `torch.utils.data.Dataset` | — | Training data. |
| `validation_dataset` | `torch.utils.data.Dataset` | `None` | Validation data. |
| `dataset_config` | `dataset_config` | `None` | Same logic as `model_config`: **not used during training**, only logged to wandb for reproducibility. |

### Flow

1. `_require_config_instance` on `training_config`, `model_config` (and `dataset_config` when given) — `TypeError` on a dict/wrong type.
2. `support_training.cross_check_config(training_config, model_config)`.
3. `model = get_model(model_config)`.
4. `return wandb_train_model(training_config, model, train_dataset, validation_dataset, model_config=model_config, dataset_config=dataset_config)` — forwards both configs so they are backed up on wandb.

---

## `wandb_train_model(training_config, model, train_dataset, validation_dataset=None, model_config=None, dataset_config=None)`

The wandb counterpart of `train_model` (already-built model). Opens a `wandb.init` run (and optional model `wandb.Artifact`), then calls `train_model` inside it. See this page's `train_model` for the training details.

### Parameters

| Name | Type | Default | Meaning |
|---|---|---|---|
| `training_config` | `training_config` | — | Config instance; its `wandb` field (a `wandb_training_config`) **must** be set. |
| `model` | `torch.nn.Module` | — | Model to train (already built). |
| `train_dataset` | `torch.utils.data.Dataset` | — | Training data. |
| `validation_dataset` | `torch.utils.data.Dataset` | `None` | Validation data. |
| `model_config` | `model_config` | `None` | **Not used during training** (an already-built `model` is passed) — only logged to wandb (with the training config) for reproducibility, via `to_dict()`. |
| `dataset_config` | `dataset_config` | `None` | Same logic as `model_config`: not used during training, only logged to wandb for reproducibility. |

### Returns

`(model, computed_metrics_during_training)` — exactly what `train_model` returns.

### Flow

1. `_require_config_instance` on `training_config` (and `model_config`/`dataset_config` when given) — `TypeError` on a dict/wrong type.
2. Delegates the entire run to [`wandb_utils.run_training_with_wandb(train_model, training_config, model, train_dataset, validation_dataset, model_config=model_config, dataset_config=dataset_config)`](wandb_utils.md), passing its own `train_model` as the training function to call. All wandb mechanics now live there:
   - `wandb_utils.wandb_run` (a context manager) calls `require_wandb()` (`ImportError` if wandb isn't installed) and checks `training_config.wandb is not None` (`ValueError` otherwise).
   - Builds `config_to_log` via `build_config_to_log` — `{"training_config": training_config.to_dict()}`, plus `"model_config"`/`"dataset_config"` (each `to_dict`-ed) when provided.
   - `run_id = build_run_id(training_config)` — `f"{seed}_{name_training_run}"` when **both** `training_config.seed` and `wandb_config.name_training_run` are set, else `None` (wandb assigns its own id).
   - Opens `wandb.init(project=..., entity=..., name=..., job_type="train", config=config_to_log, notes=..., id=run_id, resume="allow")`.
   - `create_model_artifact` builds the artifact named `wandb.model_artifact_name + "_trained"` when `log_model_artifact` is true, else `None`.
3. Inside the run, `run_training_with_wandb` calls `train_model(training_config, model, train_dataset, validation_dataset, wandb_model_artifact)`.
4. On a clean exit from the run context, `run.log_artifact(wandb_model_artifact)` if an artifact was created.
5. Returns `train_model`'s result.

### Side effects / errors

- Opens (and closes) a wandb run; logs config, watches the model, logs metrics/artifacts (through `train_model`).
- `TypeError` (non-dataclass config), `ImportError` (wandb missing), `ValueError` (`wandb` config unset). Any error inside the `with` still lets the run context close.

### Related pages

- Single-epoch logic: [classifier_epoch.md](classifier_epoch.md)
- Factories/helpers: [support_training.md](support_training.md)
- Metrics: [metrics.md](metrics.md)
