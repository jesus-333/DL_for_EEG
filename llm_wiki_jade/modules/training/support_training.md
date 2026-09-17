# `jade.training.support_training`

Source: `src/jade/training/support_training.py`

Shared code used by more than one training file: the **config → torch object factories**, the **training helpers**, and a few optional **memory-profiling utilities**. This is the single place where a (deliberately torch-free, serializable) config is materialized into a real `torch` object.

Module logger: `logger = logging.getLogger(__name__)`. External deps: `torch`, `numpy`; stdlib `gc`, `os`, `time`, `threading.Thread`; `psutil` imported lazily inside `cpu_memory_usage_in_gb`. Internal config imports: `training_config` (as `training_config_class`), `dataloader_config`, `get_optimizer_config`, `get_lr_scheduler_config`, `get_loss_config`, and `get_model_config` (the last two used by `cross_check_config`).

---

## Factories (config → torch)

Each factory accepts either a built config instance **or a plain dict** (promoted through the config's own `get_*_config` / `from_dict`), then dispatches on the config's `name` field. All four raise a `ValueError` on an unknown name, though the docstrings note this "should never happen" because the configs validate `name` themselves.

### `get_dataloader(dataset, batch_size, dataloader_config_to_use=None) -> torch.utils.data.DataLoader`

| Param | Type | Default | Meaning |
|---|---|---|---|
| `dataset` | `torch.utils.data.Dataset` | — | Dataset to wrap. |
| `batch_size` | `int` | — | Mini-batch size (comes from `training_config`, not from the dataloader config). |
| `dataloader_config_to_use` | `dataloader_config` / `dict` / `None` | `None` | `None` → a default `dataloader_config()`; a dict → promoted via `dataloader_config.from_dict`. |

Builds `torch.utils.data.DataLoader(dataset, batch_size=..., shuffle=..., num_workers=..., prefetch_factor=..., pin_memory=..., persistent_workers=...)`, reading every field except `batch_size` from the config. Note `batch_size` is intentionally **not** in `dataloader_config` — it lives on `training_config` because it is shared by every loader.

### `get_optimizer(optimizer_config_to_use, model) -> torch.optim.Optimizer`

Dispatches on `config.name` to a `torch.optim` class:

| `name` | torch class | Forwarded config fields |
|---|---|---|
| `adam` | `torch.optim.Adam` | `lr`, `betas` (→ tuple), `eps`, `weight_decay`, `decoupled_weight_decay`, `amsgrad`, `maximize` |
| `adamw` | `torch.optim.AdamW` | `lr`, `betas` (→ tuple), `eps`, `weight_decay`, `amsgrad`, `maximize` |
| `sgd` | `torch.optim.SGD` | `lr`, `momentum`, `dampening`, `weight_decay`, `nesterov`, `maximize` |
| `rmsprop` | `torch.optim.RMSprop` | `lr`, `alpha`, `eps`, `weight_decay`, `momentum`, `centered` |
| `lbfgs` | `torch.optim.LBFGS` | `lr`, `max_iter`, `max_eval`, `tolerance_grad`, `tolerance_change`, `history_size`, `line_search_fn` |

Optimizes `model.parameters()`. `betas` is stored as a list in the config and wrapped in `tuple(...)` here.

### `get_lr_scheduler(lr_scheduler_config_to_use, optimizer)`

Returns a `torch.optim.lr_scheduler.*`. Called by `train` only when a scheduler config is present; the scheduler is stepped once per epoch by the training loop.

| `name` | torch class | Forwarded config fields |
|---|---|---|
| `exponential_lr` | `ExponentialLR` | `gamma` |
| `step_lr` | `StepLR` | `step_size`, `gamma` |
| `cosine_annealing_lr` | `CosineAnnealingLR` | `T_max`, `eta_min` |
| `cosine_annealing_warm_restarts` | `CosineAnnealingWarmRestarts` | `T_0`, `T_mult`, `eta_min` |
| `cyclic_lr` | `CyclicLR` | `base_lr`, `max_lr`, `step_size_up`, `step_size_down`, `mode`, `gamma`, `cycle_momentum` |

### `get_loss_function(loss_config_to_use) -> torch.nn.Module`

| `name` | torch class | Forwarded config fields |
|---|---|---|
| `cross_entropy` | `torch.nn.CrossEntropyLoss` | `weight`, `ignore_index`, `reduction`, `label_smoothing` |
| `nll` | `torch.nn.NLLLoss` | `weight`, `ignore_index`, `reduction` |
| anything else (currently `elbo`) | the matching class of [`custom_loss_function`](custom_loss_function/0_module_quickstart.md) | the whole config |

Class weights are stored as a plain list in the config and converted here: `weight = torch.tensor(config.weight, dtype=torch.float) if config.weight is not None else None`. Reminder embedded in the docstring: `CrossEntropyLoss` expects **logits**, `NLLLoss` expects **log-probabilities** — the caller (`train_model`) moves the loss to the device after building it, which matters precisely because `weight` is a tensor.

> **Gotcha fixed with the ELBO.** That conversion used to run *before* the dispatch, so it read `config.weight` for every loss. An `elbo_config` has no `weight` field (a per-class rescaling is meaningless for a reconstruction objective), so it would have raised an `AttributeError` before reaching any branch. The conversion now lives **inside** the classification branches, guarded by `config.name in CLASSIFICATION_LOSSES`.

The custom-loss branch does a **function-local import** of `jade.training.custom_loss_function` — both to keep this module's import light and to avoid a cycle, since that subpackage lives under `jade.training`, whose `__init__` imports this very module. It is the same idiom used by `loss_config.from_dict`, `get_pooling_layer` and `cpu_memory_usage_in_gb`. An unknown name still raises the original `"unknown loss name ..."`.

---

## Cross checks between configs

### `cross_check_config(training_config, model_config) -> None`

The single place where constraints spanning **two** configs are verified. `jade.training.classifier.train` calls it first, so a mismatch fails fast (with a clear `ValueError`) before any model is built or trained. It is not torch-dependent — it only inspects config fields.

| Param | Type | Meaning |
|---|---|---|
| `training_config` | `training_config` or `dict` | Promoted via `training_config_class.from_dict` if a dict. |
| `model_config` | `model_config` or `dict` | Promoted via `get_model_config` if a dict. |

Checks enforced (extend here for future constraints):

1. **Loss ↔ last-layer activation.** `loss.name == "nll"` requires `model_config.use_softmax == True` (the classifier ends in `LogSoftmax`, emitting log-probabilities); `loss.name == "cross_entropy"` requires `use_softmax == False` (the classifier ends in `Identity`, emitting raw logits). Read via `getattr(model_config, "use_softmax", None)`, so a model without that field (e.g. a pure feature extractor, or a generative model) skips the check.
2. **Class weights ↔ number of classes.** If `loss.weight` is set, `len(weight)` must equal `model_config.n_classes`.
3. **Training objective.** A model with `task == "generative"` (see [`model_config`](../config/model/generic.md)) returns a reconstruction and the parameters of a latent distribution, which a loss in `CLASSIFICATION_LOSSES` cannot consume; conversely a loss in `GENERATIVE_LOSSES` needs those parameters, which a classifier never produces. Both directions raise, with a message naming the two sides.

Raises `ValueError` on any violation; returns `None` otherwise. It also **warns** (it does not raise) when `loss.kl_warmup_epochs > training_config.epochs`: a truncated run is legitimate, but the KL term then never reaches its full weight.

### Why a `task` marker and not a name list

Two tempting alternatives were rejected. *Inferring the objective from the absence of `n_classes`* would falsely reject `braindecode_wrapper_config`, whose hyper-parameters travel as a raw `model_kwargs` dict. *Hard-coding a list of generative model names here* would put model knowledge in the training module and would have to be edited for every future generative model — exactly the coupling the registries exist to avoid. The `ClassVar` on the config keeps the fact where the model is described, at no serialization cost.

### `require_config_instance(value, expected_class, name) -> None`

Raises `TypeError` unless `value` is an instance of `expected_class`. The training entry points accept only config *instances*, so a wrong type fails fast with a clear message. It lives here (rather than in one training module) because **every** cycle needs it; `classifier._require_config_instance` is kept as an alias for backward compatibility.

## Module constants

`CLASSIFICATION_LOSSES = ("cross_entropy", "nll")` and `GENERATIVE_LOSSES = ("elbo",)` — the two loss families, used by both `get_loss_function` and `cross_check_config`.

---

## Training helpers

### `backup_every_epoch(path_to_save, epoch, model, optimizer=None, lr_scheduler=None) -> None`

Writes an alternating A/B backup so a run interrupted mid-write still leaves one intact file ("backup of the backup"), useful for long HPC runs split across jobs.

| Param | Type | Default | Meaning |
|---|---|---|---|
| `path_to_save` | `str` | — | Folder for the backup files. |
| `epoch` | `int` | — | Current epoch; its **parity** picks the slot: even → `_backup_A`, odd → `_backup_B`. |
| `model` | `torch.nn.Module` | — | Model to back up (always saved). |
| `optimizer` | `torch.optim.Optimizer` | `None` | Saved only if provided. |
| `lr_scheduler` | scheduler | `None` | Saved only if provided. |

Writes `model{suffix}.pth` and, when provided, `optimizer{suffix}.pth` / `lr_scheduler{suffix}.pth` (all `state_dict`s via `torch.save`). Files are **overwritten** each time the same slot is hit (only the last two epochs survive). Returns `None`.

### `update_log_dict_metrics(metrics_dict, log_dict, label=None) -> None`

Copies each `key: value` from `metrics_dict` into `log_dict` **in place**. With `label` set, the key becomes `"{key}_{label}"` (e.g. `accuracy` + `train` → `"accuracy_train"`); with `label=None` the key is copied unchanged. This is the convention behind the wandb `log_dict` keys and the `computed_metrics_during_training` metric keys — always a suffix (`_train` / `_validation` / `_test`). Returns `None`.

### `set_seed(seed) -> None`

Seeds `numpy` (`np.random.seed`), torch CPU (`torch.manual_seed`), and, when CUDA is available, all GPUs (`torch.cuda.manual_seed_all`). Called by `train` when `training_config.seed is not None`.

---

## Optional memory-profiling utilities

Copied from an external blog post (analyticsvidhya); not part of the training loop, handy for profiling.

| Function | Signature | Behavior |
|---|---|---|
| `cpu_memory_usage_in_gb` | `(func, *args, **kwargs) -> float` | Runs `func(*args, **kwargs)` in the main thread while a background `Thread` polls RSS every 0.1 s (via lazily-imported `psutil`); returns peak RSS in GB **above baseline**. |
| `gpu_start_memory_tracking` | `() -> None` | `torch.cuda.reset_peak_memory_stats()` if CUDA available, else logs a WARNING. |
| `gpu_print_memory_usage` | `() -> None` | Logs `torch.cuda.max_memory_allocated()` in GB at INFO. |
| `gpu_cleanup` | `() -> None` | `gc.collect()`, `torch.cuda.empty_cache()`, `time.sleep(3)`, reset peak stats, log max allocated. |

> Gotcha: `gpu_print_memory_usage` and `gpu_cleanup` call CUDA APIs unconditionally and will error if CUDA is unavailable; only `gpu_start_memory_tracking` guards against it.

---

## The `log_dict` convention (summary)

`log_dict` is a single flat dict passed around during a run and logged to wandb per epoch. Keys seen across the module:

- `"train_loss"`, `"validation_loss"` — set by the epoch functions ([classifier_epoch](classifier_epoch.md)).
- `"learning_rate"` — set by `train` before each scheduler step.
- `"{metric}_train"`, `"{metric}_validation"` — set by `update_log_dict_metrics`.
- `"{label}_loss"`, `"{metric}_{label}"` — set by [`test`](test_functions.md) (which renames `"validation_loss"`).
