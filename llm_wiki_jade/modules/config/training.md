# `config/training.py`

Source: `src/jade/config/training.py`

Defines `training_config`, the **root** config for a training run — every parameter consumed by `jade.training.classifier` **or** `jade.training.vae` (both accept the same config). It follows the module's standard design (dataclass + `generic_config` + `__post_init__` validation) and, crucially, holds five **nested configs** that are promoted from plain dicts on load. Serialized to TOML, each nested config becomes its own section (`[optimizer]`, `[loss]`, `[wandb]`, …).

## Imports of note

Pulls in the nested config types and their factories: `dataloader_config`, `wandb_training_config`, `optimizer_config`/`adam_config`/`get_optimizer_config`, `lr_scheduler_config`/`get_lr_scheduler_config`, `loss_config`/`cross_entropy_config`/`get_loss_config`.

## Class `training_config(generic_config)`

`@dataclass`. Fields grouped by the source's own thematic blocks.

### Core hyper-parameters

| Field | Type | Default | Meaning |
|---|---|---|---|
| `epochs` | `int` | `100` | Training epochs. Must be positive. |
| `batch_size` | `int` | `32` | Mini-batch size passed to every dataloader. Must be positive. |
| `device` | `str` | `"cpu"` | Device (`"cpu"`, `"cuda"`, `"cuda:0"`, …). Must be a non-empty string. |

### Nested configs

| Field | Type | Default | Meaning |
|---|---|---|---|
| `optimizer` | `optimizer_config` | `adam_config()` | Optimizer config (polymorphic; dict promoted via `get_optimizer_config`). |
| `lr_scheduler` | `lr_scheduler_config \| None` | `None` | Scheduler config; `None` disables scheduling. Stepped once per epoch. |
| `loss` | `loss_config` | `cross_entropy_config()` | Loss config (polymorphic). Must match the model output — see gotcha. |
| `dataloader` | `dataloader_config \| None` | `None` | Dataloader config (optional). `None` (the default) creates a default `dataloader_config` — torch's default `DataLoader` values (with `shuffle=True`); a dict is promoted. `batch_size` comes from this class, not from here. |
| `wandb` | `wandb_training_config \| None` | `None` | wandb logging config; `None` disables wandb. |

Defaults use `field(default_factory=...)` for the mutable dataclass defaults (`adam_config`, `cross_entropy_config`); `dataloader`, `lr_scheduler` and `wandb` default to plain `None` (for `dataloader`, `None` is turned into a default `dataloader_config()` in `__post_init__`, so it is optional but never actually `None` afterwards).

### Checkpoint / logging options

| Field | Type | Default | Meaning |
|---|---|---|---|
| `path_to_save_model` | `str` | `"model_weights"` | Folder for checkpoints (created if missing). |
| `epoch_to_save_model` | `int` | `0` | Period (epochs) of periodic checkpoints. `<= 0` disables them (see below). |
| `backup_model_every_epoch` | `bool` | `False` | If `True`, an alternating A/B backup is written every epoch (see `support_training.backup_every_epoch`). |
| `measure_metrics_during_training` | `bool` | `True` | Compute classification metrics (accuracy, F1, …) each epoch. |
| `print_var` | `bool` | `True` | Log per-epoch losses/metrics at INFO level. |
| `seed` | `int \| None` | `None` | Optional random seed; if given must be non-negative. Seeds the run (`support_training.set_seed`) and is used as the wandb run id. |

### Optional training resume

| Field | Type | Default | Meaning |
|---|---|---|---|
| `path_past_weights` | `str \| None` | `None` | Path to a `state_dict` to load into the model before training. |
| `path_past_optimizer` | `str \| None` | `None` | Path to a `state_dict` to load into the optimizer before training. |

## `__post_init__` — validation & promotion

1. **Core hyper-parameters**: `epochs` and `batch_size` must be positive ints (`bool` rejected); `device` must be a non-empty string. Else `ValueError`.
2. **Promote nested configs** (a dict from a file becomes the proper dataclass):
   - `optimizer`: always `self.optimizer = get_optimizer_config(self.optimizer)` — polymorphic, concrete subclass chosen from its `name`.
   - `lr_scheduler`: if not `None`, `get_lr_scheduler_config(...)`.
   - `loss`: always `get_loss_config(self.loss)` — polymorphic.
   - `dataloader`: if `None`, create a default `dataloader_config()` (torch's default DataLoader values); if a dict, `dataloader_config.from_dict`; if not a `dataloader_config`, `ValueError`.
   - `wandb`: if not `None`, promote a dict via `wandb_training_config.from_dict`, else require a `wandb_training_config` (else `ValueError`).
3. **`epoch_to_save_model`**: if not a positive int (`bool` rejected), it is silently set to `self.epochs + 1` — a period the epoch loop can never hit, so periodic checkpoints are effectively disabled and only `model_END.pth` (and `model_BEST.pth` when validating) are written. This is an **auto-fix, not an error**.
4. **`seed`**: if not `None`, must be a non-negative int (`bool` rejected), else `ValueError`.

## Gotchas

- **Loss ↔ model coupling**: pair `cross_entropy_config` with a model built `use_softmax=False` (logits); pair `nll_config` with `use_softmax=True` (log-probabilities). Choosing wrong double-applies or omits `log_softmax`. See [loss_functions quickstart](loss_functions/0_module_quickstart.md).
- The optimizer/loss promotion uses the polymorphic factories, so a nested `{"name": "sgd", ...}` dict correctly becomes an `sgd_config` — the `name` discriminator drives the dispatch.
- `epoch_to_save_model=0` (the default) means "no periodic checkpoints", implemented by the `epochs + 1` trick, not by a guard elsewhere.
- Relationship to other pages: [dataloader](dataloader.md) (no `batch_size`), [wandb_training](wandb_training.md), [optimizers](optimizers/0_module_quickstart.md), [lr_schedulers](lr_schedulers/0_module_quickstart.md), [loss_functions](loss_functions/0_module_quickstart.md).
