# `jade.config` — Module Quickstart

> READ THIS FIRST. After reading only this page you should understand the whole `config` module without opening any source file. The per-file pages exist for exhaustive field-level detail.

## Purpose

`jade.config` holds every **configuration object** of the `jade` package (Just Another Deep-learning EEG package). A config is a small, validated, (de)serializable **dataclass** that describes *what to do* (which dataset to download and how to filter it, which optimizer/scheduler/loss/dataloader to build, how to log to wandb, the training hyper-parameters) without doing any of it. The heavy lifting (building torch objects, downloading data, running the training loop) lives in other `jade` modules that **consume** these configs.

The whole module is data + validation only. It has **no runtime side effects** beyond reading/writing files and importing torch class names indirectly (it does not even import `torch`).

## Structure

```
config/
  generic.py          generic_config      -> base class: (de)serialization helpers
  filter.py           filter_config       -> moabb/MNE filtering parameters
  dataset.py          dataset_config      -> dataset download + preprocessing (holds a filter_config)
  dataloader.py       dataloader_config   -> torch.utils.data.DataLoader parameters (NO batch_size)
  wandb_training.py   wandb_training_config -> wandb logging parameters
  training.py         training_config     -> training hyper-params (holds optimizer/scheduler/loss/dataloader/wandb)
  optimizers/         optimizer_config + Adam/AdamW/SGD/RMSprop/LBFGS + registry + get_optimizer_config
  lr_schedulers/      lr_scheduler_config + ExponentialLR/StepLR/CosineAnnealingLR/…WarmRestarts/CyclicLR + registry + factory
  loss_functions/     loss_config + CrossEntropy/NLL + registry + get_loss_config
  model/              model_config + EEGNet_config + the pooling config family + registries + get_model_config/get_pooling_config
  default_config/     commented TOML skeleton configs (package data, copied by the `create-default-config` cli command)
```

Everything is re-exported from `jade.config.__init__`, so the flat public API is `from jade.config import training_config, dataset_config, adam_config, ...`.

## The three big design decisions

### 1. One base class with (de)serialization: `generic_config`
Every config inherits (directly or transitively) from `generic_config` (a fieldless `@dataclass`). It provides six helpers, all recursive over nested configs:

| Method | Direction | Notes |
|---|---|---|
| `to_dict()` / `from_dict(data)` | dict ↔ config | `from_dict` ignores unknown keys (forward/backward compatible files); rebuilds nested configs into the proper subclass |
| `to_toml(path)` / `from_toml(path)` | TOML file ↔ config | nested config → its own `[section]`; `None` fields are dropped on write (TOML has no null) and fall back to defaults on read; needs the optional `toml` package |
| `to_json(path)` / `from_json(path)` | JSON file ↔ config | `indent=4` by default |

Value **validation is per-subclass**, done in `__post_init__` (dataclasses call it automatically), which raises `ValueError` on bad values. A common idiom: `bool` is explicitly rejected wherever an `int` is expected (because `bool` subclasses `int`).

### 2. Registry + `get_*_config` factory for the polymorphic families
Optimizers, schedulers, losses and **models** are **families** of interchangeable configs selected by a string discriminator (`name`; the pooling family uses `pooling_type`). Each subpackage defines:

- a base class (`optimizer_config` / `lr_scheduler_config` / `loss_config`) carrying only `name`, and overriding `from_dict` to **dispatch polymorphically**: when called on the base class it reads `name`, looks it up in the registry, and delegates to the concrete subclass; when called on a concrete subclass it behaves normally.
- a `*_CONFIG_REGISTRY` dict mapping lowercase `name` → config class.
- a `get_*_config(config)` factory that returns a config untouched if already built, promotes a `dict` via the base `from_dict`, and raises `TypeError` otherwise.

Concrete subclasses **freeze** their `name` in `__post_init__`, so a config always round-trips as the right kind. This mirrors the `MODEL_REGISTRY` / `get_model` convention used elsewhere in `jade`.

### 3. Nested configs compose, and dicts are promoted
Configs hold other configs as fields:

- `dataset_config.filter` → a `filter_config`
- `training_config.{optimizer, lr_scheduler, loss, dataloader, wandb}` → the respective configs

(The **model** config is not nested in `training_config`: `jade.training.train` takes `model_config` as a separate argument, so a model config lives on its own.)

When a nested field arrives as a plain `dict` (e.g. from a TOML/JSON file), the parent's `__post_init__` **promotes** it to the proper dataclass (`filter_config.from_dict`, `get_optimizer_config`, etc.). This means a fully nested config tree survives a file round-trip and is always made of real dataclass instances, not dicts.

## End-to-end flow

```
TOML/JSON file ──from_toml/from_json──> training_config (nested tree rebuilt & validated)
                                             │
        ┌────────────────────────────────────┼───────────────────────────────┐
   optimizer_config   lr_scheduler_config   loss_config   dataloader_config   wandb_training_config
        │                    │                  │               │                     │
 get_optimizer()      get_lr_scheduler()   get_loss_function() get_dataloader()   wandb_train()/train()
 (support_training)                                                              (classifier)
        ▼                    ▼                  ▼               ▼
 torch.optim.*      torch.optim.lr_scheduler.*  torch.nn.*   torch.utils.data.DataLoader
```

Configs are pure descriptions; the `jade.training.support_training` factory functions turn them into torch objects (dispatching on the `name` field), and `jade.training.classifier` runs the loop.

## Key relationships between the configs

- `training_config` is the **root** for a training run. `batch_size` lives here (not in `dataloader_config`) because it is shared by every loader. Defaults: `optimizer = adam_config()`, `loss = cross_entropy_config()`, `dataloader = dataloader_config()`, `lr_scheduler = None`, `wandb = None`.
- `dataset_config` is the **root** for data preparation; its default `filter` is an 8–32 Hz band-pass (motor-imagery style).
- **Loss ↔ model coupling** (important gotcha): `cross_entropy_config` expects logits → pair with a model built `use_softmax = False`; `nll_config` expects log-probabilities → pair with `use_softmax = True`. Choosing the wrong one applies `log_softmax` twice or not at all.

## Dependencies

- **Internal**: everything is rooted at `generic_config`; the family bases import their registries lazily (inside `from_dict`) to avoid circular imports with their package `__init__`. `training_config` imports the optimizer/scheduler/loss factories; `dataset_config` imports `filter_config`.
- **External**: standard library `dataclasses`, `json`, `typing`, `os`, `logging`; the optional `toml` package (only needed for TOML I/O; a clear `ImportError` is raised if missing). Note: this module does **not** import `torch`, `moabb`, `mne` or `wandb` — it only mirrors their parameter names. Those are pulled in by the consumers.

## How it connects to the rest of `jade`

| Config | Consumed by | Produces / drives |
|---|---|---|
| `dataset_config` (+`filter_config`) | `jade.dataset` (moabb download, MNE filtering, `EEG_Dataset`) | preprocessed EEG dataset |
| `dataloader_config` | `jade.training.support_training.get_dataloader` | `torch.utils.data.DataLoader` |
| `optimizer_config` family | `…support_training.get_optimizer` | `torch.optim.*` |
| `lr_scheduler_config` family | `…support_training.get_lr_scheduler` | `torch.optim.lr_scheduler.*` |
| `loss_config` family | `…support_training.get_loss_function` | `torch.nn.*` (weights list → tensor) |
| `wandb_training_config` | `jade.training.classifier.wandb_train` / `train` | wandb run + model artifact |
| `training_config` | `jade.training.classifier` | the whole training run |
| `model_config` family (`EEGNet_config`) | `jade.model.get_model`; `jade.training.train` | the `torch.nn.Module` to train |
| pooling config family | `jade.model.pooling_1d.get_pooling_layer` | a 1D pooling layer (research subpackage) |

Model configuration now lives in the [`model/` subpackage](model/0_module_quickstart.md): `EEGNet_config` plus the 1D pooling config family, with the same registry + `get_*_config` pattern. `jade.model.get_model` and the `EEGNet` constructors consume these configs directly.
