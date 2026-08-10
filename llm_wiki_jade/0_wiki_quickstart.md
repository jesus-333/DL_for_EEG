# `jade` Wiki — Quickstart (READ THIS FIRST)

> This is the entry point of the `jade` wiki. It exists so that a future session can understand the codebase **without re-reading the source from scratch**. Read this page first, every time. Then read the relevant *module* quickstart. Only open a per-file page (or the actual `.py`) when you need field-level detail for a specific task.

## What is `jade`?

`jade` — **J**ust **A**nother **D**eep-learning **E**EG package — is a personal research library for deep learning on EEG signals. Both the PyPI distribution name and the importable package name are `jade`; the code lives under `src/jade/` (a standard *src-layout*, wired up in `pyproject.toml` via hatchling). The project is at version `0.0.1` and is actively evolving.

The library is organized around a clean separation of concerns: **describe** a run with serializable config objects, **prepare** data, **define** a model, and **run** training — each in its own module.

## How to use this wiki (reading protocol)

1. **Always start here** (`0_wiki_quickstart.md`).
2. Then read the **module quickstart** for the area you're touching: `modules/<module>/0_module_quickstart.md`. That alone should give you a working mental model of the module.
3. Need the list of files in a module? Read its `modules/<module>/0_module_index.md`.
4. **Only if** you need exhaustive detail about a specific file (every function, parameter, gotcha) open `modules/<module>/<file>.md` — and, if still needed, the `.py` itself. These per-file pages are *complementary* to the source, not a replacement.
5. Submodules mirror this structure recursively (e.g. `modules/config/optimizers/0_module_quickstart.md`, `modules/model/pooling_1d/0_module_quickstart.md`).

**Coding style**: all code in `jade` follows the author's conventions documented in [`coding_style_instructions_ENG.md`](coding_style_instructions_ENG.md). Read it before writing or editing any code (dividers, spacing around `:`/`=`, snake_case everywhere, numpydoc docstrings with sphinx cross-references, extensive in-function comments, `z` spelling, etc.).

## Repository layout (what the wiki covers)

```
src/jade/                 <- the package (this wiki documents everything under here)
  __init__.py
  cli/                    <- thin command-line interface: console scripts that wrap existing jade code
  config/                 <- dataclass-based configuration objects (+ optimizers/, lr_schedulers/, loss_functions/, model/ subpackages; default_config/ = commented TOML skeletons shipped as package data)
  dataset/                <- download / preprocess / torch Dataset wrapping (moabb + MNE)
  model/                  <- torch EEG models (EEGNet) + pooling_1d/ anti-aliasing research subpackage
  training/               <- config-driven training loop, metrics, optional wandb
other/antialiased_pooling_theory.pdf   <- background theory for pooling_1d
```

Everything outside `src/jade/` (notably anything under `test/` and secondary experiments) is **out of scope** for this wiki.

## The five modules at a glance

| Module | Purpose | Read | Public API highlights |
|---|---|---|---|
| **cli** | Thin command-line interface: exposes existing `jade` functionality as console scripts (declared in `[project.scripts]`). Entry points do argparse only and delegate to thematic implementation files; adds no new behavior. | [modules/cli/0_module_quickstart.md](modules/cli/0_module_quickstart.md) | `download-dataset-moabb` → `download_dataset_moabb` (wraps `jade.dataset.download_dataset`); `create-default-config` → `default_config` (copies the skeleton TOML tree) |
| **config** | Validated, (de)serializable dataclasses that *describe* a run (dataset, optimizer, scheduler, loss, dataloader, wandb, training, **and now models**). Data + validation only; imports no torch/moabb/mne/wandb. | [modules/config/0_module_quickstart.md](modules/config/0_module_quickstart.md) | `training_config`, `dataset_config`, `filter_config`, `dataloader_config`, `wandb_training_config`; `*_config` families + `get_optimizer_config` / `get_lr_scheduler_config` / `get_loss_config` / `get_model_config` / `get_pooling_config` + registries (incl. `model/`: `EEGNet_config`, the pooling config family) |
| **dataset** | Data management: download public EEG (moabb), preprocess (optional filter + resample) / epoch (MNE), wrap as torch `Dataset`, plus time/freq inspection plots. | [modules/dataset/0_module_quickstart.md](modules/dataset/0_module_quickstart.md) | `EEG_Dataset`, `EEG_Dataset_ChWi`; `download_dataset`, `get_dataset` (lazy moabb/mne) |
| **model** | torch EEG models (EEGNet family) **plus** a self-contained research subpackage on anti-aliased 1D pooling. | [modules/model/0_module_quickstart.md](modules/model/0_module_quickstart.md) | `get_model` / `MODEL_REGISTRY` (now config-instance driven); `EEGNet`, `EEGNet_Classifier` (consume an `EEGNet_config`); `pooling_1d` (`get_pooling_layer`, avg/blur/max/sinc/dwt) |
| **training** | Config-driven orchestration of the training loop (classification only, for now); metrics; optional wandb run + model artifact. | [modules/training/0_module_quickstart.md](modules/training/0_module_quickstart.md) | `train` (builds the model from a `model_config`), `train_model` (trains a prebuilt model), `wandb_train`; `train_epoch_function`, `validation_epoch_function`; `test`; `setup_logging` (opt-in logger setup); `support_training` (incl. `cross_check_config`), `metrics` |

## End-to-end mental model (how a run flows)

```
        ┌──────────────┐        describes           ┌───────────────┐
        │ jade.config  │ ─────────────────────────▶ │ everything     │
        │ (dataclasses)│                            │ below reads it │
        └──────────────┘                            └───────────────┘
                │                                            │
   dataset_config (+ optional filter_config)   training_config             model_config
                │                    (+ optimizer/scheduler/loss/dataloader/wandb)  (e.g. EEGNet_config)
                ▼                                            │                            │
     jade.dataset: download → optional filter/epoch (MNE)   │                            │
                → EEG_Dataset  [trials, 1, channels, time]  │        jade.model.get_model(model_config) ─▶ nn.Module
                │                                            │                            │  (e.g. EEGNet_Classifier)
                └──────────────┐              ┌─────────────┴──────────────┬─────────────┘
                               ▼              ▼                            ▼
                 jade.training.train(training_config, model_config, train_dataset, validation_dataset)
                   • cross_check_config(training_config, model_config) (e.g. loss ↔ use_softmax), then get_model, then train_model
                   • support_training turns config → torch optimizer/scheduler/loss/DataLoader (dispatch on `name`)
                   • loops epochs: train_epoch_function / validation_epoch_function
                   • metrics.compute_metrics (torchmetrics); logging via logging (NullHandler)
                   • optional wandb_train wrapper: wandb run + model Artifact; test() for evaluation
```

The linchpin is **`jade.training.support_training`**: it is the *only* place where config dataclasses become live torch objects (optimizer, LR scheduler, loss, DataLoader), dispatching on each config's `name` discriminator.

## Cross-cutting conventions (true across modules)

- **Registry + `get_*` factory pattern.** Interchangeable implementations are selected by a lowercase string `name`/`pooling_type`: a `*_REGISTRY` dict maps the name to a class, and a `get_*` factory builds it. Used for the runtime models (`MODEL_REGISTRY`/`get_model` — now driven by a `model_config` instance, selecting on `config.name`), pooling layers (`POOLING_REGISTRY`/`get_pooling_layer`), and the config families: optimizer/scheduler/loss **and now model** (`MODEL_CONFIG_REGISTRY`/`get_model_config`) and pooling (`POOLING_CONFIG_REGISTRY`/`get_pooling_config`). Config factories are idempotent (a built config passes through) and dict promotion never mutates the caller's dict.
- **Configs are dataclasses rooted at `generic_config`** with recursive `to_dict`/`from_dict`/`to_toml`/`from_toml`/`to_json`/`from_json`. `from_dict` ignores unknown keys; nested dicts are promoted to the right dataclass on load; validation happens in `__post_init__` (raises `ValueError`). `print(config)`/`str(config)` give a pretty multi-line layout (one field per line, nested configs indented further); `repr(config)` stays the compact single line.
- **Optional heavy deps are imported lazily.** `moabb`/`mne` (the `dataset` extra) are pulled in only when the moabb loader is actually used (PEP 562 `__getattr__`); `wandb`/`psutil`/`scipy` are imported lazily where used. So `import jade.dataset` / `import jade.training` stay light.
- **Logging, not prints.** `training` attaches a `logging.NullHandler` to its package logger (library best practice); each module logs via `logging.getLogger(__name__)`, with INFO gated by a `print_var` flag. Users who don't set up their own logger can call the opt-in `jade.training.setup_logging()` ([`logging_utils`](modules/training/logging_utils.md)) to configure a root logger (console + optional file) in one call.
- **Tensor shape contract.** `EEG_Dataset` yields `[trials, 1, channels, time]` for `Conv2d`-style EEGNet; the channel-wise `EEG_Dataset_ChWi` yields `[(trials·channels), 1, time]` for a `Conv1d` autoencoder.

## Dependencies & install

- Core (`pyproject.toml`): `numpy`, `matplotlib`, `pandas`, `toml`, `torch`, `torchvision`, `torchmetrics`.
- Optional extras: `extra` → `dtw-loss-functions`, `wandb`; `dataset` → `mne`, `moabb`.
- Build backend: `hatchling`; wheel points at `src/jade` (exposes top-level import `jade`).
- Console scripts: declared in `[project.scripts]` and implemented by the `cli` module. Currently `download-dataset-moabb` (→ `jade.cli.cli:download_dataset_moabb`) and `create-default-config` (→ `jade.cli.cli:default_config`). The latter needs the TOML skeletons under `config/default_config/` to ship as package data — guaranteed by a `force-include` in `pyproject.toml`.

## Known WIP / gotchas to keep in mind

- **Models are now config-driven.** `config.model` provides `EEGNet_config` (and the pooling config family); `get_model` takes a `model_config` instance (or a dict promoted via `get_model_config`) and selects the class on `config.name`. The old `get_model({"model_name": ...})` dict key is gone — the discriminator is `name`. `EEGNet`/`EEGNet_Classifier` read the dataclass attributes directly.
- **`train` vs `train_model`.** `train(training_config, model_config, …)` is the standard entry point: it cross-checks the configs, builds the model with `get_model`, then delegates to `train_model(training_config, model, …)` (which trains an already-built model). `wandb_train` also delegates to `train_model`. `cross_check_config` (in `support_training`) enforces the loss ↔ `use_softmax` coupling and the class-weight ↔ `n_classes` match.
- **`aEEGNet.py` is an attention variant and a WIP stub** with a real bug: `channels_attention_head` subclasses `torch.nn.module` (lowercase — should be `nn.Module`). It is *not* about anti-aliasing.
- **`model/support_modules.py` is an empty TODO placeholder.**
- **`pooling_1d` is wired into EEGNet** via the `[pooling]` section of `EEGNet_config` and `support_functions.get_multichannel_pooling` (channel-wise `*_multichannel` wrappers). `pooling_type='default'` keeps the classic `nn.AvgPool2d`. The subpackage is still standalone and self-testable (`python -m jade.model.pooling_1d`).
- **`batch_size` lives in `training_config`, not `dataloader_config`** (it's shared by all loaders).
- **Loss ↔ model softmax coupling**: `cross_entropy_config` expects logits (`use_softmax=False`); `nll_config` expects log-probs (`use_softmax=True`). Mismatch applies `log_softmax` twice or not at all.
- **Only the classification (finetuning) objective is implemented** in `training`; reconstruction/pretraining and a federated-learning submodule are anticipated in docstrings but absent.

## Full wiki index

See [`0_wiki_index.md`](0_wiki_index.md) for the complete file tree of this wiki with links.
