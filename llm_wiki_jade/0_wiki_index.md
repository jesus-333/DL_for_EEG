# `jade` Wiki — Complete Index

Complete map of the `jade` wiki. **Reading protocol**: start with [`0_wiki_quickstart.md`](0_wiki_quickstart.md), then the relevant module quickstart, then a per-file page only if you need field-level detail. See the quickstart for the rationale.

Every module folder has the same shape:
- `0_module_quickstart.md` — read-first overview of the module (always read this before the per-file pages).
- `0_module_index.md` — the module's own table of contents.
- `<file>.md` — one exhaustive page per `.py` file (including `__init__.py` → `__init__.md`), complementary to the source.

Submodules recurse with the same three-part structure.

## Top level

| File | Purpose |
|---|---|
| [0_wiki_quickstart.md](0_wiki_quickstart.md) | **Entry point.** Whole-repo overview, reading protocol, end-to-end flow, cross-cutting conventions, known WIP/gotchas. |
| [0_wiki_index.md](0_wiki_index.md) | This page — the complete wiki index. |
| [coding_style_instructions_ENG.md](coding_style_instructions_ENG.md) | The author's coding conventions (dividers, spacing, snake_case, numpydoc/sphinx docstrings, …). Read before editing code. |

> The package root `src/jade/__init__.py` is intentionally empty (0 bytes) — it only marks `jade` as a package; there is nothing to document, so it has no dedicated wiki page. Each module's own `__init__.py` (which *does* carry the public API) is documented in that module's `__init__.md`.

## Module: `config` — `modules/config/`

Dataclass-based configuration objects (data + validation only; imports no torch/moabb/mne/wandb).

| Page | Source `.py` |
|---|---|
| [0_module_quickstart.md](modules/config/0_module_quickstart.md) | — |
| [0_module_index.md](modules/config/0_module_index.md) | — |
| [__init__.md](modules/config/__init__.md) | `config/__init__.py` |
| [generic.md](modules/config/generic.md) | `config/generic.py` |
| [filter.md](modules/config/filter.md) | `config/filter.py` |
| [dataset.md](modules/config/dataset.md) | `config/dataset.py` |
| [dataloader.md](modules/config/dataloader.md) | `config/dataloader.py` |
| [wandb_training.md](modules/config/wandb_training.md) | `config/wandb_training.py` |
| [training.md](modules/config/training.md) | `config/training.py` |

### Submodule: `config/loss_functions` — `modules/config/loss_functions/`

| Page | Source `.py` |
|---|---|
| [0_module_quickstart.md](modules/config/loss_functions/0_module_quickstart.md) | — |
| [0_module_index.md](modules/config/loss_functions/0_module_index.md) | — |
| [__init__.md](modules/config/loss_functions/__init__.md) | `config/loss_functions/__init__.py` |
| [generic.md](modules/config/loss_functions/generic.md) | `config/loss_functions/generic.py` |
| [cross_entropy.md](modules/config/loss_functions/cross_entropy.md) | `config/loss_functions/cross_entropy.py` |
| [nll.md](modules/config/loss_functions/nll.md) | `config/loss_functions/nll.py` |
| [elbo.md](modules/config/loss_functions/elbo.md) | `config/loss_functions/elbo.py` |

### Submodule: `config/lr_schedulers` — `modules/config/lr_schedulers/`

| Page | Source `.py` |
|---|---|
| [0_module_quickstart.md](modules/config/lr_schedulers/0_module_quickstart.md) | — |
| [0_module_index.md](modules/config/lr_schedulers/0_module_index.md) | — |
| [__init__.md](modules/config/lr_schedulers/__init__.md) | `config/lr_schedulers/__init__.py` |
| [generic.md](modules/config/lr_schedulers/generic.md) | `config/lr_schedulers/generic.py` |
| [exponential_lr.md](modules/config/lr_schedulers/exponential_lr.md) | `config/lr_schedulers/exponential_lr.py` |
| [step_lr.md](modules/config/lr_schedulers/step_lr.md) | `config/lr_schedulers/step_lr.py` |
| [cosine_annealing_lr.md](modules/config/lr_schedulers/cosine_annealing_lr.md) | `config/lr_schedulers/cosine_annealing_lr.py` |
| [cosine_annealing_warm_restarts.md](modules/config/lr_schedulers/cosine_annealing_warm_restarts.md) | `config/lr_schedulers/cosine_annealing_warm_restarts.py` |
| [cyclic_lr.md](modules/config/lr_schedulers/cyclic_lr.md) | `config/lr_schedulers/cyclic_lr.py` |

### Submodule: `config/optimizers` — `modules/config/optimizers/`

| Page | Source `.py` |
|---|---|
| [0_module_quickstart.md](modules/config/optimizers/0_module_quickstart.md) | — |
| [0_module_index.md](modules/config/optimizers/0_module_index.md) | — |
| [__init__.md](modules/config/optimizers/__init__.md) | `config/optimizers/__init__.py` |
| [generic.md](modules/config/optimizers/generic.md) | `config/optimizers/generic.py` |
| [adam.md](modules/config/optimizers/adam.md) | `config/optimizers/adam.py` |
| [adamw.md](modules/config/optimizers/adamw.md) | `config/optimizers/adamw.py` |
| [sgd.md](modules/config/optimizers/sgd.md) | `config/optimizers/sgd.py` |
| [rmsprop.md](modules/config/optimizers/rmsprop.md) | `config/optimizers/rmsprop.py` |
| [lbfgs.md](modules/config/optimizers/lbfgs.md) | `config/optimizers/lbfgs.py` |

### Submodule: `config/model` — `modules/config/model/`

| Page | Source `.py` |
|---|---|
| [0_module_quickstart.md](modules/config/model/0_module_quickstart.md) | — |
| [0_module_index.md](modules/config/model/0_module_index.md) | — |
| [__init__.md](modules/config/model/__init__.md) | `config/model/__init__.py` |
| [generic.md](modules/config/model/generic.md) | `config/model/generic.py` |
| [EEGNet_cfg.md](modules/config/model/EEGNet_cfg.md) | `config/model/EEGNet_cfg.py` |
| [EEGConformer_cfg.md](modules/config/model/EEGConformer_cfg.md) | `config/model/EEGConformer_cfg.py` |
| [ShallowFBCSPNet_cfg.md](modules/config/model/ShallowFBCSPNet_cfg.md) | `config/model/ShallowFBCSPNet_cfg.py` |
| [Deep4Net_cfg.md](modules/config/model/Deep4Net_cfg.md) | `config/model/Deep4Net_cfg.py` |
| [TIDNet_cfg.md](modules/config/model/TIDNet_cfg.md) | `config/model/TIDNet_cfg.py` |
| [vEEGNet_cfg.md](modules/config/model/vEEGNet_cfg.md) | `config/model/vEEGNet_cfg.py` |
| [braindecode_pooling_extension_cfg.md](modules/config/model/braindecode_pooling_extension_cfg.md) | `config/model/braindecode_pooling_extension_cfg.py` |
| [pooling_1d_cfg.md](modules/config/model/pooling_1d_cfg.md) | `config/model/pooling_1d_cfg.py` |

> The commented TOML skeleton configs live under `config/default_config/` (shipped as package data, copied by the `create-default-config` command); see [cli/default_config.md](modules/cli/default_config.md).

## Module: `cli` — `modules/cli/`

Thin command-line interface: exposes existing `jade` functionality as console scripts (declared in `[project.scripts]`). Entry points in `cli.py` do argparse only and delegate to thematic implementation files; the cli adds no new behavior.

| Page | Source `.py` |
|---|---|
| [0_module_quickstart.md](modules/cli/0_module_quickstart.md) | — |
| [0_module_index.md](modules/cli/0_module_index.md) | — |
| [__init__.md](modules/cli/__init__.md) | `cli/__init__.py` |
| [cli.md](modules/cli/cli.md) | `cli/cli.py` |
| [download_dataset.md](modules/cli/download_dataset.md) | `cli/download_dataset.py` |
| [default_config.md](modules/cli/default_config.md) | `cli/default_config.py` |
| [wandb.md](modules/cli/wandb.md) | `cli/wandb.py` |

## Module: `dataset` — `modules/dataset/`

Download / preprocess / torch `Dataset` wrapping (moabb + MNE, lazily imported) plus frequency plots.

| Page | Source `.py` |
|---|---|
| [0_module_quickstart.md](modules/dataset/0_module_quickstart.md) | — |
| [0_module_index.md](modules/dataset/0_module_index.md) | — |
| [__init__.md](modules/dataset/__init__.md) | `dataset/__init__.py` |
| [dataset_time.md](modules/dataset/dataset_time.md) | `dataset/dataset_time.py` |
| [moabb_dataset.md](modules/dataset/moabb_dataset.md) | `dataset/moabb_dataset.py` |
| [support_plot_freq.md](modules/dataset/support_plot_freq.md) | `dataset/support_plot_freq.py` |

## Module: `model` — `modules/model/`

torch EEG models (EEGNet family, EEG Conformer) + the `pooling_1d` anti-aliasing research subpackage.

| Page | Source `.py` |
|---|---|
| [0_module_quickstart.md](modules/model/0_module_quickstart.md) | — |
| [0_module_index.md](modules/model/0_module_index.md) | — |
| [__init__.md](modules/model/__init__.md) | `model/__init__.py` |
| [EEGNet.md](modules/model/EEGNet.md) | `model/EEGNet.py` |
| [EEGConformer.md](modules/model/EEGConformer.md) | `model/EEGConformer.py` |
| [ShallowFBCSPNet.md](modules/model/ShallowFBCSPNet.md) | `model/ShallowFBCSPNet.py` |
| [Deep4Net.md](modules/model/Deep4Net.md) | `model/Deep4Net.py` |
| [TIDNet.md](modules/model/TIDNet.md) | `model/TIDNet.py` |
| [vEEGNet.md](modules/model/vEEGNet.md) | `model/vEEGNet.py` |
| [braindecode_pooling_extension.md](modules/model/braindecode_pooling_extension.md) | `model/braindecode_pooling_extension.py` |
| [aEEGNet.md](modules/model/aEEGNet.md) | `model/aEEGNet.py` |
| [support_functions.md](modules/model/support_functions.md) | `model/support_functions.py` |
| [support_models.md](modules/model/support_models.md) | `model/support_models.py` (empty TODO placeholder) |

> Note: `src/jade/model/README.md` and `other/antialiased_pooling_theory.pdf` provide additional background; their relevant content is folded into the model quickstart.

### Submodule: `model/pooling_1d` — `modules/model/pooling_1d/`

| Page | Source `.py` |
|---|---|
| [0_module_quickstart.md](modules/model/pooling_1d/0_module_quickstart.md) | — |
| [0_module_index.md](modules/model/pooling_1d/0_module_index.md) | — |
| [__init__.md](modules/model/pooling_1d/__init__.md) | `model/pooling_1d/__init__.py` |
| [__main__.md](modules/model/pooling_1d/__main__.md) | `model/pooling_1d/__main__.py` |
| [support_pooling.md](modules/model/pooling_1d/support_pooling.md) | `model/pooling_1d/support_pooling.py` |
| [avg_pool.md](modules/model/pooling_1d/avg_pool.md) | `model/pooling_1d/avg_pool.py` |
| [blur_pool.md](modules/model/pooling_1d/blur_pool.md) | `model/pooling_1d/blur_pool.py` |
| [max_pool.md](modules/model/pooling_1d/max_pool.md) | `model/pooling_1d/max_pool.py` |
| [sinc_pool.md](modules/model/pooling_1d/sinc_pool.md) | `model/pooling_1d/sinc_pool.py` |
| [dwt_pool.md](modules/model/pooling_1d/dwt_pool.md) | `model/pooling_1d/dwt_pool.py` |

## Module: `training` — `modules/training/`

Config-driven training loops — **classification** and **generative** (VAE) — metrics, the custom losses, optional wandb.

| Page | Source `.py` |
|---|---|
| [0_module_quickstart.md](modules/training/0_module_quickstart.md) | — |
| [0_module_index.md](modules/training/0_module_index.md) | — |
| [__init__.md](modules/training/__init__.md) | `training/__init__.py` |
| [classifier.md](modules/training/classifier.md) | `training/classifier.py` |
| [classifier_epoch.md](modules/training/classifier_epoch.md) | `training/classifier_epoch.py` |
| [vae.md](modules/training/vae.md) | `training/vae.py` |
| [vae_epoch.md](modules/training/vae_epoch.md) | `training/vae_epoch.py` |
| [wandb_utils.md](modules/training/wandb_utils.md) | `training/wandb_utils.py` |
| [test_functions.md](modules/training/test_functions.md) | `training/test_functions.py` |
| [support_training.md](modules/training/support_training.md) | `training/support_training.py` |
| [metrics.md](modules/training/metrics.md) | `training/metrics.py` |
| [logging_utils.md](modules/training/logging_utils.md) | `training/logging_utils.py` |

### Submodule: `training/custom_loss_function` — `modules/training/custom_loss_function/`

The losses with no `torch.nn` counterpart (currently the ELBO). Their configs live in `config/loss_functions/`, not here.

| Page | Source `.py` |
|---|---|
| [0_module_quickstart.md](modules/training/custom_loss_function/0_module_quickstart.md) | — |
| [0_module_index.md](modules/training/custom_loss_function/0_module_index.md) | — |
| [__init__.md](modules/training/custom_loss_function/__init__.md) | `training/custom_loss_function/__init__.py` |
| [elbo.md](modules/training/custom_loss_function/elbo.md) | `training/custom_loss_function/elbo.py` |
| [support_loss.md](modules/training/custom_loss_function/support_loss.md) | `training/custom_loss_function/support_loss.py` |
