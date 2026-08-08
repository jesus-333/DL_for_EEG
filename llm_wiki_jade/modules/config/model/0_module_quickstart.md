# `jade.config.model` — Subpackage Quickstart

> READ FIRST. Overview of the model-config subpackage. Field-level detail is on the per-file pages.

## Purpose

`jade.config.model` holds the **configuration objects that describe a `jade.model` network**. This is the piece that finally moves models off the old plain-dict configuration and onto the same validated, (de)serializable dataclass system as the rest of `jade.config`. A model config only *describes* a model; `jade.model.get_model(config)` turns it into the real `torch.nn.Module`, selecting the class from the config `name`.

Two concerns live here:

1. **Model configs** — one dataclass per model, in a `{model_name}_cfg.py` file (the `cfg` suffix keeps config files apart from the model implementation files in `jade.model`). Currently: [`EEGNet_config`](EEGNet_cfg.md) (covers both `EEGNet` and `EEGNet_Classifier`).
2. **The 1D pooling config family** — a single file, [`pooling_1d_cfg.py`](pooling_1d_cfg.md), holding one config per pooling operator (avg/blur/sinc/dwt/max), because pooling is reused several times inside a model and the operators share most parameters.

## The registry + factory pattern (models)

Defined in [`__init__.py`](__init__.md), mirroring the optimizer/loss families:

- **`model_config`** ([generic.md](generic.md)) — base class carrying only `name`; overrides `from_dict` for **polymorphic** deserialization (reads `name`, looks it up in the registry, delegates to the concrete subclass). The registry is imported lazily inside the method to avoid a circular import.
- **`MODEL_CONFIG_REGISTRY`** — `{"eegnet": EEGNet_config, "eegnet_classifier": EEGNet_config, "eegnetclassifier": EEGNet_config}` (the aliases mirror `jade.model.MODEL_REGISTRY`; all resolve to the one `EEGNet_config`, which carries the actual `name` used to pick `EEGNet` vs `EEGNet_Classifier`).
- **`get_model_config(config)`** — returns a config untouched if already a `model_config`, promotes a `dict` (keyed on `name`) via `model_config.from_dict`, else raises `TypeError`. This is the promoter used by `jade.model.get_model`, `EEGNet.__init__` and `cross_check_config`.

## The pooling config family

A parallel family, all in [`pooling_1d_cfg.py`](pooling_1d_cfg.md), discriminated on `pooling_type` (not `name`, because that is the key the runtime `get_pooling_layer` reads):

- `pooling_config` (base: `pooling_type`, `decimation_factor`, polymorphic `from_dict`) → `_fir_pooling_config` (shared FIR options: `padding_mode`, `phase`, `gain`, `warn_if_worse_than_boxcar`) → `avg_pooling_config` / `blur_pooling_config` / `sinc_pooling_config` / `dwt_pooling_config`; `max_pooling_config` inherits the minimal base only (the max layer has no FIR filter, just `ceil_mode`).
- **`POOLING_CONFIG_REGISTRY`** maps every runtime alias (`avg`/`average`/`boxcar`, `max`, `blur`/`blurpool`/`binomial`/`gaussian`, `sinc`/`kaiser`, `dwt`/`wavelet`) to its config class, and **`get_pooling_config(config)`** promotes a dict. The canonical field names match the pooling layers' keys, so `config.to_dict()` can feed `jade.model.pooling_1d.get_pooling_layer` directly.
- **Used by EEGNet.** `EEGNet_config`'s `[pooling]` section is its own dataclass, `eegnet_pooling_config` (two independently-typed stages), documented on the [EEGNet_cfg page](EEGNet_cfg.md). It reuses `get_pooling_config` here to validate each stage's type-specific params.

## Design notes

- Every field has a sensible (BCI IV-2a style) default so a bare `EEGNet_config()` / pooling config is valid and can seed a skeleton TOML (used by the `create-default-config` cli command).
- Kernels/tuples are normalized from lists in `__post_init__` (a TOML/JSON round-trip turns tuples into lists), exactly like `adam_config.betas`.
- The subpackage is **torch-free** like the rest of `jade.config`: it validates activation names against a module constant, it does not import `torch` or `jade.model`.
- **Loss ↔ model coupling**: `EEGNet_config.use_softmax` must stay consistent with the training loss (`nll` ⇒ `use_softmax=True`, `cross_entropy` ⇒ `use_softmax=False`); this is enforced by [`cross_check_config`](../../training/support_training.md).
