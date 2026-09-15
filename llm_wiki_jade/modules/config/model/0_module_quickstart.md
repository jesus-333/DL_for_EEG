# `jade.config.model` — Subpackage Quickstart

> READ FIRST. Overview of the model-config subpackage. Field-level detail is on the per-file pages.

## Purpose

`jade.config.model` holds the **configuration objects that describe a `jade.model` network**. This is the piece that finally moves models off the old plain-dict configuration and onto the same validated, (de)serializable dataclass system as the rest of `jade.config`. A model config only *describes* a model; `jade.model.get_model(config)` turns it into the real `torch.nn.Module`, selecting the class from the config `name`.

Two concerns live here:

1. **Model configs** — one dataclass per model, in a `{model_name}_cfg.py` file (the `cfg` suffix keeps config files apart from the model implementation files in `jade.model`). Currently: [`EEGNet_config`](EEGNet_cfg.md) (covers both `EEGNet` and `EEGNet_Classifier`), [`EEGConformer_config`](EEGConformer_cfg.md), [`ShallowFBCSPNet_config`](ShallowFBCSPNet_cfg.md), [`Deep4Net_config`](Deep4Net_cfg.md), [`TIDNet_config`](TIDNet_cfg.md) (each covering both its extractor and its `_Classifier`) and [`braindecode_wrapper_config`](braindecode_pooling_extension_cfg.md) (the [braindecode pooling wrapper](../../model/braindecode_pooling_extension.md)). The braindecode one is deliberately different: it does **not** describe the model hyper-parameters (braindecode owns and validates those, so they travel as a raw `model_kwargs` dict) — it only carries the model name and the pooling operator to swap in.
2. **The 1D pooling config family** — a single file, [`pooling_1d_cfg.py`](pooling_1d_cfg.md), holding one config per pooling operator (avg/blur/sinc/dwt/max), because pooling is reused several times inside a model and the operators share most parameters.

## The registry + factory pattern (models)

Defined in [`__init__.py`](__init__.md), mirroring the optimizer/loss families:

- **`model_config`** ([generic.md](generic.md)) — base class carrying only `name`; overrides `from_dict` for **polymorphic** deserialization (reads `name`, looks it up in the registry, delegates to the concrete subclass). The registry is imported lazily inside the method to avoid a circular import.
- **`MODEL_CONFIG_REGISTRY`** — the `eegnet` / `eegnet_classifier` / `eegnetclassifier` aliases resolve to `EEGNet_config` (which carries the actual `name` used to pick `EEGNet` vs `EEGNet_Classifier`), the `eegconformer*`, `shallowfbcspnet*`, `deep4net*` and `tidnet*` alias triplets resolve to `EEGConformer_config`, `ShallowFBCSPNet_config`, `Deep4Net_config` and `TIDNet_config` respectively, and `braindecode` / `braindecode_wrapper` / `braindecode_pooling` resolve to `braindecode_wrapper_config`. The aliases mirror `jade.model.MODEL_REGISTRY`.
- **`get_model_config(config)`** — returns a config untouched if already a `model_config`, promotes a `dict` (keyed on `name`) via `model_config.from_dict`, else raises `TypeError`. This is the promoter used by `jade.model.get_model`, `EEGNet.__init__` and `cross_check_config`.

## The pooling config family

A parallel family, all in [`pooling_1d_cfg.py`](pooling_1d_cfg.md), discriminated on `pooling_type` (not `name`, because that is the key the runtime `get_pooling_layer` reads):

- `pooling_config` (base: `pooling_type`, `decimation_factor`, polymorphic `from_dict`) → `_fir_pooling_config` (shared FIR options: `padding_mode`, `phase`, `gain`, `warn_if_worse_than_boxcar`) → `avg_pooling_config` / `blur_pooling_config` / `sinc_pooling_config` / `dwt_pooling_config`; `max_pooling_config` inherits the minimal base only (the max layer has no FIR filter, just `ceil_mode`).
- **`POOLING_CONFIG_REGISTRY`** maps every runtime alias (`avg`/`average`/`boxcar`, `max`, `blur`/`blurpool`/`binomial`/`gaussian`, `sinc`/`kaiser`, `dwt`/`wavelet`) to its config class, and **`get_pooling_config(config)`** promotes a dict. The canonical field names match the pooling layers' keys, so a `pooling_config` object (or its `to_dict()`) can feed `jade.model.pooling_1d.get_pooling_layer` / `jade.model.support_functions.get_multichannel_pooling` directly — the pooling layers read their config through a `_get` helper that accepts a dataclass *or* a dict, so no dict round-trip is needed.
- **Used by every model.** Each model config owns a bespoke `[pooling]` section dataclass, sized to that model's pooling stages, and all of them reuse `get_pooling_config` here to validate each stage's type-specific params:

  | Model config | `[pooling]` section | Stages | Shape of the section |
  |---|---|---|---|
  | [`EEGNet_config`](EEGNet_cfg.md) | `eegnet_pooling_config` | 2 | `pooling_type_N` / `pooling_factor_N` / `params_N` (non-overlapping) |
  | [`EEGConformer_config`](EEGConformer_cfg.md) | `conformer_pooling_config` | 1 | `pooling_kernel` separate from `pooling_stride` (the original pool is **overlapping**) |
  | [`ShallowFBCSPNet_config`](ShallowFBCSPNet_cfg.md) | `shallowfbcspnet_pooling_config` | 1 | same as the Conformer's, plus `pool_mode` (the original pool is **overlapping**) |
  | [`Deep4Net_config`](Deep4Net_cfg.md) | `deep4net_pooling_config` | 4 | `pooling_type_N` / `pooling_factor_N` / `pool_mode_N` / `params_N` (non-overlapping) |
  | [`TIDNet_config`](TIDNet_cfg.md) | `tidnet_pooling_config` | 1 | `pooling_factor` + `pool_mode` (non-overlapping; the factor also sets the default `summary`) |

  The rule of thumb: a model whose original pooling window **overlaps** needs the window and the decimation as two keys (the `pooling_1d` operators are decimators and always output `T // M`); a model whose pooling is non-overlapping needs only the factor, and is then perfectly interchangeable across operators.

## Design notes

- Every field has a sensible (BCI IV-2a style) default so a bare `EEGNet_config()` / pooling config is valid and can seed a skeleton TOML (used by the `create-default-config` cli command).
- Kernels/tuples are normalized from lists in `__post_init__` (a TOML/JSON round-trip turns tuples into lists), exactly like `adam_config.betas`.
- The subpackage is **torch-free** like the rest of `jade.config`: it validates activation names against a module constant, it does not import `torch` or `jade.model`.
- **Loss ↔ model coupling**: `EEGNet_config.use_softmax` must stay consistent with the training loss (`nll` ⇒ `use_softmax=True`, `cross_entropy` ⇒ `use_softmax=False`); this is enforced by [`cross_check_config`](../../training/support_training.md).
