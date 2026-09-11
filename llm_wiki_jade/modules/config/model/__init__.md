# `config/model/__init__.py`

Source: `src/jade/config/model/__init__.py`

Subpackage aggregator for the model configs. Defines the model registry + factory and re-exports the model configs and the pooling config family. Mirrors the optimizer/loss subpackage `__init__` pattern.

## `MODEL_CONFIG_REGISTRY`

Maps a lowercase model name to its config class (the aliases mirror `jade.model.MODEL_REGISTRY`):

| Alias | Config class |
|---|---|
| `eegnet` | `EEGNet_config` |
| `eegnet_classifier` | `EEGNet_config` |
| `eegnetclassifier` | `EEGNet_config` |
| `braindecode_wrapper` | `braindecode_wrapper_config` |
| `braindecode` | `braindecode_wrapper_config` |
| `braindecode_pooling` | `braindecode_wrapper_config` |

The three `eegnet*` aliases resolve to the single `EEGNet_config` (its own `name` field later selects `EEGNet` vs `EEGNet_Classifier`); the three `braindecode*` aliases resolve to [`braindecode_wrapper_config`](braindecode_pooling_extension_cfg.md).

## `get_model_config(config) -> model_config`

Idempotent promoter:
- an already-built `model_config` is returned untouched,
- a `dict` is promoted through the polymorphic `model_config.from_dict` (keyed on `name`),
- anything else raises `TypeError`.

Used by `jade.model.get_model`, both `EEGNet` constructors, and `jade.training.support_training.cross_check_config`.

## Public API (`__all__`)

- Model configs: `model_config`, `EEGNet_config`, `eegnet_pooling_config` (the EEGNet `[pooling]` section), `braindecode_wrapper_config`, `MODEL_CONFIG_REGISTRY`, `get_model_config`.
- Pooling configs (re-exported from [`pooling_1d_cfg`](pooling_1d_cfg.md)): `pooling_config`, `avg_pooling_config`, `blur_pooling_config`, `sinc_pooling_config`, `dwt_pooling_config`, `max_pooling_config`, `POOLING_CONFIG_REGISTRY`, `get_pooling_config`.

Everything here is also re-exported one level up from `jade.config.__init__`.
