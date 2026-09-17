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
| `eegconformer` | `EEGConformer_config` |
| `eegconformer_classifier` | `EEGConformer_config` |
| `eegconformerclassifier` | `EEGConformer_config` |
| `shallowfbcspnet` | `ShallowFBCSPNet_config` |
| `shallowfbcspnet_classifier` | `ShallowFBCSPNet_config` |
| `shallowfbcspnetclassifier` | `ShallowFBCSPNet_config` |
| `deep4net` | `Deep4Net_config` |
| `deep4net_classifier` | `Deep4Net_config` |
| `deep4netclassifier` | `Deep4Net_config` |
| `tidnet` | `TIDNet_config` |
| `tidnet_classifier` | `TIDNet_config` |
| `tidnetclassifier` | `TIDNet_config` |
| `veegnet` | [`vEEGNet_config`](vEEGNet_cfg.md) |
| `vae_eegnet` | [`vEEGNet_config`](vEEGNet_cfg.md) |
| `braindecode_wrapper` | `braindecode_wrapper_config` |
| `braindecode` | `braindecode_wrapper_config` |
| `braindecode_pooling` | `braindecode_wrapper_config` |

The three `eegnet*` aliases resolve to the single `EEGNet_config` (its own `name` field later selects `EEGNet` vs `EEGNet_Classifier`); likewise each of the `eegconformer*`, `shallowfbcspnet*`, `deep4net*` and `tidnet*` alias triplets resolves to the single [`EEGConformer_config`](EEGConformer_cfg.md), [`ShallowFBCSPNet_config`](ShallowFBCSPNet_cfg.md), [`Deep4Net_config`](Deep4Net_cfg.md) or [`TIDNet_config`](TIDNet_cfg.md) (each selecting the extractor vs its `_Classifier`); the three `braindecode*` aliases resolve to [`braindecode_wrapper_config`](braindecode_pooling_extension_cfg.md).

[`vEEGNet_config`](vEEGNet_cfg.md) is the exception to the triplet pattern: it has only `veegnet` and the alias `vae_eegnet`, because the model is a variational autoencoder with no classifier variant. **`veegnet_classifier` / `veegnetclassifier` are deliberately left unregistered** — mapping them to the generative class would silently build the wrong model once the classification branch lands, so the slot is reserved and asking for one today raises the usual `ValueError` listing the known names.

## `get_model_config(config) -> model_config`

Idempotent promoter:
- an already-built `model_config` is returned untouched,
- a `dict` is promoted through the polymorphic `model_config.from_dict` (keyed on `name`),
- anything else raises `TypeError`.

Used by `jade.model.get_model`, every model constructor (both `EEGNet`, `EEGConformer`, `ShallowFBCSPNet`, `Deep4Net` and `TIDNet` constructors), and `jade.training.support_training.cross_check_config`.

## Public API (`__all__`)

- Model configs: `model_config`, `EEGNet_config`, `eegnet_pooling_config` (the EEGNet `[pooling]` section, reused as is by `vEEGNet_config`), `EEGConformer_config`, `conformer_pooling_config`, `ShallowFBCSPNet_config`, `shallowfbcspnet_pooling_config`, `Deep4Net_config`, `deep4net_pooling_config`, `TIDNet_config`, `tidnet_pooling_config` (each model's `[pooling]` section), `vEEGNet_config`, `braindecode_wrapper_config`, `MODEL_CONFIG_REGISTRY`, `get_model_config`.
- Pooling configs (re-exported from [`pooling_1d_cfg`](pooling_1d_cfg.md)): `pooling_config`, `avg_pooling_config`, `blur_pooling_config`, `sinc_pooling_config`, `dwt_pooling_config`, `max_pooling_config`, `POOLING_CONFIG_REGISTRY`, `get_pooling_config`.

Everything here is also re-exported one level up from `jade.config.__init__`.
