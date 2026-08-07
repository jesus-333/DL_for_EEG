# `jade.config.model` — Subpackage Index

Table of contents. Start with the [subpackage quickstart](0_module_quickstart.md). Parent: [config index](../0_module_index.md).

| File | Page | Description |
|---|---|---|
| `__init__.py` | [__init__.md](__init__.md) | `MODEL_CONFIG_REGISTRY` + `get_model_config` factory; re-exports the model + pooling configs. |
| `generic.py` | [generic.md](generic.md) | `model_config` base class + polymorphic `from_dict`. |
| `EEGNet_cfg.py` | [EEGNet_cfg.md](EEGNet_cfg.md) | `EEGNet_config` → `jade.model.EEGNet.EEGNet` / `EEGNet_Classifier`. |
| `pooling_1d_cfg.py` | [pooling_1d_cfg.md](pooling_1d_cfg.md) | The 1D pooling config family (`avg`/`blur`/`sinc`/`dwt`/`max`) + `POOLING_CONFIG_REGISTRY` + `get_pooling_config`. |
