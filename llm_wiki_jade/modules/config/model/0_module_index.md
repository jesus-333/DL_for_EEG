# `jade.config.model` — Subpackage Index

Table of contents. Start with the [subpackage quickstart](0_module_quickstart.md). Parent: [config index](../0_module_index.md).

| File | Page | Description |
|---|---|---|
| `__init__.py` | [__init__.md](__init__.md) | `MODEL_CONFIG_REGISTRY` + `get_model_config` factory; re-exports the model + pooling configs. |
| `generic.py` | [generic.md](generic.md) | `model_config` base class + polymorphic `from_dict`. |
| `EEGNet_cfg.py` | [EEGNet_cfg.md](EEGNet_cfg.md) | `EEGNet_config` (+ the `eegnet_pooling_config` `[pooling]` section) → `jade.model.EEGNet.EEGNet` / `EEGNet_Classifier`. |
| `EEGConformer_cfg.py` | [EEGConformer_cfg.md](EEGConformer_cfg.md) | `EEGConformer_config` (+ the `conformer_pooling_config` `[pooling]` section) → `jade.model.EEGConformer.EEGConformer` / `EEGConformer_Classifier`. |
| `ShallowFBCSPNet_cfg.py` | [ShallowFBCSPNet_cfg.md](ShallowFBCSPNet_cfg.md) | `ShallowFBCSPNet_config` (+ the `shallowfbcspnet_pooling_config` `[pooling]` section, one stage) → `jade.model.ShallowFBCSPNet.ShallowFBCSPNet` / `ShallowFBCSPNet_Classifier`. |
| `Deep4Net_cfg.py` | [Deep4Net_cfg.md](Deep4Net_cfg.md) | `Deep4Net_config` (+ the `deep4net_pooling_config` `[pooling]` section, **four** independent stages) → `jade.model.Deep4Net.Deep4Net` / `Deep4Net_Classifier`. |
| `TIDNet_cfg.py` | [TIDNet_cfg.md](TIDNet_cfg.md) | `TIDNet_config` (+ the `tidnet_pooling_config` `[pooling]` section, one decimating stage) → `jade.model.TIDNet.TIDNet` / `TIDNet_Classifier`. |
| `vEEGNet_cfg.py` | [vEEGNet_cfg.md](vEEGNet_cfg.md) | `vEEGNet_config` (reusing the `eegnet_pooling_config` `[pooling]` section) → `jade.model.vEEGNet.vEEGNet`. The first **generative** model config (`task = 'generative'`); it has no classifier wrapper and no `n_classes` / `use_softmax`. |
| `hvEEGNet_cfg.py` | [hvEEGNet_cfg.md](hvEEGNet_cfg.md) | `hvEEGNet_config` (reusing the `eegnet_pooling_config` `[pooling]` section, defaulting to the paper's no-pool / ×10 stages) → `jade.model.hvEEGNet.hvEEGNet`. Generative (`task = 'generative'`), no classifier. |
| `braindecode_pooling_extension_cfg.py` | [braindecode_pooling_extension_cfg.md](braindecode_pooling_extension_cfg.md) | `braindecode_wrapper_config` → `jade.model.braindecode_pooling_extension.braindecode_pooling_wrapper` (raw braindecode `model_kwargs` + the pooling operator to swap in). |
| `pooling_1d_cfg.py` | [pooling_1d_cfg.md](pooling_1d_cfg.md) | The 1D pooling config family (`avg`/`blur`/`sinc`/`dwt`/`max`) + `POOLING_CONFIG_REGISTRY` + `get_pooling_config`. |
