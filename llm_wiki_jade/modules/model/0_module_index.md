# `jade.model` — Module Index

Table of contents for the `model` module wiki. Start with the [quickstart](0_module_quickstart.md).

## Root (`src/jade/model/`)

| Page | Source | One-line description |
|---|---|---|
| [0_module_quickstart.md](0_module_quickstart.md) | — | Read-first overview of the whole module (models + pooling research). |
| [0_module_index.md](0_module_index.md) | — | This page. |
| [__init__.md](__init__.md) | `model/__init__.py` | `get_model` / `MODEL_REGISTRY` top-level model factory. |
| [EEGNet.md](EEGNet.md) | `model/EEGNet.py` | `EEGNet` (feature extractor) and `EEGNet_Classifier` (with head + `classify`). |
| [EEGConformer.md](EEGConformer.md) | `model/EEGConformer.py` | `EEGConformer` (conv patch embedding + transformer encoder) and `EEGConformer_Classifier` (with flatten + MLP head + `classify`); `get_conformer_pooling` wires the `pooling_1d` operators into the patch embedding. |
| [ShallowFBCSPNet.md](ShallowFBCSPNet.md) | `model/ShallowFBCSPNet.py` | `ShallowFBCSPNet` (temporal + spatial conv, squaring, pooling, logarithm) and `ShallowFBCSPNet_Classifier` (with flatten + `Linear` head + `classify`); `get_shallowfbcspnet_pooling` wires the `pooling_1d` operators into its single pooling stage. |
| [Deep4Net.md](Deep4Net.md) | `model/Deep4Net.py` | `Deep4Net` (four convolution + pooling blocks) and `Deep4Net_Classifier`; `get_deep4net_pooling` wires the `pooling_1d` operators into each of the **four** independently configured stages. |
| [TIDNet.md](TIDNet.md) | `model/TIDNet.py` | `TIDNet` (residual dilated temporal filter, decimating pooling, dense spatial filter, adaptive summary) and `TIDNet_Classifier`; `get_tidnet_pooling` wires the `pooling_1d` operators into its single decimating stage. |
| [vEEGNet.md](vEEGNet.md) | `model/vEEGNet.py` | `vEEGNet` — a variational autoencoder (EEGNet encoder + mirrored decoder), with `reconstruct` and `generate`; plus `gaussian_sampling_layer`, `eegnet_decoder` and `same_padding_conv_transpose_2d`. **The first generative model of the package.** |
| [braindecode_pooling_extension.md](braindecode_pooling_extension.md) | `model/braindecode_pooling_extension.py` | `braindecode_pooling_wrapper` — wrap a braindecode model and swap its pooling layers for the `pooling_1d` operators. |
| [aEEGNet.md](aEEGNet.md) | `model/aEEGNet.py` | Attention-augmented EEGNet variant — WIP stub — plus `channels_attention_head`. |
| [support_functions.md](support_functions.md) | `model/support_functions.py` | `get_activation`, `get_nonlinearity` (+ the `square` / `safe_log` modules), `get_dropout`, `get_multichannel_pooling`, `count_trainable_parameters`, `create_head`, `compute_attention_values`. |
| [support_modules.md](support_modules.md) | `model/support_modules.py` | Empty placeholder file (TODO). |

## Sub-package `pooling_1d/` (`src/jade/model/pooling_1d/`)

Anti-aliased 1D decimation layers unified as "FIR low-pass filter → subsample". Background theory: `other/antialiased_pooling_theory.pdf`.

| Page | Source | One-line description |
|---|---|---|
| [pooling_1d/0_module_quickstart.md](pooling_1d/0_module_quickstart.md) | — | Read-first overview of the pooling family and the ablation design. |
| [pooling_1d/0_module_index.md](pooling_1d/0_module_index.md) | — | Sub-package table of contents. |
| [pooling_1d/__init__.md](pooling_1d/__init__.md) | `pooling_1d/__init__.py` | `get_pooling_layer` / `POOLING_REGISTRY` factory, `self_test`, and the motivation docstring. |
| [pooling_1d/__main__.md](pooling_1d/__main__.md) | `pooling_1d/__main__.py` | `python -m jade.model.pooling_1d` entry point (self-test + figures-of-merit table). |
| [pooling_1d/support_pooling.md](pooling_1d/support_pooling.md) | `pooling_1d/support_pooling.py` | Base class `_fir_decimation_pool_1d`, functional core, kernel helpers, DSP math. |
| [pooling_1d/avg_pool.md](pooling_1d/avg_pool.md) | `pooling_1d/avg_pool.py` | `avg_pool_1d` — boxcar baseline. |
| [pooling_1d/blur_pool.md](pooling_1d/blur_pool.md) | `pooling_1d/blur_pool.py` | `blur_pool_1d` — binomial / Gaussian / boxcar fixed kernel. |
| [pooling_1d/max_pool.md](pooling_1d/max_pool.md) | `pooling_1d/max_pool.py` | `max_pool_1d` — nonlinear un-anti-aliased reference. |
| [pooling_1d/sinc_pool.md](pooling_1d/sinc_pool.md) | `pooling_1d/sinc_pool.py` | `sinc_pool_1d` — windowed-sinc (Kaiser) designed filter. |
| [pooling_1d/dwt_pool.md](pooling_1d/dwt_pool.md) | `pooling_1d/dwt_pool.py` | `dwt_pool_1d` + `WAVELET_FILTERS` — orthogonal wavelet filter bank. |
