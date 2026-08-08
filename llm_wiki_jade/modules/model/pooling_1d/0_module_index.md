# `jade.model.pooling_1d` — Sub-package Index

Table of contents for the `pooling_1d` sub-package. Start with the [quickstart](0_module_quickstart.md). Parent module: [`jade.model`](../0_module_index.md). Background theory: `other/antialiased_pooling_theory.pdf`.

| Page | Source | One-line description |
|---|---|---|
| [0_module_quickstart.md](0_module_quickstart.md) | — | Read-first overview: the "FIR filter → subsample" family and the ablation design. |
| [0_module_index.md](0_module_index.md) | — | This page. |
| [__init__.md](__init__.md) | `pooling_1d/__init__.py` | `get_pooling_layer` / `POOLING_REGISTRY` (+ `MULTICHANNEL_POOLING_REGISTRY`) factory, `self_test`, motivation docstring. |
| [__main__.md](__main__.md) | `pooling_1d/__main__.py` | `python -m jade.model.pooling_1d`: self-test + figures-of-merit table for the two EEGNet stages. |
| [support_pooling.md](support_pooling.md) | `pooling_1d/support_pooling.py` | Base classes `_fir_decimation_pool_1d` and `_multichannel_pool_1d`, functional core `_filter_and_subsample`, kernel helpers, DSP figures of merit. |
| [avg_pool.md](avg_pool.md) | `pooling_1d/avg_pool.py` | `avg_pool_1d` (+ `avg_pool_1d_multichannel`) — boxcar baseline (Dirichlet response). |
| [blur_pool.md](blur_pool.md) | `pooling_1d/blur_pool.py` | `blur_pool_1d` (+ `blur_pool_1d_multichannel`) — binomial / Gaussian / boxcar fixed kernel; length must scale with `M`. |
| [max_pool.md](max_pool.md) | `pooling_1d/max_pool.py` | `max_pool_1d` (+ `max_pool_1d_multichannel`) — nonlinear un-anti-aliased reference (no frequency response). |
| [sinc_pool.md](sinc_pool.md) | `pooling_1d/sinc_pool.py` | `sinc_pool_1d` (+ `sinc_pool_1d_multichannel`) — windowed-sinc (Kaiser) designed decimation filter (needs SciPy). |
| [dwt_pool.md](dwt_pool.md) | `pooling_1d/dwt_pool.py` | `dwt_pool_1d` (+ `dwt_pool_1d_multichannel`) + `WAVELET_FILTERS` — orthogonal wavelet filter bank, power-of-two `M`, noble identity. |
