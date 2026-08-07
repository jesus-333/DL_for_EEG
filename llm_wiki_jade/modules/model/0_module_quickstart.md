# `jade.model` — Module Quickstart

> **Read this first, every session.** After this page you should understand the whole `model` module without opening the source. Depth over length; no code dumps here.

## Purpose

`jade.model` holds the PyTorch neural-network models of the `jade` package (Just Another Deep-learning EEG package) **plus** a self-contained research sub-package for anti-aliased 1D pooling. Two concerns live side by side:

1. **EEG models** — an `EEGNet` implementation (and a classifier wrapper) plus small creation helpers. This is the production part used by `jade.training`.
2. **`pooling_1d`** — a DSP-grounded family of 1D decimation ("pooling") layers (`avg` / `blur` / `sinc` / `dwt` / `max`) built for an anti-aliasing ablation study. EEGNet consumes them channel-wise through its `[pooling]` config section (via `get_multichannel_pooling`); they also stand alone as research infrastructure.

## Top-level factory: `get_model` / `MODEL_REGISTRY`

The single entry point for building a model is `get_model(config) -> nn.Module` (in [`__init__.py`](__init__.md)). It takes a **model config** — an instance of a `jade.config.model.model_config` subclass (e.g. `EEGNet_config`), or a plain dict that is promoted to one via `get_model_config`. It reads `config.name` (case-insensitive), looks it up in `MODEL_REGISTRY`, and calls that class with the config. Every model class takes exactly one argument — the config — so the factory is uniform.

| `name` alias | Class |
|---|---|
| `eegnet` | `EEGNet` |
| `eegnet_classifier`, `eegnetclassifier` | `EEGNet_Classifier` |

Raises `ValueError` if `name` is unknown. This mirrors the `get_pooling_layer` / `POOLING_REGISTRY` pattern one level down in `pooling_1d`, and the `get_optimizer` etc. registries elsewhere in `jade`. (The old `get_model({"model_name": ...})` dict interface is gone — the discriminator is `name`, matching the config families.)

## EEGNet at a high level

[`EEGNet`](EEGNet.md) is the classic Lawhern et al. architecture in PyTorch, operating on 4D tensors `(B, depth, C, T)` (batch, input depth = 1 for raw EEG, channels `C`, time `T`) via `nn.Conv2d`. Three blocks:

1. **Temporal filter** — `Conv2d` (temporal kernel, `padding='same'`) + `BatchNorm2d`.
2. **Spatial filter** — depthwise `Conv2d` (`groups=filter_1`, learns `D` spatial filters per temporal filter) + `BatchNorm2d` + activation + `AvgPool2d` + dropout.
3. **Separable convolution** — depthwise `Conv2d` + pointwise `1x1 Conv2d` + `BatchNorm2d` + activation + `AvgPool2d` + dropout.

The final feedforward + softmax classification layer of the original paper is **not** part of `EEGNet` — it outputs features (optionally flattened via `flatten_output`). [`EEGNet_Classifier`](EEGNet.md) adds it: it wraps an `EEGNet`, infers the flattened feature count with a dummy forward pass (`compute_number_of_neurons`), then appends `Linear -> (LogSoftmax | Identity)`. It also exposes `classify(x)` returning predicted class indices, which `jade.training.metrics` prefers over `argmax(model(x))`.

Both classes now consume an [`EEGNet_config`](../config/model/EEGNet_cfg.md) dataclass (a dict is still accepted and promoted via `get_model_config`); they call `get_model_config(config)` at the top of `__init__` and then read the attributes, so all defaulting and validation live in the config, not in the model.

**Key fact:** EEGNet's two pooling stages are configured by the `[pooling]` section of `EEGNet_config` (fields `pooling_type_*`, `pooling_factor_*`, `params_*`) and built by `support_functions.get_multichannel_pooling`. `pooling_type='default'` reproduces the classic `nn.AvgPool2d`; `avg`/`max`/`blur`/`sinc`/`dwt` select a `pooling_1d` operator applied channel-wise. This replaced the old `p_kernel_1`/`p_kernel_2` fields.

## `aEEGNet` — attention variant (work in progress)

[`aEEGNet`](aEEGNet.md) is despite its filename **not** about anti-aliasing: its own docstring describes an **attention-augmented** EEGNet. It is currently a **stub** (`aEEGNet.forward` returns its input unchanged). The real work-in-progress there is `channels_attention_head`, a multichannel-EEG attention head using query/key/value linear maps from [`support_functions.create_head`](support_functions.md). It has a live bug (subclasses `torch.nn.module`, lowercase) and is not usable yet — see its page.

## The `pooling_1d` family (research)

See the dedicated sub-quickstart: [`pooling_1d/0_module_quickstart.md`](pooling_1d/0_module_quickstart.md).

In one paragraph: every pooling operator is unified as **"dense FIR low-pass filter (stride 1) → subsample by M"**, following Zhang (ICML 2019). Decimation by `M` without a low-pass filter folds all energy above the new Nyquist `π/M` back into the retained band (aliasing); the low-pass filter is the anti-aliasing step. The arms differ **only** in the filter `h`:

| Layer | Filter | Role in the study |
|---|---|---|
| `avg_pool_1d` | boxcar (length `M`) | **baseline** (standard average pooling, made explicit) |
| `blur_pool_1d` | binomial / Gaussian / boxcar | fixed heuristic (1D BlurPool) |
| `sinc_pool_1d` | windowed-sinc (Kaiser) | designed "principled ceiling" |
| `dwt_pool_1d` | orthogonal wavelet low-pass | filter-bank band-split |
| `max_pool_1d` | none (nonlinear) | un-anti-aliased reference |

The family is **provably nested**: average pooling *is* the boxcar special case, so `avg_pool_1d` ≡ `blur_pool_1d(kernel='boxcar', kernel_size=M)`, and all arms of an ablation differ in exactly one designed object `h`. Every layer outputs exactly `T // M` samples (matching `nn.AvgPool1d(M, stride=M)`), has **zero trainable parameters**, filters **depthwise** on `(B, D, T)`, and is unit-DC-gain normalized by default. A base class `_fir_decimation_pool_1d` ([`support_pooling.py`](pooling_1d/support_pooling.md)) implements everything (padding, subsampling, and frequency-domain figures of merit: `stopband_leakage`, `stopband_attenuation_db`, `passband_ripple_db`, `group_delay_samples`); subclasses only build the kernel. A `get_pooling_layer` factory + a `self_test()` of the study's invariants complete the package, and `python -m jade.model.pooling_1d` prints a figures-of-merit table for the two EEGNet pooling stages. Background DSP theory for these layers is in `other/antialiased_pooling_theory.pdf`.

**Central methodological warning** (baked into the code): short fixed BlurPool kernels (`[1,2,1]`, `[1,4,6,4,1]`) have a *fixed* bandwidth, but the required cutoff `π/M` shrinks as `M` grows. At `M=8` a length-3 binomial kernel is ~3× worse than plain averaging. Hence defaults scale kernel length with `M`, and `warn_if_worse_than_boxcar` / `stopband_leakage()` catch the trap.

## Structure

```
model/
├── __init__.py            get_model / MODEL_REGISTRY factory
├── EEGNet.py              EEGNet + EEGNet_Classifier
├── aEEGNet.py             attention variant (WIP stub) + channels_attention_head
├── support_functions.py   get_activation, get_dropout, count_trainable_parameters, create_head, compute_attention_values
├── support_modules.py     empty placeholder (TODO)
└── pooling_1d/            anti-aliased 1D pooling research sub-package (self-contained)
```

## Dependencies

- **External:** `torch` (all models/layers), `numpy` (pooling kernel math), and — lazily, only inside `sinc_pool_1d` / optional `dwt_pool_1d` wavelets — `scipy` and `pywt`.
- **Internal:** `EEGNet` and `aEEGNet` import `jade.model.support_functions`. `pooling_1d` is otherwise standalone (imports only its own submodules). Nothing in the models currently imports `pooling_1d`.

## How `model` connects to the rest of `jade`

- **`jade.training`** consumes models as opaque `nn.Module`s. It builds/receives a model, calls `model.to(device)` and `model(x)`, and for metrics prefers `model.classify(x)` (which `EEGNet_Classifier` provides), falling back to `argmax(model(x))`. `get_model` is the intended construction path.
- **`jade.config`** now has a [`model/` sub-package](../config/model/0_module_quickstart.md) (`EEGNet_config` + the pooling config family), so model hyperparameters are a validated dataclass like everything else. `get_model` consumes a `model_config` instance (selecting on `config.name`); `config/optimizers/__init__.py` cites `MODEL_REGISTRY` / `get_model` as the convention its own registry mirrors, and the new `MODEL_CONFIG_REGISTRY` / `get_model_config` closes the loop on the config side.
- **`jade.dataset`** produces the `(B, depth, C, T)` tensors EEGNet consumes.
