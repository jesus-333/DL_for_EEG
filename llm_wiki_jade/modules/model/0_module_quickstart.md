# `jade.model` — Module Quickstart

> **Read this first, every session.** After this page you should understand the whole `model` module without opening the source. Depth over length; no code dumps here.

## Purpose

`jade.model` holds the PyTorch neural-network models of the `jade` package (Just Another Deep-learning EEG package) **plus** a self-contained research sub-package for anti-aliased 1D pooling. Two concerns live side by side:

1. **EEG models** — `EEGNet`, `EEGConformer` (convolutional transformer), `ShallowFBCSPNet` and `Deep4Net` (the two Schirrmeister 2017 ConvNets) and `TIDNet`, each with a classifier wrapper, plus `vEEGNet`, the **generative** one (a variational autoencoder), and small creation helpers. This is the production part used by `jade.training`.
2. **`pooling_1d`** — a DSP-grounded family of 1D decimation ("pooling") layers (`avg` / `blur` / `sinc` / `dwt` / `max`) built for an anti-aliasing ablation study. EEGNet consumes them channel-wise through its `[pooling]` config section (via `get_multichannel_pooling`); they also stand alone as research infrastructure.

## Top-level factory: `get_model` / `MODEL_REGISTRY`

The single entry point for building a model is `get_model(config) -> nn.Module` (in [`__init__.py`](__init__.md)). It takes a **model config** — an instance of a `jade.config.model.model_config` subclass (e.g. `EEGNet_config`), or a plain dict that is promoted to one via `get_model_config`. It reads `config.name` (case-insensitive), looks it up in `MODEL_REGISTRY`, and calls that class with the config. Every model class takes exactly one argument — the config — so the factory is uniform.

| `name` alias | Class |
|---|---|
| `eegnet` | `EEGNet` |
| `eegnet_classifier`, `eegnetclassifier` | `EEGNet_Classifier` |
| `eegconformer` | `EEGConformer` |
| `eegconformer_classifier`, `eegconformerclassifier` | `EEGConformer_Classifier` |
| `shallowfbcspnet` | `ShallowFBCSPNet` |
| `shallowfbcspnet_classifier`, `shallowfbcspnetclassifier` | `ShallowFBCSPNet_Classifier` |
| `deep4net` | `Deep4Net` |
| `deep4net_classifier`, `deep4netclassifier` | `Deep4Net_Classifier` |
| `tidnet` | `TIDNet` |
| `tidnet_classifier`, `tidnetclassifier` | `TIDNet_Classifier` |
| `veegnet`, `vae_eegnet` | `vEEGNet` (no classifier variant — see below) |
| `braindecode_wrapper`, `braindecode`, `braindecode_pooling` | `braindecode_pooling_wrapper` |

Raises `ValueError` if `name` is unknown. This mirrors the `get_pooling_layer` / `POOLING_REGISTRY` pattern one level down in `pooling_1d`, and the `get_optimizer` etc. registries elsewhere in `jade`. (The old `get_model({"model_name": ...})` dict interface is gone — the discriminator is `name`, matching the config families.)

## EEGNet at a high level

[`EEGNet`](EEGNet.md) is the classic Lawhern et al. architecture in PyTorch, operating on 4D tensors `(B, depth, C, T)` (batch, input depth = 1 for raw EEG, channels `C`, time `T`) via `nn.Conv2d`. Three blocks:

1. **Temporal filter** — `Conv2d` (temporal kernel, `padding='same'`) + `BatchNorm2d`.
2. **Spatial filter** — depthwise `Conv2d` (`groups=filter_1`, learns `D` spatial filters per temporal filter) + `BatchNorm2d` + activation + `AvgPool2d` + dropout.
3. **Separable convolution** — depthwise `Conv2d` + pointwise `1x1 Conv2d` + `BatchNorm2d` + activation + `AvgPool2d` + dropout.

The final feedforward + softmax classification layer of the original paper is **not** part of `EEGNet` — it outputs features (optionally flattened via `flatten_output`). [`EEGNet_Classifier`](EEGNet.md) adds it: it wraps an `EEGNet`, infers the flattened feature count with a dummy forward pass (`compute_number_of_neurons`), then appends `Linear -> (LogSoftmax | Identity)`. It also exposes `classify(x)` returning predicted class indices, which `jade.training.metrics` prefers over `argmax(model(x))`.

Both classes now consume an [`EEGNet_config`](../config/model/EEGNet_cfg.md) dataclass (a dict is still accepted and promoted via `get_model_config`); they call `get_model_config(config)` at the top of `__init__` and then read the attributes, so all defaulting and validation live in the config, not in the model.

**Key fact:** EEGNet's two pooling stages are configured by the `[pooling]` section of `EEGNet_config` (fields `pooling_type_*`, `pooling_factor_*`, `params_*`) and built by `support_functions.get_multichannel_pooling`. `pooling_type='default'` reproduces the classic `nn.AvgPool2d`; `avg`/`max`/`blur`/`sinc`/`dwt` select a `pooling_1d` operator applied channel-wise. This replaced the old `p_kernel_1`/`p_kernel_2` fields.

## EEG Conformer at a high level

[`EEGConformer`](EEGConformer.md) is the convolutional-transformer of Song et al. (IEEE TNSRE 2023): a shallow EEGNet-like **convolutional patch embedding** feeds a **transformer encoder**, and (for the classifier) a **flatten + MLP head**. Like `EEGNet` it operates on 4D `(B, depth, C, T)` tensors and is a jade-native re-implementation driven entirely by an [`EEGConformer_config`](../config/model/EEGConformer_cfg.md) (a dict is accepted and promoted). It depends only on `torch` — the original `einops` rearrangements are rewritten with plain tensor ops.

- **`EEGConformer`** (feature extractor): patch embedding → transformer encoder → token sequence `(B, n_tokens, emb_size)` (optionally flattened via `flatten_output`). The patch embedding = temporal conv → spatial conv (collapses the channel axis) → BatchNorm → activation → **pooling** → dropout → `1×1` projection → reshape to tokens.
- **`EEGConformer_Classifier`** (feature extractor + head): forces the extractor to flatten, sizes the MLP head with a dummy forward (`compute_number_of_neurons`), then appends `Linear → activation → Dropout → Linear → activation → Dropout → Linear → (LogSoftmax | Identity)`. Also exposes `classify(x)`. With all defaults a 22×1000 trial gives 61 tokens (61·40 = 2440, matching the reference implementation's hard-coded flatten size).

**Key fact (pooling):** the patch-embedding pooling is the `[pooling]` section of `EEGConformer_config` (`conformer_pooling_config`) and is built by `get_conformer_pooling` (in `EEGConformer.py`). `pooling_type='default'` reproduces the original **overlapping** `nn.AvgPool2d((1,75),(1,15))`; `avg`/`max`/`blur`/`sinc`/`dwt` swap in a [`pooling_1d`](pooling_1d/0_module_quickstart.md) operator applied channel-wise (decimation factor `pooling_stride`), delegating to `get_multichannel_pooling`. This is how the anti-aliasing ablation runs on the Conformer. (Not to be confused with braindecode's own `EEGConformer`, whose overlapping-pool→transformer head the `braindecode_pooling_wrapper` cannot auto-resize — its build-time probe raises a clear error instead.)

## ShallowFBCSPNet and Deep4Net at a high level

[`ShallowFBCSPNet`](ShallowFBCSPNet.md) (Shallow ConvNet) and [`Deep4Net`](Deep4Net.md) (Deep ConvNet) are the two architectures of Schirrmeister et al. (Human Brain Mapping 2017), jade-native re-implementations driven by a [`ShallowFBCSPNet_config`](../config/model/ShallowFBCSPNet_cfg.md) / [`Deep4Net_config`](../config/model/Deep4Net_cfg.md). Both work on the same 4D `(B, depth, C, T)` layout as `EEGNet` — **time last** — instead of the reference `B × F × T × 1` layout, which is what lets the `pooling_1d` channel-wise wrappers drop in unchanged.

- **`ShallowFBCSPNet`** is a learnable FBCSP: temporal conv → spatial conv (collapses the channel axis) → BatchNorm → **squaring** → **pooling** → **logarithm** → dropout, i.e. a log-variance feature. `ShallowFBCSPNet_Classifier` adds flatten + `Linear` + (`LogSoftmax` | `Identity`).
- **`Deep4Net`** is its deep counterpart: the same split first block, then three convolution + pooling blocks with 50, 100 and 200 filters. `Deep4Net_Classifier` adds the same head.

Both replace the reference **convolutional classifier** (a `Conv2d` spanning the whole remaining time axis) with the jade flatten + `Linear` head. The two are the same linear map with the same parameter count, and with the default 22×1000/4-class geometry the classifiers have **46 084** and **282 879** trainable parameters, matching `braindecode.models.ShallowFBCSPNet` / `Deep4Net` exactly. Neither reproduces the reference Xavier initialization (jade leaves the torch defaults everywhere), and `Deep4Net` deliberately drops the reference `stride_before_pool` option, which would make the pooling non-decimating.

**Key fact (pooling):** ShallowFBCSPNet has **one** pooling stage, configured like the Conformer's (`pooling_kernel` separate from `pooling_stride`, because the original window is *overlapping*: 75 every 15) and built by `get_shallowfbcspnet_pooling`. Deep4Net has **four**, configured EEGNet-style per stage (`pooling_type_1..4`, `pooling_factor_1..4`, `pool_mode_1..4`, `params_1..4`) and built by `get_deep4net_pooling`; its original pooling is non-overlapping, so `'default'` and the 1D operators agree on the output length and a single stage can be swapped at a time. The two model-specific nonlinearities (`square`, `safe_log`) live in [`support_functions`](support_functions.md) and are selected by name through `get_nonlinearity`.

## TIDNet at a high level

[`TIDNet`](TIDNet.md), the Thinker Invariance DenseNet of Kostas & Rudzicz (J. Neural Eng. 2020), is a jade-native re-implementation driven by a [`TIDNet_config`](../config/model/TIDNet_cfg.md). It chains a **residual dilated temporal filter** (weight-normalized convolutions with decreasing dilation, padded so the time axis is preserved), a **decimating pooling stage**, a **densely connected spatial filter** (DenseNet-style layers that grow the feature axis, then a `(C, 1)` convolution that collapses the EEG-channel axis), and an **adaptive average pooling** that summarizes the time axis. `TIDNet_Classifier` adds flatten + `Linear` + (`LogSoftmax` | `Identity`); with the default geometry it has **240 404** trainable parameters, matching `braindecode.models.TIDNet` exactly.

The auxiliary **subject and run heads** of the original (used there for domain adaptation) are deliberately **not** implemented: `forward` returns a plain tensor, so the model plugs into `jade.training` like every other one. Note that the original always ends with a log-softmax, so `use_softmax = True` (paired with the `nll` loss) is the faithful setting.

**Key fact (pooling):** the `[pooling]` section configures the single **decimating** stage that closes the temporal block, built by `get_tidnet_pooling`. Unlike the other models this pooling runs *before* the channel axis is collapsed, on a `(B, t_filters, C, T)` tensor — which is exactly the shape the `*_multichannel` wrappers are built for. The trailing `nn.AdaptiveAvgPool1d(summary)` is **not** part of that section: it is an adaptive resize, not a decimation, so it has no anti-aliasing counterpart. Because `summary = -1` defaults to `T / pooling_factor`, changing the factor also changes the head width — set an explicit `summary` to keep it fixed across an ablation.

## `braindecode_pooling_wrapper` — the pooling ablation on braindecode models

[`braindecode_pooling_wrapper`](braindecode_pooling_extension.md) lets the [`pooling_1d`](pooling_1d/0_module_quickstart.md) ablation run on models that already exist in [braindecode](https://braindecode.org/) (`EEGITNet`, `EEGInceptionERP`, `EEGNeX`, `ATCNet`, …), instead of re-implementing each one. Since `ShallowFBCSPNet`, `Deep4Net` and `TIDNet` now have jade-native implementations, prefer those three natively (no optional dependency, validated hyper-parameters, per-stage pooling control) and keep the wrapper for everything else. It builds the braindecode model named by its [`braindecode_wrapper_config`](../config/model/braindecode_pooling_extension_cfg.md), then walks the module tree and **replaces every pooling layer** (`nn.AvgPool1d`/`MaxPool1d`/`AvgPool2d`/`MaxPool2d`) with the chosen jade operator, **keeping each layer's decimation factor** — only the pooling *type* changes.

Key points (details on the [page](braindecode_pooling_extension.md)):

- **Factor and axis are read from the original kernel**, not hard-coded per model: a 2D kernel `(M, 1)` decimates axis 2 (ShallowFBCSPNet), `(1, M)` decimates axis 3 (EEGNet); a jade 1D operator is applied through a small axis-aware adapter (`_pool_along_axis_2d`) that reuses the same flatten-pool-restore trick as the `*_multichannel` wrappers. This is how the spec's "check 1D vs 2D / which axis" is answered.
- **The braindecode model params are a raw `model_kwargs` dict** (braindecode validates them); only the pooling side uses a jade `pooling_config`.
- **`keep_pool_nonlin`** (default `True`) keeps the post-pool nonlinearity some models wrap the pooling with (ShallowFBCSPNet's `safe_log` `pool_nonlin_exp`); `False` drops it for an extra ablation arm.
- **Overlapping pooling** (ShallowFBCSPNet: kernel 75, stride 15) uses the stride as the factor and the wrapper re-sizes the trailing classifier with a dummy forward; non-overlapping models (EEGNet, Deep4Net) keep every downstream shape.
- braindecode is an **optional dependency** (`pip install jade[braindecode]`), imported lazily — `import jade.model` never requires it.

## vEEGNet at a high level

[`vEEGNet`](vEEGNet.md) is the **first generative model** of the package: a variational autoencoder whose encoder is an `EEGNet` and whose decoder mirrors it (transposed convolutions in place of convolutions, upsampling in place of pooling). Between the two, a `gaussian_sampling_layer` turns the encoder output into the mean and log variance of a Gaussian and samples from it with the reparametrization trick.

It breaks three assumptions that hold for every other model here, which is why it needed its own loss, training cycle and config marker:

- **`forward` returns a 4-tuple** `(x_reconstructed, mu, log_var, z)`, not a tensor of scores. The order is append-only, so the reconstruction stays at index 0.
- **There is no classification branch** (deliberately out of scope for now), so no `classify`, and no `n_classes` / `use_softmax` in its config. `vEEGNet_config.task = "generative"` makes `cross_check_config` reject a classification loss. For the same reason `veegnet_classifier` is **not** a registered name: the slot is reserved so that asking for it today fails loudly instead of silently building a generative model.
- **It is trained with the [ELBO](../training/custom_loss_function/elbo.md)** by [`jade.training.vae`](../training/vae.md), not with `jade.training.classifier`.

Two extra methods complete the generative side: `reconstruct(x)` and `generate(n_samples, z, device)`.

**Key fact (pooling):** the encoder reuses EEGNet's `[pooling]` section unchanged, but **the decoder never reads the pooling factors**. Each decoder upsampler is sized from the shape its counterpart in the encoder *received*, measured at build time by a hooked dummy forward. This is what makes the reconstruction have exactly the shape of the input for any `T`: with the paper's own setup (`T = 1000`, factors 4 and 8) the encoder gives `1000 → 250 → 31`, so upsampling by the factors returns **992** samples — the bug the original implementation shipped with. Measuring the shape also makes every `pooling_1d` operator, and a disabled stage, work through one code path. Note that upsampling is *not* the inverse of pooling: each anti-aliasing arm changes how much information reaches the latent space, so reconstruction errors across arms compare different bottlenecks.

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
├── __init__.py                        get_model / MODEL_REGISTRY factory
├── EEGNet.py                          EEGNet + EEGNet_Classifier
├── EEGConformer.py                    EEGConformer + EEGConformer_Classifier (conv patch embedding + transformer)
├── ShallowFBCSPNet.py                 ShallowFBCSPNet + ShallowFBCSPNet_Classifier (Schirrmeister 2017 Shallow ConvNet)
├── Deep4Net.py                        Deep4Net + Deep4Net_Classifier (Schirrmeister 2017 Deep ConvNet, 4 pooling stages)
├── TIDNet.py                          TIDNet + TIDNet_Classifier (Kostas 2020 Thinker Invariance DenseNet)
├── vEEGNet.py                         vEEGNet (variational autoencoder: EEGNet encoder + mirrored decoder) + its building blocks
├── braindecode_pooling_extension.py   braindecode_pooling_wrapper (swap braindecode pooling for pooling_1d)
├── aEEGNet.py                         attention variant (WIP stub) + channels_attention_head
├── support_functions.py               get_activation, get_nonlinearity (+ square / safe_log), get_dropout, get_multichannel_pooling, count_trainable_parameters, create_head, compute_attention_values
├── support_modules.py                 empty placeholder (TODO)
└── pooling_1d/                        anti-aliased 1D pooling research sub-package (self-contained)
```

## Dependencies

- **External:** `torch` (all models/layers), `numpy` (pooling kernel math), and — lazily — `scipy` / `pywt` (inside `sinc_pool_1d` / optional `dwt_pool_1d` wavelets) and `braindecode` (the optional `braindecode` extra, imported only inside `braindecode_pooling_wrapper`).
- **Internal:** every model (`EEGNet`, `EEGConformer`, `ShallowFBCSPNet`, `Deep4Net`, `TIDNet`, `aEEGNet`) imports `jade.model.support_functions` (and its `jade.config.model` config); `EEGConformer`, `ShallowFBCSPNet`, `Deep4Net` and `TIDNet` also reuse `support_functions.get_multichannel_pooling` (hence `jade.model.pooling_1d`) for their 1D-pooling stages, and `ShallowFBCSPNet` / `Deep4Net` reuse `support_functions.get_nonlinearity` for the `square` / `safe_log` nonlinearities; `braindecode_pooling_extension` imports `jade.model.pooling_1d` (`get_pooling_layer`) and `jade.config.model`. `pooling_1d` is otherwise standalone (imports only its own submodules).

## How `model` connects to the rest of `jade`

- **`jade.training`** consumes models as opaque `nn.Module`s. It builds/receives a model, calls `model.to(device)` and `model(x)`, and for metrics prefers `model.classify(x)` (which `EEGNet_Classifier` provides), falling back to `argmax(model(x))` — or, for a generative model, prefers `model.reconstruct(x)` falling back to `model(x)[0]`. `get_model` is the intended construction path. A classifier is trained by `jade.training.classifier`, a generative model by `jade.training.vae`.
- **`jade.config`** now has a [`model/` sub-package](../config/model/0_module_quickstart.md) (one `{model}_cfg.py` per model + the pooling config family), so model hyperparameters are a validated dataclass like everything else. `get_model` consumes a `model_config` instance (selecting on `config.name`); `config/optimizers/__init__.py` cites `MODEL_REGISTRY` / `get_model` as the convention its own registry mirrors, and the new `MODEL_CONFIG_REGISTRY` / `get_model_config` closes the loop on the config side.
- **`jade.dataset`** produces the `(B, depth, C, T)` tensors EEGNet consumes.
