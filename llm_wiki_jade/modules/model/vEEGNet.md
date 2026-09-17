# `model/vEEGNet.py` — `vEEGNet` and its building blocks

Source: `src/jade/model/vEEGNet.py`

PyTorch implementation of **vEEGNet** (Zancanaro, Cisotto, Zoppis & Manzoni, *vEEGNet: learning latent representations to reconstruct EEG raw data via variational autoencoders*), a **variational autoencoder (VAE)** for EEG: the encoder is an [`EEGNet`](EEGNet.md), the decoder mirrors it, and between the two the trial is compressed into a probabilistic latent space. Built via [`get_model`](__init__.md) from a [`vEEGNet_config`](../config/model/vEEGNet_cfg.md), and trained by [`jade.training.vae`](../training/vae.md) with the [ELBO](../training/custom_loss_function/elbo.md).

This is the **first generative model of the package**: `forward` does not return a class score but the reconstruction plus the parameters of the latent distribution, which is why it needs its own loss, its own training cycle and its own `task` marker (see [`model_config`](../config/model/generic.md)).

**Two deliberate omissions** with respect to the original paper and the [original implementation](https://github.com/jesus-333/Variational-Autoencoder-for-EEG-analysis):
- **no classification branch** — only the generative half is implemented, so there is no `classify` method and pairing the model with a classification loss is rejected by [`cross_check_config`](../training/support_training.md);
- **no MBEEGNet multi-branch encoder** — the encoder is always a single EEGNet.

Imports `torch`, `torch.nn`, `jade.config.model.get_model_config`, `jade.model.support_functions` and `jade.model.EEGNet.EEGNet`.

---

## What a VAE is, in one paragraph

An autoencoder compresses its input and reconstructs it. A *variational* autoencoder makes the compressed code **probabilistic**: the encoder does not output a point, it outputs the mean and the variance of a Gaussian, from which a point is drawn. Training minimizes the ELBO, which is a reconstruction error plus a Kullback-Leibler term pulling that Gaussian towards a standard normal. That second term is what makes the latent space *continuous and well covered*, so a point drawn from the prior decodes into a plausible signal — which is what [`generate`](#generaten_samples--1-z--none-device--none-no_grad--true---torchtensor) exploits.

---

## `class same_padding_conv_transpose_2d(nn.Module)`

A stride-1 transposed convolution that **preserves the spatial size**, used to invert the two EEGNet convolutions built with `padding='same'`.

It exists because `nn.ConvTranspose2d` has no `padding='same'` and the usual workaround is unavailable: torch requires `output_padding < max(stride, dilation)`, i.e. `0` at stride 1. With `padding = (k - 1) // 2` the output has size `n` for an **odd** kernel and `n + 1` for an **even** one, so this wrapper drops the trailing row/column of every axis whose kernel is even.

> **Not a corner case.** Both EEGNet temporal defaults are even (`c_kernel_1 = (1, 64)`, `c_kernel_3 = (1, 16)`), so without this the time axis would drift by one at two different places.

---

## `class gaussian_sampling_layer(nn.Module)`

Maps the encoder output to the parameters of the approximate posterior and draws a sample with the **reparametrization trick** (`sigma = exp(0.5 * log_var)`, `z = mu + sigma * randn_like(sigma)`). Written this way the randomness does not depend on the parameters, so the gradient flows through the sampling.

The network predicts `log_var` (and not the variance or the standard deviation) because it is unconstrained: any real value maps to a positive variance, so training cannot produce an invalid distribution.

Two mappings, selected by `config.parameters_map_type`:

| `parameters_map_type` | Maps | Latent variable | `latent_shape` |
|---|---|---|---|
| `'linear'` (default, the paper) | the encoder output is flattened, then two `nn.Linear(n, d)` | a **vector** `(B, d)` | `(hidden_space_dimension,)` |
| `'convolutional'` | two `nn.Conv2d(filter_2, filter_2, (1, 1))` | a **tensor** `(B, filter_2, H_e, T_e)`, i.e. the latent space keeps the time axis | `encoder_output_shape` |

`forward(x)` returns `(z, mu, log_var)`. Two separate `1x1` convolutions are used instead of the original's single depth-doubling convolution followed by `chunk(2, dim=1)`: same parameter count, but each map is explicit and the layer stays symmetric with the linear branch.

> The projection **back** to the feature-map shape belongs to the decoder only. The original implementation did it here *and* in the decoder (a double mapping).

---

## `class eegnet_decoder(nn.Module)`

The mirrored EEGNet. Each encoder block is inverted by the corresponding block here, in reverse order, with transposed convolutions in place of convolutions and an upsampling in place of each pooling stage.

| Stage | Layers |
|---|---|
| Latent input | `nn.Linear(d, n) + activation` then reshape (`'linear'`), or `nn.Identity` (`'convolutional'`) |
| Block A — inverse of the separable convolution | `nn.Upsample(size=target_2)` → `same_padding_conv_transpose_2d(filter_2 → filter_1*D, (1,1))` → `same_padding_conv_transpose_2d(filter_1*D → filter_1*D, c_kernel_3, groups=filter_1*D)` → `BatchNorm2d` → activation → dropout |
| Block B — inverse of the spatial filter | `nn.Upsample(size=target_1)` → `nn.ConvTranspose2d(filter_1*D → filter_1, c_kernel_2, groups=filter_1)` (**unpadded**) → `BatchNorm2d` → activation → dropout |
| Block C — inverse of the temporal filter | `same_padding_conv_transpose_2d(filter_1 → depth_first_layer, c_kernel_1)`, **and nothing after it** |

Three fixes with respect to the original implementation:

1. **Execution order.** Every block runs `upsample → transposed conv(s) → BatchNorm → activation → dropout`. The original wrote its blocks in visually-reversed order inside an `nn.Sequential`, which executes in listed order, so BatchNorm, dropout and activation actually ran *before* the convolutions.
2. **Channel counts are derived, never assumed.** The original only type-checked because `filter_1 * D == filter_2` under the default hyper-parameters; changing `filter_2` broke it. Here every transposed convolution states its own in/out and every `groups` value is legal for any `D` / `filter_1`.
3. **No normalization or activation on the output layer.** A literal mirror would end with `BatchNorm2d(depth_first_layer)`, which constrains the scale of the reconstruction and fights the MSE — an EEG amplitude is not bounded.

**The spatial inverse generalizes.** The encoder's unpadded spatial convolution gives `H_e = C - kh + 1`; an unpadded transposed convolution with the same kernel gives back `H_e + kh - 1 = C` exactly, for any `kh <= C`. So there is no "expand 1 → C" special case and no requirement that `c_kernel_2` span the whole channel axis (checked for `(C, kh)` = (22,22), (22,11), (32,32), (22,1)).

---

## Key fact — how the pooling is inverted (and why)

**Each decoder upsampler is sized from the shape its counterpart in the encoder *received*** (`nn.Upsample(size=...)`), not by upsampling by the pooling factor.

The reason is arithmetic: a pooling stage outputs `T // M` samples, so upsampling by `M` returns `T` only when `M` divides `T`. With the paper's own setup (`T = 1000`, factors 4 and 8) the encoder gives `1000 → 250 → 31` and the naive inversion returns `31 · 8 · 4 = 992` — **eight samples short**. The original implementation had exactly this bug; nothing cropped or padded, and its DTW loss silently absorbed the mismatch, while an MSE would simply throw.

Sizing from the recorded shape also means:
- it holds for **arbitrary** `T` and factors (verified for `T` ∈ {1000, 999, 512, 257} × four factor pairs × two channel counts × two spatial kernel heights — 64 combinations, all exact);
- every [`pooling_1d`](pooling_1d/0_module_quickstart.md) operator works through the same code path, including ones that do not follow an exact floor division (`max` with `ceil_mode`, `dwt` with its wavelet padding);
- a **disabled** stage (negative factor → `nn.Identity`) needs no branch: the recorded input shape equals the output shape, so the resize is an exact no-op.

> **Upsampling is not, and cannot be, the inverse of pooling.** Pooling discards information: `avg`/`blur`/`sinc` low-pass above `π/M`, `dwt` drops the detail sub-band, `max` is non-linear. The upsampling restores the **geometry**; the transposed convolutions learn to repair the **content** (the standard upsample+conv decoder surrogate). For the anti-aliasing ablation this matters: each arm changes how much information reaches the latent space, so the reconstruction errors of two arms compare **different bottlenecks**, not only different filters.

---

## `class vEEGNet(nn.Module)`

### Tensor shapes

- **Input:** 4D `(B, depth_first_layer, C, T)`.
- **Output:** a **4-tuple** `(x_reconstructed, mu, log_var, z)`, with `x_reconstructed.shape == x.shape` exactly.

The order is **append-only** by design: the reconstruction stays at index 0 (so `reconstruct(x) == forward(x)[0]`, the original's contract), and `predicted_label` would land at index 4 when the classification branch is implemented, breaking no existing indexing. `z` is returned although the loss does not need it, because the latent code is the object most analyses of a VAE work on. The [ELBO](../training/custom_loss_function/elbo.md) unpacks the first three elements and ignores the rest.

### The build-time probe

`_probe_encoder_geometry(input_size)` runs **once** in `__init__`, right after the encoder is built and before the decoder is. It attaches a `register_forward_pre_hook` to each of the encoder's two pooling stages (reached through the `EEGNet.pooling_layers` tuple, not by indexing into the `nn.Sequential`), pushes one `torch.randn(input_size)` through the encoder, and records:

- `encoder_output_shape` — sizes the latent maps and the projection back;
- `upsample_target_shapes` — the `(H, T)` each pooling stage receives, i.e. what the decoder restores.

These cannot be computed in closed form, because a stage may be any `pooling_1d` operator or be disabled, so they are **measured**.

> **The probe runs in `eval()` mode inside `torch.no_grad()`, restoring the previous mode in a `finally`.** Both matter: a random probe must not pollute the BatchNorm running statistics, and `BatchNorm2d` raises *"Expected more than 1 value per channel"* in training mode when the probe (batch 1) has a single element per channel — reachable for a short `T` with large pooling factors. `EEGNet_Classifier.compute_number_of_neurons` does not take this precaution; here the probe is on the critical path of the whole decoder geometry, so it does.

### Input-geometry guard

Because the decoder geometry is fixed at build time, `forward` rejects a tensor whose shape is not `(batch, depth_first_layer, C, T)` with a `ValueError` naming the expected geometry. Without it, the `'convolutional'` mapping (which would otherwise be length-agnostic) would silently return a `T`-long reconstruction of a different-length input.

### Public methods

| Method | Purpose |
|---|---|
| `forward(x)` | `(x_reconstructed, mu, log_var, z)`. |
| `encode(x)` | `(mu, log_var)` — the parameters of the approximate posterior. |
| `decode(z)` | Map a point of the latent space back to a signal. |
| `reconstruct(x, no_grad=True)` | `forward(x)[0]`. Stochastic (the latent variable is sampled), and it does **not** change the mode of the model. |
| `generate(n_samples=1, z=None, device=None, no_grad=True)` | Draw latent variables from the prior and decode them. |
| `debug_shape(input_size)` | Print the shape after each stage, ending with an explicit `input shape == output shape` line. |

`generate` fixes three defects of the original, whose version was `z = torch.randn(1, hidden_space)` followed by `decoder(z)`: it samples `torch.randn(n_samples, *self.latent_shape)` so it is **correct for both latent mappings** (the original was hard-wired to the vector one and would have failed on the convolutional latent), it defaults the device to `next(self.parameters()).device` (the original ignored the device entirely), and it takes a sample count and an optional explicit `z` (for latent traversal / interpolation).

---

## Parameter counts (default 22×1000 geometry)

| Component | `'linear'` (d = 64) | `'convolutional'` |
|---|---|---|
| Encoder | 1 456 | 1 456 |
| Sampling | 63 616 | 544 |
| Decoder | 33 664 | 1 728 |

The `'linear'` mapping is dominated by the two `Linear(496, 64)` maps and the `Linear(64, 496)` projection back — which is also why it, and not the convolutional one, fixes `T` at build time.

---

## Gotchas

- **The model is stochastic even in `eval()` mode**: the latent variable is sampled at every forward pass, so two reconstructions of the same trial differ. `mu` and `log_var` *are* deterministic in `eval()`.
- **It must be paired with a generative loss.** `vEEGNet_config.task = "generative"`, and `cross_check_config` rejects `cross_entropy` / `nll`.
- **`veegnet_classifier` is deliberately not a registered name** — see [`__init__.md`](__init__.md).
- Checked end to end by `test/custom/vEEGNet.py` (architecture, the 64-geometry sweep, both latent mappings, reparametrization, `reconstruct`/`generate`, gradient flow, every pooling operator, the config round-trip and the guards) and by `test/custom/vEEGNet_creation_and_training.py` (the training cycle).
