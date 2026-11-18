# `model/hvEEGNet.py` — `hvEEGNet` and its building blocks

Source: `src/jade/model/hvEEGNet.py`

PyTorch implementation of **hvEEGNet** (Cisotto, Zancanaro, Zoppis & Manzoni, *hvEEGNet: a novel deep learning model for high-fidelity EEG reconstruction*, Frontiers in Neuroinformatics 2024), the **hierarchical** version of [`vEEGNet`](vEEGNet.md), built after the hierarchical VAE of **NVAE** (Vahdat & Kautz, 2020). Built via [`get_model`](__init__.md) from an [`hvEEGNet_config`](../config/model/hvEEGNet_cfg.md), trained by [`jade.training.vae`](../training/vae.md) with the [ELBO](../training/custom_loss_function/elbo.md) (which recognizes the hierarchical output).

**A direct implementation, not a generic hierarchical VAE.** The [original code](https://github.com/jesus-333/Variational-Autoencoder-for-EEG-analysis) built hvEEGNet through a generic `hVAE(encoder_cell_list, decoder_cell_list, config)`; here the three levels are hard-wired. Kept from the original: the vector latent mappings (feedforward and LoRA, `parameters_map_type = 'linear' / 'lora'`), the activation before the latent maps (`use_activation_in_sampling`) and the NVAE extra decoder input `h`. Omitted: the classification branch, `convert_logvar_to_var`, and the `dtw_comparison_2` helper.

Imports `torch`, `torch.nn`, `jade.config.model.get_model_config`, `jade.model.support_functions`, `jade.model.EEGNet.EEGNet` and `jade.model.vEEGNet.same_padding_conv_transpose_2d`.

---

## Architecture

The encoder is a plain [`EEGNet`](EEGNet.md) whose three blocks are run one by one (`_encoder_blocks_forward`), so each block output is available. There is one latent space per encoder block (convolutional by default; see *Latent mappings* below); the index follows the decoder order (deepest first):

| Latent | Encoder block it pairs with | Shape (convolutional, defaults 22 × 1000) | Prior |
|---|---|---|---|
| `z_0` | separable convolution (encoder output) | `(16, 1, 100)` | `N(0, I)` |
| `z_1` | spatial filter | `(16, 1, 1000)` | `p(z_1 | z_0)`, from decoder features |
| `z_2` | temporal filter | `(8, 22, 1000)` | `p(z_2 | z_<2)`, from decoder features |

> The **paper numbers them the other way** (`z1` deepest … `z3` shallowest).

Decoder flow (`_decode`):

```
x = latent_projections[0](z_0)                  # identity (convolutional) or feedforward + reshape (vector)
x = h_decoder_combination(h, x)                 # 1x1 conv on concat; h = user-given / trainable / zeros
x = separable_convolution_transpose(x)          # -> shape of spatial-filter output
for level in (z_1, z_2):
    mu_p, lv_p = prior_maps[l](x)
    dmu, dlv  = posterior_shift_maps[l](encoder_decoder_combinations[l](x, encoder_feature_l))   # training only
    z = reparametrize(mu_p + dmu, lv_p + dlv)   # or (mu_p, lv_p) when generating / ignoring
    x = latent_decoder_combinations[l](x, latent_projections[l](z))    # 1x1 conv on concat
    x = next decoder block (spatial_filter_transpose, then temporal_filter_transpose)
```

The three decoder blocks follow the **paper order** (Table 2, same as the original code): each one is the encoder block run backwards, dropout → upsample (to the shape measured at build time) → activation → BN → transposed conv(s); the last block is BN → transposed conv. So each block outputs the raw conv output, and the total is 8,224 parameters with the defaults (as in the paper). **This differs from [`eegnet_decoder`](vEEGNet.md)** (convs → BN → activation → dropout) on purpose: with that order the priors of `z_1`/`z_2` are computed from already normalized and activated features, and the training diverged to NaN within a few steps (3 seeds out of 4 at lr 0.01, µV data), while the paper order is stable. Geometry is probed at build time exactly as in vEEGNet (`_probe_encoder_geometry`, forward pre-hooks on `encoder.pooling_layers`, eval mode, no grad), so any `T`, any pooling factor, a disabled stage and every `pooling_1d` operator work.

## Building blocks

- `reparametrize(mu, log_var)` — module-level function, `mu + exp(0.5 * log_var) * randn_like`.
- `gaussian_parameters_map(feature_shape, latent_dimension, config)` — (activation, if `use_activation_in_sampling`) → `'convolutional'`: `Conv2d(depth, 2*depth, 1x1)` + `chunk(2)`; `'linear'` / `'lora'`: flatten + two feedforward layers (`nn.Linear` or [`lora_linear`](support_functions.md), optional spectral norm) giving `mu`, `log_var` vectors. The original `sample_layer` parameter maps.
- `latent_projection(feature_shape, latent_dimension, config)` — latent sample → feature map: identity (`'convolutional'`), or feedforward (+ activation) + reshape. **One per level, shared by prior and posterior samples.** The original had a separate projection inside the prior and the posterior sampling layers, so the one used at generation time (prior) was never trained — fixed here.
- `features_combination(depth_1, depth_2, depth_output)` — `Conv2d` 1x1 on `cat([x_1, x_2], dim=1)`; the original `weighted_sum_tensor`. The paper text says "sum", the code (and this port) uses this learned mix.

## The extra input `h` (NVAE Fig. 2)

The deepest decoder level combines the `z_0` feature map with `h` (shape = encoder output) through `h_decoder_combination` (`features_combination`, order `cat([h, z])` as in the original). Resolution in `_get_h`: an `h` passed to `forward` / `generate` wins (a batch axis of 1 is broadcast; a wrong shape raises `ValueError`); else the trainable `nn.Parameter` `self.h` (zeros-initialized, present only if `use_h_in_decoder`); else zeros (the trained hvEEGNet of the paper). Passing `h` to `generate` is the "seed" use.

## Public API

| Method | Returns / does |
|---|---|
| `forward(x, h=None, use_mean=False)` | `(x_reconstructed, mu_list, log_var_list, delta_mu_list, delta_log_var_list, z_list)`. With `use_mean=True` every z is the mean of its (approximate posterior) distribution instead of a sample (deterministic; never for training). `mu_list = [mu_0 (posterior), mu_1 (prior), mu_2 (prior)]`; deltas for levels 1-2; `z_list = [z_0, z_1, z_2]`. First five elements = original output order; append-only. Rejects an input of the wrong geometry. |
| `encode(x)` | `(mu_0, log_var_0)` of the deepest level only (the others depend on the decoder). |
| `reconstruct(x, no_grad=True, use_mean=False)` | `forward(x, use_mean=use_mean)[0]`; sampled by default (as the original), deterministic with `use_mean=True`; does not change train/eval mode. |
| `reconstruct_ignoring_latent_spaces(x, [b0, b1, b2])` | Ignored levels are sampled from their prior instead of the posterior. Flags are **deepest first** (`z_0, z_1, z_2`), unlike the original (encoder order, and ignoring the deepest had no effect there). Ignoring `z_0` samples it from `N(0, I)`. |
| `generate(n_samples=1, z=None, device=None, no_grad=True, h=None)` | Samples `z_0 ~ N(0, I)` (shape `latent_shapes[0]`), then `z_1`, `z_2` from the conditional priors. |
| `debug_shape(input_size)` | Shape printout (run when `print_var = True`). |

Attributes: `latent_shapes` (deepest first), `encoder_outputs_shapes` (encoder order), `upsample_target_shapes`.

## Loss

[`elbo_loss`](../training/custom_loss_function/elbo.md) detects `mu_list` and uses [`kl_divergence_hierarchical`](../training/custom_loss_function/support_loss.md) (works for both convolutional and vector latents). For the paper's training use `reconstruction_type = "soft_dtw"` (or `"block_dtw"`), see [`dtw_reconstruction_loss`](../training/custom_loss_function/support_loss.md) — needs the optional `dtw-loss-functions`.

> **Data scale.** The SoftDTW (γ = 1) needs the signal in µV, as in the original code (which multiplies by 1e6). With the volt values of MNE / moabb (~1e-5) the soft-min is dominated by the number of alignment paths, the reconstruction term becomes a large negative number and the decoder minimizes it without reconstructing. `training/hvEEGNet/train_hvEEGNet.py` applies the conversion (`--scale_factor`, default 1e6).

> The latent spaces are large (with the defaults `z_2` alone is 176 000 elements/trial) and the KL is summed over them, so at initialization the KL term dominates the loss by orders of magnitude (the same happened in the original). Use `beta` / the KL warm-up accordingly.

Checked by `test/custom/hvEEGNet.py`.
