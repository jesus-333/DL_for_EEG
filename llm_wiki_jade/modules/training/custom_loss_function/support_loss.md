# `custom_loss_function/support_loss.py` — the stateless pieces

Source: `src/jade/training/custom_loss_function/support_loss.py`

Everything the custom losses are built from that is a **pure function of its inputs**. Kept apart from [`elbo.py`](elbo.md) so it can be tested and reasoned about without building a model. Mirrors the role `pooling_1d/support_pooling.py` plays for the pooling family.

Imports `torch` and `torch.nn.functional`.

---

## Reconstruction terms

Each has the signature `(x_reconstructed, x, config = None) -> torch.Tensor` and reads `reduction` from the config (defaulting to `'mean'` when it is `None`, so the functions are usable standalone).

| Function | Registry key | Notes |
|---|---|---|
| `reconstruction_loss_mse` | `"mse"` | `F.mse_loss`. The term used in the original vEEGNet paper, and the default. |
| `reconstruction_loss_l1` | `"l1"` | `F.l1_loss`. Less sensitive to outliers, which on EEG means less dominated by the artifacts. |
| `reconstruction_loss_gaussian_nll` | `"gaussian_nll"` | `F.gaussian_nll_loss` with a **unit variance** (`torch.ones_like`), also reading `gaussian_nll_eps` / `gaussian_nll_full`. |

> **`gaussian_nll` is currently a scaled MSE.** The decoders of the package predict only the *mean* of the output distribution, so with a fixed unit variance this term reduces to a scaled mean squared error plus a constant. It is implemented anyway because it is the **extension point**: a decoder that also predicts a variance only has to change the `variance` argument. This is documented in the function's own docstring so it is not mistaken for a meaningful third option today.

### `class dtw_reconstruction_loss(nn.Module)` — `'soft_dtw'` / `'block_dtw'`

The only **stateful** reconstruction term: wraps `dtw_loss_functions.soft_dtw.soft_dtw` / `dtw_loss_functions.block_dtw.block_dtw` (optional dependency, imported in `__init__` with an actionable `ImportError`). Built from the `elbo_config` (`dtw_*` fields).

- Input `(B, depth, C, T)` is reshaped (contiguous copy — the package's block DTW uses `.view`) into `B * C` sequences `(T, depth)` and processed in **one batched call**, then `(B, C)` → sum over channels (or mean, `dtw_average_channels`) → optional `/ T` (`dtw_average_time_samples`) → batch `reduction`. Defaults reproduce the original hvEEGNet term (sum over channels, mean over batch).
- `mag` / `pysdtw` fix `use_cuda` at construction, so the underlying DTW module is built **lazily per device type** (plain dict, not in the state dict): the same loss works on CPU and GPU.
- Verified against a per-channel loop over the package for `soft_dtw` and both block paths (`T % block_size == 0` optimized, otherwise naive).

`RECONSTRUCTION_LOSS_REGISTRY` + `get_reconstruction_loss_function(name, config = None)` complete the usual pair, raising `ValueError` on an unknown name. Registry values are functions, except the DTW entries, which are the class: the factory instantiates it with `config` (required for them).

---

## `kl_divergence_normal(mu, log_var) -> torch.Tensor`

The closed form of the divergence between the approximate posterior and a standard normal prior, both Gaussian with diagonal covariance:

```
KL = -0.5 * sum_i ( 1 + log(sigma_i^2) - mu_i^2 - sigma_i^2 )
```

summed over every dimension of the latent variable and **averaged over the batch**.

**Key detail:** the axes to sum are computed as `range(1, mu.dim())`, i.e. every axis but the batch one. This is what makes the function valid for **both** latent layouts — a vector latent `(B, d)` and a convolutional one `(B, depth, H, T)`. The original implementation branched on `len(mu.shape) > 2` and hard-coded `[1, 2, 3]`.

Exactly `0` for `mu = 0`, `log_var = 0` (a standard normal posterior), which is what the test asserts for both layouts.

---

## `kl_divergence_shifted_normal(prior_log_var, delta_mu, delta_log_var) -> torch.Tensor`

The divergence of a posterior written as a **shift of its prior** (NVAE, Section 3.2): with `p = N(mu, sigma^2)` and `q = N(mu + delta_mu, sigma^2 * exp(delta_log_var))`,

```
KL(q || p) = -0.5 * sum_i ( 1 + delta_log_var_i - delta_mu_i^2 / sigma_i^2 - exp(delta_log_var_i) )
```

Same axis handling and batch averaging as `kl_divergence_normal`. It does not depend on the prior mean. Same formula as the original `kl_loss_normal_function` with deltas.

## `kl_divergence_hierarchical(mu_list, log_var_list, delta_mu_list, delta_log_var_list) -> torch.Tensor`

KL term of a hierarchical VAE ([`hvEEGNet`](../../model/hvEEGNet.md)): `kl_divergence_normal` on level 0 (standard normal prior) plus `kl_divergence_shifted_normal` on every other level. `mu_list` / `log_var_list` hold the posterior of level 0 followed by the **priors** of the other levels; the delta lists have one element less (level 0 has no shift). Inconsistent lengths raise `ValueError`. Checked against the general two-Gaussian KL in `test/custom/hvEEGNet.py`.

---

## `compute_kl_weight(epoch, warmup_type, warmup_epochs, warmup_start) -> float`

The warm-up (annealing) multiplier applied to the KL term, in `[warmup_start, 1]`.

| `warmup_type` | Result |
|---|---|
| `"none"` | `1.0` at every epoch. |
| `"linear"` | `warmup_start + (1 - warmup_start) * min(1, epoch / warmup_epochs)` |

So the weight reaches exactly `1.0` **at** `epoch == warmup_epochs` and stays there.

**Why a warm-up exists at all:** without it the KL term can dominate from the very first epochs and drive the model into *posterior collapse* — a latent space that carries no information about the input, with the decoder ignoring `z` entirely. Starting with a small KL weight lets the decoder learn to reconstruct first.

A non-positive `warmup_epochs` returns `1.0` rather than dividing by zero (the config forbids that combination, but the function stays safe when called directly). An unknown schedule raises `ValueError`.
