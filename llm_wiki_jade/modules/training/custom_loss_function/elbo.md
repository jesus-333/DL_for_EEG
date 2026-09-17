# `custom_loss_function/elbo.py` — `elbo_loss` and `elbo_terms`

Source: `src/jade/training/custom_loss_function/elbo.py`

The ELBO (Evidence Lower BOund), i.e. the objective of a variational autoencoder. Described by [`elbo_config`](../../config/loss_functions/elbo.md), built from it by [`get_custom_loss_function`](__init__.md), and consumed by [`jade.training.vae_epoch`](../vae_epoch.md).

```
L = alpha * L_R  +  beta * w(e) * L_KL
```

`L_R` measures how far the reconstruction is from the input; `L_KL` is a regularizer pulling the approximate posterior towards the standard normal prior, which is what makes the latent space continuous enough for [`generate`](../../model/vEEGNet.md) to work; `w(e)` is the warm-up multiplier at epoch `e`.

---

## `elbo_terms`

```python
elbo_terms = namedtuple("elbo_terms", ["total", "reconstruction", "kl", "kl_weight", "total_full_kl"])
```

| Field | Meaning |
|---|---|
| `total` | `alpha * reconstruction + beta * kl_weight * kl` — **the tensor to backpropagate**. |
| `reconstruction` | The raw (unweighted) reconstruction term. |
| `kl` | The raw (unweighted) Kullback-Leibler term. |
| `kl_weight` | The warm-up multiplier applied this epoch (a plain float). |
| `total_full_kl` | `alpha * reconstruction + beta * kl`, i.e. the same objective at the **final** KL weight. |

### Why a named tuple

The tempting minimal design is to return a scalar and stash the terms on the module (`self.last_reconstruction_loss`). That is hidden mutable state: it breaks the moment the loss is called for the training set and then for the validation set before being read, it is wrong under gradient accumulation, and it cannot be tested without poking at internals. A named tuple is explicit, still indexable (`terms[0]`), and prints usefully in a traceback. A plain dict was rejected for the lack of attribute access and field validation.

### Why `total_full_kl` exists

While a warm-up is running, **`total` changes meaning at every epoch**: the KL term is literally cheaper at the beginning. Comparing it across epochs — which is exactly what selecting `model_BEST.pth` does — would systematically favour the earliest, low-KL-weight epochs. `total_full_kl` uses the final `beta` at every epoch, so it is comparable from epoch 0, and it costs nothing to compute.

With `kl_warmup_type = "none"` the two are numerically identical, so nothing changes for the default configuration. [`jade.training.vae`](../vae.md) selects the best model on `total_full_kl` and logs both.

> This is visible in `test/custom/vEEGNet_creation_and_training.py`: with a linear warm-up the **optimized** loss goes `3.15 → 4.04 → 4.95` (it grows, by construction) while the reconstruction term goes `3.15 → 2.67` (the model is learning). Reading the first series as "training diverged" would be a mistake.

---

## `class elbo_loss(nn.Module)`

### Why a module

Three concrete reasons, none cosmetic: `train_model` calls `loss_function.to(device)`; `wandb.watch(criterion = ...)` expects a module; and a learned-variance `gaussian_nll`, or a future DTW term, would carry buffers or parameters.

### Call contract

```python
forward(model_output, x) -> elbo_terms
```

The **whole output tuple of the model** comes first and the target second, mirroring `loss_function(predict_label, true_label)` in [`classifier_epoch`](../classifier_epoch.md). Two payoffs: [`vae_epoch`](../vae_epoch.md) stays line-for-line parallel to its classification twin, and the model's output tuple can grow (the vEEGNet classification branch) with **no change here** — `forward` unpacks the first three elements and ignores the rest. A tuple shorter than three elements raises a `ValueError` naming what is expected.

Note that for an autoencoder the target **is** the input, so `vae_epoch` calls `loss_function(model(x), x)`.

`compute_loss(x, x_reconstructed, mu, log_var)` is the explicit entry point for tests and direct callers; `forward` only unpacks and delegates to it.

### KL warm-up state

| Attribute / method | Purpose |
|---|---|
| `current_epoch` | The epoch the training is in (a plain `int`). |
| `kl_weight` | The multiplier for that epoch, recomputed by `set_epoch`. |
| `set_epoch(epoch)` | Declare the epoch and update `kl_weight`. Validates a non-negative int. |

`set_epoch` is called **once per epoch** by [`vae.train_model`](../vae.md) (guarded by `hasattr`, so the cycle also works with a schedule-less loss), not once per batch.

**Why the epoch lives here rather than in the epoch-function signature:** (a) `vae_epoch`'s functions keep argument lists byte-identical to the classification ones, so the two families stay interchangeable; (b) the validation epoch automatically uses the *same* KL weight as the training epoch it follows, which is what makes the two losses comparable — with an extra argument it would have to be threaded through `validation_epoch_function` too, which has a different signature; (c) the schedule is configured in `elbo_config`, so config and behaviour stay in one place.

`current_epoch` and `kl_weight` are plain python attributes and **not buffers**: they are training-cycle state, not model parameters, and must not travel with a `state_dict` or a `.to(device)` call.

The schedule itself lives in [`support_loss.compute_kl_weight`](support_loss.md), so this class holds no schedule logic.
