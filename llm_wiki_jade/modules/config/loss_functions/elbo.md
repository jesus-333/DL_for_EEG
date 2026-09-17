# `config/loss_functions/elbo.py` — `elbo_config`

Source: `src/jade/config/loss_functions/elbo.py`

Configuration of the **ELBO** (Evidence Lower BOund), the loss used to train a variational autoencoder. It is the first member of this family that is **not** a mirror of a `torch.nn` loss: the implementation lives in [`jade.training.custom_loss_function.elbo`](../../training/custom_loss_function/elbo.md).

The loss it describes is

```
L = alpha * L_R  +  beta * w(e) * L_KL
```

where `L_R` is the reconstruction term, `L_KL` is the Kullback-Leibler divergence between the approximate posterior and a standard normal prior, and `w(e)` is the warm-up multiplier at epoch `e`.

---

## Why the config lives here and not next to its implementation

The instructions put the *loss module* under `jade.training`; the config stays in `jade.config.loss_functions`, and that is not a style preference — it is the only placement that works:

- `training_config.__post_init__` promotes its `loss` field through `get_loss_config`, which only knows [`LOSS_CONFIG_REGISTRY`](__init__.md). A config outside that registry **cannot be loaded from a TOML `[loss]` table at all**, which would make the whole feature unreachable from `training_vae.toml`.
- Registering it from `jade.training` would make `jade.config` depend on `jade.training`, inverting the package's one-way `training → config` dependency and creating an import cycle (`support_training` imports `jade.config.loss_functions` at module level).
- `training_config.to_dict()` is what gets logged to wandb and round-tripped; a torch-importing config would drag torch into every serialization path.

---

## Fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | `"elbo"` | Discriminator, **frozen** in `__post_init__`. |
| `reconstruction_type` | `str` | `"mse"` | The reconstruction term. One of `RECONSTRUCTION_TYPES = ("mse", "l1", "gaussian_nll")`. `mse` is the term used in the original vEEGNet paper. |
| `alpha` | `float` | `1.0` | Weight of the reconstruction term. Must be `>= 0`. |
| `beta` | `float` | `1.0` | Weight of the KL term. Must be `>= 0`. |
| `reduction` | `str` | `"mean"` | How both terms are reduced over the batch. One of `REDUCTIONS = ("mean", "sum")`. |
| `kl_warmup_type` | `str` | `"none"` | Warm-up schedule of the KL weight. One of `KL_WARMUP_TYPES = ("none", "linear")`. |
| `kl_warmup_epochs` | `int` | `0` | Duration of the warm-up, i.e. the first epoch at which the weight reaches 1. |
| `kl_warmup_start` | `float` | `0.0` | KL weight at epoch 0, ramped to 1. Must be in `[0, 1]`. |
| `gaussian_nll_eps` | `float` | `1e-6` | Lower bound on the variance. `gaussian_nll` only. |
| `gaussian_nll_full` | `bool` | `False` | Include the constant term. `gaussian_nll` only. |

### Notable values

- **`beta = 0`** removes the regularization entirely, i.e. it turns the model into a plain (non-variational) autoencoder. A value different from 1 is the so-called **beta-VAE**. Zero is allowed for both weights because `alpha = 0` / `beta = 0` are degenerate but legitimate ablations.
- **`reduction = "none"` is rejected** (contrary to the classification losses, which accept it): a non-reduced ELBO is not a scalar and cannot be backpropagated.

### What is deliberately absent

**No `class_weight`, no `ignore_index`.** They describe a per-class rescaling, which has no meaning for a reconstruction objective. Their absence is precisely why `get_loss_function` had to be restructured: it used to read the class-weight field *before* dispatching, which would raise an `AttributeError` on an `elbo_config` before reaching any branch. See [`support_training.md`](../../training/support_training.md).

---

## Validation gotcha — an active warm-up needs a duration

```python
elbo_config(kl_warmup_type = "linear")          # ValueError: kl_warmup_epochs must be > 0
```

The config **raises** instead of silently downgrading to `"none"`, so a typo in a TOML cannot quietly train a different objective. A warm-up *longer than the whole run* is a different matter: it is legitimate for a deliberately truncated run, so `cross_check_config` only logs a warning.

---

## Skeleton

`src/jade/config/default_config/train/loss/elbo.toml`, **header-less** like `nll.toml` and `cross_entropy.toml` — keeping a `[loss]` header re-nests it a level too deep and raises `unknown loss name ''`. The full `training_vae.toml` embeds the same keys under an inline `[loss]` section.
