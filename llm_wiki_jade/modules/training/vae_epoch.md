# `training/vae_epoch.py` — the single-epoch functions of a generative model

Source: `src/jade/training/vae_epoch.py`

One epoch of training (`train_epoch_function`, with the backward pass) and one of validation (`validation_epoch_function`, without it), for a model that reconstructs its input. The counterpart of [`classifier_epoch`](classifier_epoch.md); the full cycle that calls them is [`vae`](vae.md).

Like its classification twin it imports only `logging` and `torch` — **no internal `jade` import**.

---

## Signatures — deliberately identical to the classification ones

```python
train_epoch_function(model, train_loader, loss_function, optimizer, device, log_dict=None, print_var=True) -> dict
validation_epoch_function(model, validation_dataloader, loss_function, device, log_dict=None) -> dict
```

Byte-identical argument lists, including the unused `print_var` (kept for signature compatibility exactly as `classifier_epoch` documents; the per-epoch logging is done by the caller). This is what keeps the two families interchangeable, and it is why the epoch index is **not** an argument: the KL warm-up is carried as state by the loss itself (see [`elbo_loss.set_epoch`](custom_loss_function/elbo.md#kl-warm-up-state)).

---

## The three differences with a classification epoch

1. **The target is the input.** An autoencoder reconstructs what it receives, so the loss is called as `loss_function(model_output, x)` and the label yielded by the dataloader is **not used**.
2. **A batch may or may not carry a label.** `_unpack_batch` accepts a plain tensor as well as a `(data, label)` pair, because `EEG_Dataset_ChWi` has a `return_only_data` flag. The normal `EEG_Dataset` always yields the pair, and the label is simply ignored.
3. **The return value is a dict, not a float**, because the loss of a VAE is intrinsically multi-term.

---

## Returned terms

`LOSS_TERMS` maps each accumulated name to the field of [`elbo_terms`](custom_loss_function/elbo.md) it comes from:

| Returned key | `elbo_terms` field | Meaning |
|---|---|---|
| `loss` | `total` | The value that was optimized. |
| `recon_loss` | `reconstruction` | The raw reconstruction term. |
| `kl_loss` | `kl` | The raw Kullback-Leibler term. |
| `loss_full_kl` | `total_full_kl` | The same objective at the final KL weight — the comparable one. |
| `kl_weight` | `kl_weight` | The warm-up multiplier of this epoch. |

Every term is accumulated as `value.item() * x.shape[0]` and divided by `len(loader.sampler)`, the same **per-sample weighting** the classification epoch uses, so a short last batch does not skew the average.

> `.item()` is not cosmetic: the original implementation accumulated live tensors, keeping a whole epoch's worth of them (and their memory) alive on the GPU.

---

## `log_dict` keys

`_update_log_dict` writes `train_<term>` / `validation_<term>`, **except** `kl_weight`, which is a property of the epoch rather than of a set and is therefore stored unprefixed (both sets share it).

**`train_loss` and `validation_loss` keep exactly the names the classification epoch uses**, so the wandb exporter in [`cli/wandb.py`](../cli/wandb.md) and any existing dashboard keep working unchanged.

---

## Gotcha

**The model is still stochastic in `eval()` mode.** The latent variable is sampled at every forward pass, so two validation epochs over the same data give slightly different values. Only `mu` and `log_var` are deterministic.
