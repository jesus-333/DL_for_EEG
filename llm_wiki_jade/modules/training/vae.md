# `training/vae.py` — the full training cycle of a generative model

Source: `src/jade/training/vae.py`

The training cycle of a **generative** model, i.e. one that reconstructs its input instead of classifying it ([`vEEGNet`](../model/vEEGNet.md)). The counterpart of [`classifier.py`](classifier.md), offering the **same four entry points, with the same signatures and the same return value**, so moving a run from a classifier to a variational autoencoder only means importing another module.

| Entry point | Role |
|---|---|
| `train(training_config, model_config, train_dataset, validation_dataset=None, wandb_model_artifact=None)` | Standard entry point: checks both configs, cross-checks them, builds the model with `get_model`, delegates to `train_model`. |
| `train_model(training_config, model, train_dataset, validation_dataset=None, wandb_model_artifact=None)` | The whole cycle on an already-built model. Returns `(model, computed_metrics_during_training)`. |
| `wandb_train(training_config, model_config, train_dataset, validation_dataset=None, dataset_config=None)` | Mirrors `train` for a tracked run. |
| `wandb_train_model(training_config, model, train_dataset, validation_dataset=None, model_config=None, dataset_config=None)` | Mirrors `train_model`; delegates to [`wandb_utils.run_training_with_wandb`](wandb_utils.md). |

The loss must be a generative one (the [ELBO](custom_loss_function/elbo.md)), which [`cross_check_config`](support_training.md) enforces.

---

## What is identical to the classification cycle

Config checks (through the shared `support_training.require_config_instance`), seeding, the two dataloaders, `model.to(device)`, resume-from-weights/optimizer, the loss/optimizer/scheduler factories, the checkpoint folder, the **periodic / BEST / END / A-B backup** checkpoint logic, the learning-rate logging and the scheduler step, and the wandb calls (now delegated to `wandb_utils`).

> **Scoped duplication, on purpose.** Roughly 60 lines are copied from `classifier.train_model`. The natural follow-up — `support_training.build_training_objects(...)` + `load_past_state(...)` — would collapse it without touching any public signature, and is worth doing separately.

---

## The four differences

### 1. The loss is told which epoch it is in

```python
if hasattr(loss_function, "set_epoch") : loss_function.set_epoch(epoch)
```

the first statement of the epoch body. This drives the KL warm-up **without changing any epoch-function signature**, and the `hasattr` guard keeps the cycle usable with a schedule-less loss.

### 2. Every term of the loss is tracked, unconditionally

`train_loss`, `train_recon_loss`, `train_kl_loss`, `train_loss_full_kl`, their `validation_` counterparts and `kl_weight` all land in `computed_metrics_during_training`. They are the **loss decomposition**, not metrics, and they cost nothing beyond the forward pass that already happened, so they are not gated by `measure_metrics_during_training`.

### 3. The metrics are reconstruction metrics

The whole `metrics.compute_metrics` block is **dropped**: it calls `model.classify` / `argmax` and the torchmetrics multiclass API, none of which means anything for a VAE. In its place, gated by `measure_metrics_during_training` exactly as before, [`metrics.compute_reconstruction_metrics`](metrics.md) yields `reconstruction_mse` and `reconstruction_mae`.

This keeps the config flag meaningful instead of silently ignored, and keeps `vae.train_model` reviewable side by side with `classifier.train_model`.

### 4. The best model is selected on `loss_full_kl`

Not on the optimized loss. While a warm-up is running the optimized value **changes meaning at every epoch** (the KL term is cheaper at the beginning), so comparing it across epochs would systematically favour the earliest ones. `loss_full_kl` uses the final `beta` at every epoch and is comparable from epoch 0.

With `kl_warmup_type = "none"` the two are numerically identical, so nothing changes for the default configuration. Both are recorded, and the per-epoch log prints both — otherwise a reader would see `Validation loss = 0.365` next to `(BEST) = 0.404` and conclude, wrongly, that the best model is worse.

---

## Two bugs of `classifier.py` deliberately not reproduced

- **`best_accuracy_val` read outside the branch that assigns it.** In `classifier.py` it is only set inside `if new_best_found:` yet read whenever metrics are on; safe today only because epoch 0 always sets a new best, but it breaks the moment a resume path or an early exit is added. Here **every tracker is initialized before the loop**.
- **The `if epoch == 0: [x] else: .append(x)` idiom.** Replaced by a small `_accumulate` helper using `setdefault(key, []).append(value)`, which is shorter and does not assume the first epoch is epoch 0.

---

## Returned metrics

`computed_metrics_during_training` maps each tracked quantity to the list of its per-epoch values:

| Keys | When |
|---|---|
| `train_loss`, `train_recon_loss`, `train_kl_loss`, `train_loss_full_kl`, `kl_weight` | always |
| the `validation_*` counterparts | when a validation set is given |
| `reconstruction_mse_train`, `reconstruction_mae_train` (+ `_validation`) | when `measure_metrics_during_training` |

---

## Gotcha — the optimized loss is *expected* to grow under a warm-up

With a linear warm-up, `train_loss` rises epoch after epoch while the model is learning perfectly well, because the KL term is being weighted more each time. Look at `train_recon_loss` (or `train_loss_full_kl`) to judge progress. `test/custom/vEEGNet_creation_and_training.py` asserts exactly this: the optimized loss goes `3.15 → 4.04 → 4.95` while the reconstruction term goes `3.15 → 2.67`.
