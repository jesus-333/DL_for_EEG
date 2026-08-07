# `jade.training.test_functions`

Source: `src/jade/training/test_functions.py`

A generic evaluation function for an already-trained classifier. `test` computes the loss and (optionally) the classification metrics on any dataset and returns them in a `log_dict` ready to be logged. It does **not** log to wandb itself — the caller is expected to.

It reuses [`validation_epoch_function`](classifier_epoch.md) for the forward pass/loss and [`metrics.compute_metrics`](metrics.md) for the metrics.

Module logger: `logger = logging.getLogger(__name__)`.

---

## `test(training_config, model, test_dataset, label="test")`

### Parameters

| Name | Type | Default | Meaning |
|---|---|---|---|
| `training_config` | `training_config` or `dict` | — | Config; a dict is promoted via `from_dict`. Only a few fields are used: `batch_size`, `device`, `dataloader`, `loss`, `measure_metrics_during_training`. |
| `model` | `torch.nn.Module` | — | Model to evaluate. |
| `test_dataset` | `torch.utils.data.Dataset` | — | Data to evaluate on; each item is a `(data, label)` pair. |
| `label` | `str` | `"test"` | Name used to key the loss and metrics in the output. E.g. `label="test"` → loss under `"test_loss"`, metric `accuracy` under `"accuracy_test"`. |

### Returns

`(test_loss, log_dict)`:

- `test_loss : float` — per-sample average loss over the test set (from `validation_epoch_function`).
- `log_dict : dict` — the loss under `"{label}_loss"` and, when metrics are on, each metric under `"{metric}_{label}"`. Ready to be logged by the caller.

### Flow

1. Promote dict → `training_config`; read `device`.
2. Build the loader: `support_training.get_dataloader(test_dataset, batch_size, dataloader)`.
3. `model.to(device)`.
4. Build the loss (same `loss` config as training) via `support_training.get_loss_function` and move it to `device`.
5. `log_dict = {}`; `test_loss = validation_epoch_function(model, test_loader, loss_function, device, log_dict)`.
6. **Rename the loss key:** `validation_epoch_function` writes `"validation_loss"`, so `test` does `del log_dict["validation_loss"]` and then `log_dict["{label}_loss"] = test_loss`.
7. If `measure_metrics_during_training`, compute `metrics.compute_metrics(model, test_loader, device)` and merge them into `log_dict` with `support_training.update_log_dict_metrics(..., label=label)` (suffix each key with `label`).
8. Return `(test_loss, log_dict)`.

### Side effects

- Sets the model to eval mode and moves it to `device` (inherited from `validation_epoch_function` + explicit `.to(device)`).
- Runs two forward passes over the data when metrics are enabled: one in `validation_epoch_function` (loss) and one in `compute_metrics` (labels). No gradients, no weight updates, **no wandb calls, no file writes**.

### Design notes

- The `label` parameter is what makes this function **generic**: the same code can score a `"test"` set, or a `"server"` set for a future federated-learning central-server evaluation, just by changing the suffix.
- Metric gating reuses the training flag `measure_metrics_during_training` rather than a dedicated test flag.
