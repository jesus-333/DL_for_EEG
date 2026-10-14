# `jade.training.classifier_epoch`

Source: `src/jade/training/classifier_epoch.py`

The single-epoch primitives for classification training: one training epoch (with the backward pass) and one validation epoch (forward only, no gradient). Both are called by the epoch loop in [`classifier.train_model`](classifier.md) (`train` only builds the model and delegates; the epoch loop itself lives in `train_model`).

## Shared design: per-sample weighted loss

Both functions accumulate the loss as `batch_loss.item() * batch_size` summed over batches, then divide by `len(loader.sampler)` (the number of samples). This makes the reported average **independent of how the last, possibly smaller, batch is sized** — a plain mean over batches would over-weight a short final batch.

> Gotcha: the denominator is `len(loader.sampler)`, i.e. the sample count of the sampler, **not** `len(loader)` (the batch count) nor `len(dataset)`.

Module logger: `logger = logging.getLogger(__name__)` (declared but not used inside these functions — per-epoch logging is the caller's job).

---

## `train_epoch_function(model, train_loader, loss_function, optimizer, device, log_dict=None, print_var=True)`

Runs one training epoch (forward + backward) over `train_loader`.

### Parameters

| Name | Type | Default | Meaning |
|---|---|---|---|
| `model` | `torch.nn.Module` | — | Model to train. Set to **train mode** (`model.train()`) inside the function. |
| `train_loader` | `torch.utils.data.DataLoader` | — | Yields `(data, label)` batches. |
| `loss_function` | `torch.nn.Module` | — | Loss module. Must match the model output: `CrossEntropyLoss` expects logits, `NLLLoss` expects log-probabilities. |
| `optimizer` | `torch.optim.Optimizer` | — | Updates the model parameters. |
| `device` | `str` or `torch.device` | — | Batches are `.to(device)` before the forward pass. |
| `log_dict` | `dict` | `None` | If given, the epoch loss is stored under key `"train_loss"` (as a `float`). |
| `print_var` | `bool` | `True` | **Kept only for signature compatibility**; not used here — the caller does the per-epoch logging. |

### Returns

`float` — the per-sample average training loss over the epoch.

### Per-batch body

For each `(sample_data_batch, sample_label_batch)`: move both to `device`; `optimizer.zero_grad()`; `predict_label = model(x)`; `batch_train_loss = loss_function(predict_label, true_label)`; `.backward()`; `optimizer.step()`; accumulate `train_loss += batch_train_loss.item() * x.shape[0]`. After the loop, `train_loss /= len(train_loader.sampler)` and, if `log_dict` is not `None`, `log_dict["train_loss"] = float(train_loss)`.

### Side effects

- Sets model to train mode; mutates model params (via optimizer); mutates `log_dict` in place when provided.

---

## `validation_epoch_function(model, validation_dataloader, loss_function, device, log_dict=None)`

Runs one validation epoch (forward only, no gradient) over `validation_dataloader`.

### Parameters

| Name | Type | Default | Meaning |
|---|---|---|---|
| `model` | `torch.nn.Module` | — | Model to evaluate. Set to **eval mode** (`model.eval()`) inside. |
| `validation_dataloader` | `torch.utils.data.DataLoader` | — | Yields `(data, label)` batches. |
| `loss_function` | `torch.nn.Module` | — | Loss module (must match the model output, as above). |
| `device` | `str` or `torch.device` | — | Batches are `.to(device)` before the forward pass. |
| `log_dict` | `dict` | `None` | If given, the epoch loss is stored under key `"validation_loss"` (as a `float`). |

### Returns

`float` — the per-sample average validation loss over the epoch.

### Body

`model.eval()`, then the whole loop runs inside `with torch.no_grad():` (no gradients, no optimizer). Same per-sample-weighted accumulation and same `len(sampler)` division as the training function. If `log_dict` is not `None`, sets `log_dict["validation_loss"] = float(validation_loss)`.

### Side effects

- Sets model to eval mode; mutates `log_dict` in place when provided. Does **not** touch the optimizer or model weights.

### Reuse note

`validation_epoch_function` is reused by [`test_functions.test`](test_functions.md) for the generic evaluation loss. Because it hard-codes the key `"validation_loss"`, `test` deletes that key afterward and rewrites it under `"{label}_loss"`.
