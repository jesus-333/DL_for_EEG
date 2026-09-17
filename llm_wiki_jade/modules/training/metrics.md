# `jade.training.metrics`

Source: `src/jade/training/metrics.py`

Classification metrics computed with the `torchmetrics` **multiclass functional API**. Everything stays on tensors — there is **no scikit-learn dependency**. Specificity is provided natively by torchmetrics, avoiding the error-prone manual one-vs-rest computation of the old code.

Predicted labels come from `model.classify(x)` when the model exposes such a method (as `jade.model.EEGNet.EEGNet_Classifier` does), otherwise from `argmax(model(x), dim=1)`.

A **generative** model has no label to compare, so it gets its own function, [`compute_reconstruction_metrics`](#compute_reconstruction_metricsmodel-loader-device---dict), described at the end of this page.

Imports: `torch`, and from `torchmetrics.functional.classification`: `multiclass_accuracy`, `multiclass_cohen_kappa`, `multiclass_confusion_matrix`, `multiclass_f1_score`, `multiclass_recall`, `multiclass_specificity`. Module logger: `logger = logging.getLogger(__name__)`.

---

## The five scalar metrics

`compute_metrics` / `compute_metrics_from_labels` return a dict of **scalar Python floats** (each torchmetrics tensor is wrapped in `float(...)`):

| Key | torchmetrics function | `average` | Notes |
|---|---|---|---|
| `accuracy` | `multiclass_accuracy` | `"micro"` | Overall accuracy. |
| `cohen_kappa` | `multiclass_cohen_kappa` | — (default) | Agreement corrected for chance. |
| `sensitivity` | `multiclass_recall` | `"weighted"` | Recall = sensitivity; weighted by class support. |
| `specificity` | `multiclass_specificity` | `"weighted"` | True-negative rate; weighted. |
| `f1` | `multiclass_f1_score` | `"weighted"` | Weighted F1. |

The averaging choices deliberately mirror the old sklearn code: `micro` for overall accuracy, `weighted` for the per-class metrics.

---

## `compute_metrics(model, loader, device, num_classes=None) -> dict`

Top-level convenience: runs the model over `loader` (via `compute_label`) and turns the labels into the metrics dict (via `compute_metrics_from_labels`).

| Param | Type | Default | Meaning |
|---|---|---|---|
| `model` | `torch.nn.Module` | — | Trained model; prefers `classify`, falls back to `argmax`. |
| `loader` | `torch.utils.data.DataLoader` | — | Data to evaluate on. |
| `device` | `str` / `torch.device` | — | Device for the forward passes. |
| `num_classes` | `int` / `None` | `None` | If `None`, inferred as `max(label)+1` (see `_infer_num_classes`). |

Returns the five-key metrics dict. Called once per set (train / validation) per epoch by [`classifier.train`](classifier.md), and by [`test`](test_functions.md).

---

## `compute_label(model, loader, device) -> (true_label, predict_label)`

Runs the model over the whole loader and returns the true and predicted class indices.

- Sets `model.eval()` and `model.to(device)`.
- Loops inside `with torch.no_grad():`. For each batch, moves data to `device`; if `hasattr(model, "classify") and callable(model.classify)` uses `model.classify(x)`, else `torch.argmax(model(x), dim=1)`.
- Wraps both predicted and true labels in `torch.atleast_1d(...)` — **guards against `classify` squeezing a batch of size 1 to a scalar**, which would otherwise break `torch.cat`.
- Detaches, moves each batch to **CPU**, appends to per-batch lists, then `torch.cat(...).long()`.

### Returns

| Name | Type | Shape / dtype |
|---|---|---|
| `true_label` | `torch.Tensor` | 1D, CPU, `long` — ground-truth class indices. |
| `predict_label` | `torch.Tensor` | 1D, CPU, `long` — predicted class indices. |

> Note: predictions are class **indices**, not logits/probabilities — so all metrics are computed from hard labels (no probability-based metrics like AUROC here).

---

## `compute_metrics_from_labels(true_label, predict_label, num_classes=None) -> dict`

Computes the five scalar metrics directly from two 1D label tensors. Useful when labels are already available (e.g. aggregated elsewhere) without re-running the model.

| Param | Type | Default | Meaning |
|---|---|---|---|
| `true_label` | `torch.Tensor` | — | 1D ground-truth class indices. |
| `predict_label` | `torch.Tensor` | — | 1D predicted class indices. |
| `num_classes` | `int` / `None` | `None` | Inferred via `_infer_num_classes` when `None`. |

Returns the five-key dict described above.

---

## `compute_confusion_matrix(model, loader, device, num_classes=None, normalize=None) -> torch.Tensor`

Kept **separate** from `compute_metrics` because a matrix is not a scalar and is therefore not logged alongside the scalar metrics. Uses `compute_label` + `_infer_num_classes`, then `multiclass_confusion_matrix(predict_label, true_label, num_classes=..., normalize=...)`.

| Param | Type | Default | Meaning |
|---|---|---|---|
| `model`, `loader`, `device` | — | — | As in `compute_label`. |
| `num_classes` | `int` / `None` | `None` | Inferred when `None`. |
| `normalize` | `str` / `None` | `None` | Forwarded to torchmetrics: `"true"`, `"pred"`, `"all"`, or `None`. |

Returns a `num_classes × num_classes` `torch.Tensor`. Not called by the training loop; it is a standalone analysis helper.

---

## `_infer_num_classes(true_label, predict_label, num_classes) -> int` (private)

Returns `num_classes` unchanged when provided; otherwise infers `max(max(true_label), max(predict_label)) + 1`. If the inference yields `< 2`, it logs a WARNING and **forces `num_classes = 2`**, because torchmetrics multiclass metrics require at least two classes.

> Gotcha: inferring from labels means a set that happens to contain only some of the classes could yield a smaller `num_classes` than the true number — pass `num_classes` explicitly when you know it (note: the training loop does **not** pass it, so it always relies on inference).

---

## Reconstruction metrics

### `compute_reconstruction_metrics(model, loader, device) -> dict`

The metric function of a **generative** model, used by [`jade.training.vae`](vae.md) in place of `compute_metrics`. It needs no label: it measures how far the output of the model is from its input, over a whole dataloader, in one `torch.no_grad()` pass with the model in `eval()` mode.

| Key | Meaning |
|---|---|
| `reconstruction_mse` | Mean squared error between the input and its reconstruction. |
| `reconstruction_mae` | Mean absolute error. |

Both are **per-sample averages**, accumulated as `value * batch_size` and divided by the total, so a short last batch does not skew them.

The reconstruction comes from `model.reconstruct(x)` when the model exposes it (as `jade.model.vEEGNet.vEEGNet` does), otherwise from the first element of the model output — the same fallback idiom `compute_label` uses for `classify`. A batch may be a plain tensor or a `(data, label)` pair; the label is ignored.

> **These are not the reconstruction term of the loss**, even when that term is the MSE. They are computed in evaluation mode over the whole set, so they are comparable across epochs and across runs whatever objective is being optimized (in particular, they are unaffected by a KL warm-up). Like the classification metrics they are gated by `training_config.measure_metrics_during_training`.
