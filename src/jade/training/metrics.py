"""
Functions to compute classification metrics (accuracy, Cohen's kappa, sensitivity, specificity, F1).

The metrics are computed with :mod:`torchmetrics` (multiclass functional API), so everything stays on tensors and there is no scikit-learn dependency.
Specificity in particular is provided natively by torchmetrics, which avoids the manual (and error prone) one-vs-rest computation of the old code.

The predicted labels are obtained through ``model.classify(x)`` when the model exposes such a method (as :class:`~jade.model.EEGNet.EEGNet_Classifier` does), otherwise they fall back to ``argmax(model(x))``.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Full module imports
import logging

# Specific imports
from typing import Optional, Tuple

# Third party imports
import torch
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_cohen_kappa,
    multiclass_confusion_matrix,
    multiclass_f1_score,
    multiclass_recall,
    multiclass_specificity,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module logger

logger = logging.getLogger(__name__)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Main entry points

def compute_metrics(model, loader, device, num_classes : Optional[int] = None) -> dict :
    """
    Compute the classification metrics of a model over a dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        The (already trained) model. It should expose a ``classify`` method returning the predicted class indices; otherwise ``argmax(model(x))`` is used.
    loader : torch.utils.data.DataLoader
        The dataloader to evaluate on.
    device : str or torch.device
        Device used for the forward passes.
    num_classes : int or None, optional
        Number of classes. If ``None`` it is inferred from the labels (max label + 1).

    Returns
    -------
    dict
        A dict with the scalar metrics ``accuracy``, ``cohen_kappa``, ``sensitivity``, ``specificity`` and ``f1``.
    """

    # Compute the true and predicted labels over the whole loader.
    true_label, predict_label = compute_label(model, loader, device)

    # Turn the labels into the metrics dict.
    computed_metrics = compute_metrics_from_labels(true_label, predict_label, num_classes = num_classes)

    return computed_metrics


def compute_label(model, loader, device) -> Tuple[torch.Tensor, torch.Tensor] :
    """
    Run the model over a loader and return the true and predicted class indices.

    Parameters
    ----------
    model : torch.nn.Module
        The model used to classify the data.
    loader : torch.utils.data.DataLoader
        The dataloader to iterate over.
    device : str or torch.device
        Device used for the forward passes.

    Returns
    -------
    true_label : torch.Tensor
        1D tensor (on CPU, dtype long) with the ground-truth class indices.
    predict_label : torch.Tensor
        1D tensor (on CPU, dtype long) with the predicted class indices.
    """

    # Set the model in evaluation mode and move it to the device.
    model.eval()
    model.to(device)

    true_label_list = []
    predict_label_list = []

    # Disable the gradient computation (we only need the forward pass).
    with torch.no_grad() :
        for sample_data_batch, sample_label_batch in loader :
            # Move the data to the device.
            x = sample_data_batch.to(device)

            # Compute the predicted labels: prefer the model classify method, fall back to argmax of the output.
            if hasattr(model, "classify") and callable(model.classify) :
                tmp_predict_label = model.classify(x)
            else :
                tmp_predict_label = torch.argmax(model(x), dim = 1)

            # Make sure the labels are at least 1D (classify may squeeze a batch of size 1 to a scalar).
            tmp_predict_label = torch.atleast_1d(tmp_predict_label.detach())
            tmp_true_label = torch.atleast_1d(sample_label_batch.detach())

            # Save the true and predicted labels (on CPU).
            true_label_list.append(tmp_true_label.cpu())
            predict_label_list.append(tmp_predict_label.cpu())

    # Concatenate the per-batch labels into a single tensor.
    true_label = torch.cat(true_label_list).long()
    predict_label = torch.cat(predict_label_list).long()

    return true_label, predict_label


def compute_metrics_from_labels(true_label : torch.Tensor, predict_label : torch.Tensor, num_classes : Optional[int] = None) -> dict :
    """
    Compute the classification metrics from the true and predicted labels.

    Parameters
    ----------
    true_label : torch.Tensor
        1D tensor with the ground-truth class indices.
    predict_label : torch.Tensor
        1D tensor with the predicted class indices.
    num_classes : int or None, optional
        Number of classes. If ``None`` it is inferred as ``max(labels) + 1``.

    Returns
    -------
    dict
        A dict with the scalar metrics ``accuracy``, ``cohen_kappa``, ``sensitivity``, ``specificity`` and ``f1``.
    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Determine the number of classes

    num_classes = _infer_num_classes(true_label, predict_label, num_classes)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Compute the metrics
    # The averaging mirrors the old sklearn code: 'micro' for the overall accuracy, 'weighted' for the per-class metrics.

    computed_metrics = dict(
        accuracy    = float(multiclass_accuracy(predict_label, true_label, num_classes = num_classes, average = "micro")),
        cohen_kappa = float(multiclass_cohen_kappa(predict_label, true_label, num_classes = num_classes)),
        sensitivity = float(multiclass_recall(predict_label, true_label, num_classes = num_classes, average = "weighted")),
        specificity = float(multiclass_specificity(predict_label, true_label, num_classes = num_classes, average = "weighted")),
        f1          = float(multiclass_f1_score(predict_label, true_label, num_classes = num_classes, average = "weighted")),
    )

    return computed_metrics

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Confusion matrix (kept separate: it is a matrix, not a scalar, so it is not logged among the scalar metrics)

def compute_confusion_matrix(model, loader, device, num_classes : Optional[int] = None, normalize : Optional[str] = None) -> torch.Tensor :
    """
    Compute the confusion matrix of a model over a dataloader.

    This is provided separately from :func:`compute_metrics` because a matrix is not a scalar and is therefore not meant to be logged alongside the scalar metrics.

    Parameters
    ----------
    model : torch.nn.Module
        The model used to classify the data.
    loader : torch.utils.data.DataLoader
        The dataloader to evaluate on.
    device : str or torch.device
        Device used for the forward passes.
    num_classes : int or None, optional
        Number of classes. If ``None`` it is inferred from the labels.
    normalize : str or None, optional
        Normalization mode forwarded to torchmetrics (``'true'``, ``'pred'``, ``'all'`` or ``None``).

    Returns
    -------
    torch.Tensor
        The ``num_classes`` x ``num_classes`` confusion matrix.
    """

    true_label, predict_label = compute_label(model, loader, device)
    num_classes = _infer_num_classes(true_label, predict_label, num_classes)

    confusion_matrix = multiclass_confusion_matrix(predict_label, true_label, num_classes = num_classes, normalize = normalize)

    return confusion_matrix

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Helper functions

def _infer_num_classes(true_label : torch.Tensor, predict_label : torch.Tensor, num_classes : Optional[int]) -> int :
    """
    Return the number of classes, inferring it from the labels when not provided.

    Parameters
    ----------
    true_label : torch.Tensor
        Ground-truth class indices.
    predict_label : torch.Tensor
        Predicted class indices.
    num_classes : int or None
        A user-provided number of classes, or ``None`` to infer it.

    Returns
    -------
    int
        The number of classes (at least 2, since torchmetrics multiclass metrics require it).
    """

    if num_classes is not None :
        return num_classes

    # Infer as max(label) + 1 over both true and predicted labels.
    max_label = int(torch.max(torch.max(true_label), torch.max(predict_label)).item())
    num_classes = max_label + 1

    # torchmetrics multiclass metrics need at least two classes.
    if num_classes < 2 :
        logger.warning("Inferred a single class from the labels. Forcing num_classes = 2 for the metrics computation.")
        num_classes = 2

    return num_classes
