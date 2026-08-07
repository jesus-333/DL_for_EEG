"""
Single-epoch functions for the classification training.

This module implements one epoch of the training process (:func:`train_epoch_function`, with the backward pass) and one epoch of validation (:func:`validation_epoch_function`, without the backward pass).
Both accumulate the loss as a per-sample weighted average (``loss * batch_size`` summed, then divided by the number of samples) so that the value does not depend on how the last, possibly smaller, batch is sized.
The full training cycle that calls these functions lives in :mod:`jade.training.classifier`.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Full module imports
import logging

# Third party imports
import torch

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module logger

logger = logging.getLogger(__name__)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Epoch functions

def train_epoch_function(model, train_loader, loss_function, optimizer, device, log_dict = None, print_var = True) :
    """
    Run a single training epoch (forward + backward pass) over ``train_loader``.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train. It is set in training mode inside the function.
    train_loader : torch.utils.data.DataLoader
        Dataloader yielding ``(data, label)`` batches.
    loss_function : torch.nn.Module
        The loss module. Note that :class:`torch.nn.CrossEntropyLoss` expects logits while :class:`torch.nn.NLLLoss` expects log-probabilities, so it must match the model output.
    optimizer : torch.optim.Optimizer
        The optimizer used to update the model parameters.
    device : str or torch.device
        Device the data is moved to before the forward pass.
    log_dict : dict, optional
        If given, the training loss is stored under the key ``'train_loss'``, by default None.
    print_var : bool, optional
        Kept for signature compatibility with the rest of the training code; the per-epoch logging is done by the caller (:func:`~jade.training.classifier.train`), by default True.

    Returns
    -------
    float
        The average training loss over the epoch (per-sample average).
    """

    # Set the model in training mode.
    model.train()

    # Variable to accumulate the loss.
    train_loss = 0

    for sample_data_batch, sample_label_batch in train_loader :
        # Move the data to the training device.
        x = sample_data_batch.to(device)
        true_label = sample_label_batch.to(device)

        # Zero the gradients accumulated in the previous step.
        optimizer.zero_grad()

        # Network forward pass.
        predict_label = model(x)

        # Loss evaluation.
        batch_train_loss = loss_function(predict_label, true_label)

        # Backward pass and optimization step.
        batch_train_loss.backward()
        optimizer.step()

        # Accumulate the loss, weighted by the batch size (so the last, smaller batch is handled correctly).
        train_loss += batch_train_loss.item() * x.shape[0]

    # Compute the final per-sample average loss.
    train_loss = train_loss / len(train_loader.sampler)

    # (OPTIONAL) Save the loss in the log dict used by wandb.
    if log_dict is not None :
        log_dict["train_loss"] = float(train_loss)

    return train_loss


def validation_epoch_function(model, validation_dataloader, loss_function, device, log_dict = None) :
    """
    Run a single validation epoch (forward pass only, no gradient) over ``validation_dataloader``.

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate. It is set in evaluation mode inside the function.
    validation_dataloader : torch.utils.data.DataLoader
        Dataloader yielding ``(data, label)`` batches.
    loss_function : torch.nn.Module
        The loss module (must match the model output, as for :func:`train_epoch_function`).
    device : str or torch.device
        Device the data is moved to before the forward pass.
    log_dict : dict, optional
        If given, the validation loss is stored under the key ``'validation_loss'``, by default None.

    Returns
    -------
    float
        The average validation loss over the epoch (per-sample average).
    """

    # Set the model in evaluation mode.
    model.eval()

    # Variable to accumulate the loss.
    validation_loss = 0

    # Disable the gradient computation.
    with torch.no_grad() :
        for sample_data_batch, sample_label_batch in validation_dataloader :
            # Move the data to the device.
            x = sample_data_batch.to(device)
            true_label = sample_label_batch.to(device)

            # Network forward pass.
            predict_label = model(x)

            # Loss evaluation.
            batch_validation_loss = loss_function(predict_label, true_label)

            # Accumulate the loss, weighted by the batch size.
            validation_loss += batch_validation_loss.item() * x.shape[0]

        # Compute the final per-sample average loss.
        validation_loss = validation_loss / len(validation_dataloader.sampler)

        # (OPTIONAL) Save the loss in the log dict used by wandb.
        if log_dict is not None :
            log_dict["validation_loss"] = float(validation_loss)

    return validation_loss
