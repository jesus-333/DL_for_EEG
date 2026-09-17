"""
Single-epoch functions for the training of a generative (VAE) model.

This module implements one epoch of the training process (:func:`train_epoch_function`, with the backward pass) and one epoch of validation (:func:`validation_epoch_function`, without the backward pass).
It is the counterpart of :mod:`jade.training.classifier_epoch` for a model that reconstructs its input instead of classifying it, and it has deliberately the same signatures, so the two families stay interchangeable.

There are only two differences with the classification epoch:

* the target of the loss is the input itself (an autoencoder reconstructs what it receives), so the label yielded by the dataloader is not used;
* the loss of a VAE is made of several terms (see :class:`~jade.training.custom_loss_function.elbo.elbo_terms`), so these functions return a dict of averaged terms instead of a single float.

As in the classification epoch every term is accumulated as a per-sample weighted average (``value * batch_size`` summed, then divided by the number of samples) so that it does not depend on how the last, possibly smaller, batch is sized.
The full training cycle that calls these functions lives in :mod:`jade.training.vae`.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Full module imports
import logging

# Third party imports
import torch

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module logger and constants

logger = logging.getLogger(__name__)

# Names of the loss terms accumulated over an epoch, mapped to the field of the loss output they come from.
# 'loss' keeps the name the classification epoch uses for its own loss, so the logs of the two families stay consistent.
LOSS_TERMS = {
    "loss"          : "total",
    "recon_loss"    : "reconstruction",
    "kl_loss"       : "kl",
    "loss_full_kl"  : "total_full_kl",
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Support functions

def _unpack_batch(batch) :
    """
    Extract the data from a batch, whatever the dataset yields.

    A dataset may yield a plain tensor or a ``(data, label)`` pair (see the ``return_only_data`` flag of :class:`~jade.dataset.dataset_time.EEG_Dataset_ChWi`).
    A generative model does not use the label, so both are accepted.

    Parameters
    ----------
    batch : torch.Tensor or sequence
        A batch yielded by the dataloader.

    Returns
    -------
    torch.Tensor
        The data of the batch.
    """

    if isinstance(batch, (tuple, list)) :
        return batch[0]

    return batch


def _update_log_dict(log_dict : dict, epoch_terms : dict, label : str) -> None :
    """
    Copy the terms of an epoch into the log dict used by wandb, prefixing each key with the set it comes from.

    Parameters
    ----------
    log_dict : dict
        The dict to update.
    epoch_terms : dict
        The averaged terms of the epoch.
    label : str
        The name of the set, i.e. ``'train'`` or ``'validation'``.
    """

    for term_name, term_value in epoch_terms.items() :
        # The weight of the KL term is a property of the epoch, not of the set, so it is not prefixed (both sets share it).
        if term_name == "kl_weight" :
            log_dict["kl_weight"] = term_value
        else :
            log_dict[f"{label}_{term_name}"] = term_value

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Epoch functions

def train_epoch_function(model, train_loader, loss_function, optimizer, device, log_dict = None, print_var = True) -> dict :
    """
    Run a single training epoch (forward + backward pass) over ``train_loader``.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train. It is set in training mode inside the function. Its forward must return the reconstruction and the parameters of the latent distribution (e.g. :class:`~jade.model.vEEGNet.vEEGNet`).
    train_loader : torch.utils.data.DataLoader
        Dataloader yielding either ``(data, label)`` batches or plain data batches. The label, when present, is ignored.
    loss_function : torch.nn.Module
        The loss module, e.g. :class:`~jade.training.custom_loss_function.elbo.elbo_loss`. It is called as ``loss_function(model_output, x)`` and it must return the terms of the loss, the field ``total`` being the one to backpropagate.
    optimizer : torch.optim.Optimizer
        The optimizer used to update the model parameters.
    device : str or torch.device
        Device the data is moved to before the forward pass.
    log_dict : dict, optional
        If given, each averaged term is stored under the key ``'train_<term>'``, by default None.
    print_var : bool, optional
        Kept for signature compatibility with the rest of the training code; the per-epoch logging is done by the caller (:func:`~jade.training.vae.train_model`), by default True.

    Returns
    -------
    dict
        The averaged terms of the epoch: ``loss`` (the value that was optimized), ``recon_loss``, ``kl_loss``, ``loss_full_kl`` (the same objective with the final weight of the KL term) and ``kl_weight``.
    """

    # Set the model in training mode.
    model.train()

    # Variables to accumulate each term of the loss.
    accumulated_terms = {term_name : 0.0 for term_name in LOSS_TERMS}
    kl_weight = 1.0

    for sample_batch in train_loader :
        # Move the data to the training device. Note that the label (when the dataset yields one) is not used: the target of an autoencoder is its own input.
        x = _unpack_batch(sample_batch).to(device)

        # Zero the gradients accumulated in the previous step.
        optimizer.zero_grad()

        # Network forward pass.
        model_output = model(x)

        # Loss evaluation.
        batch_loss_terms = loss_function(model_output, x)

        # Backward pass and optimization step.
        batch_loss_terms.total.backward()
        optimizer.step()

        # Accumulate every term, weighted by the batch size (so the last, smaller batch is handled correctly).
        # Note the item() call: keeping the tensors would keep them (and their memory) alive for the whole epoch.
        for term_name, loss_field in LOSS_TERMS.items() :
            accumulated_terms[term_name] += getattr(batch_loss_terms, loss_field).item() * x.shape[0]

        kl_weight = batch_loss_terms.kl_weight

    # Compute the final per-sample average of each term.
    epoch_terms = {term_name : term_value / len(train_loader.sampler) for term_name, term_value in accumulated_terms.items()}
    epoch_terms["kl_weight"] = float(kl_weight)

    # (OPTIONAL) Save the terms in the log dict used by wandb.
    if log_dict is not None :
        _update_log_dict(log_dict, epoch_terms, "train")

    return epoch_terms


def validation_epoch_function(model, validation_dataloader, loss_function, device, log_dict = None) -> dict :
    """
    Run a single validation epoch (forward pass only, no gradient) over ``validation_dataloader``.

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate. It is set in evaluation mode inside the function.
    validation_dataloader : torch.utils.data.DataLoader
        Dataloader yielding either ``(data, label)`` batches or plain data batches.
    loss_function : torch.nn.Module
        The loss module (as for :func:`train_epoch_function`).
    device : str or torch.device
        Device the data is moved to before the forward pass.
    log_dict : dict, optional
        If given, each averaged term is stored under the key ``'validation_<term>'``, by default None.

    Returns
    -------
    dict
        The averaged terms of the epoch, with the same keys as :func:`train_epoch_function`.

    Notes
    -----
    Note that the model is still stochastic in evaluation mode: the latent variable is sampled at every forward pass, so two validation epochs on the same data give slightly different values.
    """

    # Set the model in evaluation mode.
    model.eval()

    # Variables to accumulate each term of the loss.
    accumulated_terms = {term_name : 0.0 for term_name in LOSS_TERMS}
    kl_weight = 1.0

    # Disable the gradient computation.
    with torch.no_grad() :
        for sample_batch in validation_dataloader :
            # Move the data to the device (the label, when present, is not used).
            x = _unpack_batch(sample_batch).to(device)

            # Network forward pass.
            model_output = model(x)

            # Loss evaluation.
            batch_loss_terms = loss_function(model_output, x)

            # Accumulate every term, weighted by the batch size.
            for term_name, loss_field in LOSS_TERMS.items() :
                accumulated_terms[term_name] += getattr(batch_loss_terms, loss_field).item() * x.shape[0]

            kl_weight = batch_loss_terms.kl_weight

        # Compute the final per-sample average of each term.
        epoch_terms = {term_name : term_value / len(validation_dataloader.sampler) for term_name, term_value in accumulated_terms.items()}
        epoch_terms["kl_weight"] = float(kl_weight)

        # (OPTIONAL) Save the terms in the log dict used by wandb.
        if log_dict is not None :
            _update_log_dict(log_dict, epoch_terms, "validation")

    return epoch_terms
