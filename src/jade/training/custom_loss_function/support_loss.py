"""
Support functions for the custom loss functions.

This module holds the small, stateless pieces the custom losses are built from: the reconstruction terms, the Kullback-Leibler divergence and the warm-up schedule.
They are kept apart from the losses themselves (:mod:`jade.training.custom_loss_function.elbo`) because they are pure functions of their inputs, so they can be tested, reused and reasoned about without building a model.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Third party imports
import torch
from torch.nn import functional as F

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Reconstruction terms

def reconstruction_loss_mse(x_reconstructed : torch.Tensor, x : torch.Tensor, config = None) -> torch.Tensor :
    """
    Mean squared error between the reconstruction and the original signal.

    This is the reconstruction term used in the original vEEGNet paper.

    Parameters
    ----------
    x_reconstructed : torch.Tensor
        The signal reconstructed by the decoder.
    x : torch.Tensor
        The original signal (same shape as ``x_reconstructed``).
    config : elbo_config or None, optional
        The loss configuration. Only ``reduction`` is read. When None the torch default (``'mean'``) is used.

    Returns
    -------
    torch.Tensor
        The reduced reconstruction error.
    """

    reduction = "mean" if config is None else config.reduction

    return F.mse_loss(x_reconstructed, x, reduction = reduction)


def reconstruction_loss_l1(x_reconstructed : torch.Tensor, x : torch.Tensor, config = None) -> torch.Tensor :
    """
    Mean absolute error between the reconstruction and the original signal.

    Compared to :func:`reconstruction_loss_mse` it is less sensitive to the outliers, which on EEG data means it is less dominated by the artifacts.

    Parameters
    ----------
    x_reconstructed : torch.Tensor
        The signal reconstructed by the decoder.
    x : torch.Tensor
        The original signal (same shape as ``x_reconstructed``).
    config : elbo_config or None, optional
        The loss configuration. Only ``reduction`` is read.

    Returns
    -------
    torch.Tensor
        The reduced reconstruction error.
    """

    reduction = "mean" if config is None else config.reduction

    return F.l1_loss(x_reconstructed, x, reduction = reduction)


def reconstruction_loss_gaussian_nll(x_reconstructed : torch.Tensor, x : torch.Tensor, config = None) -> torch.Tensor :
    """
    Gaussian negative log-likelihood between the reconstruction and the original signal.

    The decoder of the current models predicts only the mean of the output distribution, so the variance is fixed to one and this term reduces to a scaled mean squared error (plus a constant, when ``gaussian_nll_full`` is True).
    It is implemented anyway because it is the extension point for a decoder that also predicts a variance: in that case only the ``variance`` argument below has to change.

    Parameters
    ----------
    x_reconstructed : torch.Tensor
        The predicted mean of the output distribution.
    x : torch.Tensor
        The original signal (same shape as ``x_reconstructed``).
    config : elbo_config or None, optional
        The loss configuration. ``reduction``, ``gaussian_nll_eps`` and ``gaussian_nll_full`` are read.

    Returns
    -------
    torch.Tensor
        The reduced negative log-likelihood.
    """

    reduction = "mean" if config is None else config.reduction
    eps       = 1e-6 if config is None else config.gaussian_nll_eps
    full      = False if config is None else config.gaussian_nll_full

    # Unit variance: the decoder predicts the mean only (see the note in the docstring).
    variance = torch.ones_like(x_reconstructed)

    return F.gaussian_nll_loss(x_reconstructed, x, variance, full = full, eps = eps, reduction = reduction)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Reconstruction registry and factory

# Map the lowercase name of a reconstruction term to the function that computes it.
RECONSTRUCTION_LOSS_REGISTRY = {
    "mse"          : reconstruction_loss_mse,
    "l1"           : reconstruction_loss_l1,
    "gaussian_nll" : reconstruction_loss_gaussian_nll,
}


def get_reconstruction_loss_function(name : str) :
    """
    Return the function that computes a reconstruction term.

    Parameters
    ----------
    name : str
        The name of the reconstruction term (case-insensitive). One of the keys of :data:`RECONSTRUCTION_LOSS_REGISTRY`.

    Returns
    -------
    callable
        A function with signature ``(x_reconstructed, x, config) -> torch.Tensor``.

    Raises
    ------
    ValueError
        If ``name`` is not a known reconstruction term.
    """

    name = str(name).lower()
    if name not in RECONSTRUCTION_LOSS_REGISTRY :
        raise ValueError("unknown reconstruction loss {}; expected one of {}".format(repr(name), sorted(RECONSTRUCTION_LOSS_REGISTRY)))

    return RECONSTRUCTION_LOSS_REGISTRY[name]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Kullback-Leibler divergence

def kl_divergence_normal(mu : torch.Tensor, log_var : torch.Tensor) -> torch.Tensor :
    """
    Kullback-Leibler divergence between the approximate posterior and a standard normal prior.

    Both distributions are assumed Gaussian with diagonal covariance, so the divergence has the closed form

    .. math::

        KL = -\\frac{1}{2} \\sum_{i} \\left( 1 + \\log \\sigma_{i}^{2} - \\mu_{i}^{2} - \\sigma_{i}^{2} \\right)

    The sum runs over every dimension of the latent variable and the result is averaged over the batch.
    Note that the sum is computed over every non-batch axis, so the function works both for a vector latent space (shape ``batch x d``) and for a convolutional one (shape ``batch x depth x height x width``).

    Parameters
    ----------
    mu : torch.Tensor
        The mean of the approximate posterior. The first axis is the batch.
    log_var : torch.Tensor
        The logarithm of the variance of the approximate posterior (same shape as ``mu``).

    Returns
    -------
    torch.Tensor
        A scalar tensor, the batch-averaged divergence.
    """

    # Every axis but the batch one. Computing it from the number of dimensions keeps the function valid for both latent-space layouts.
    axis_to_sum = list(range(1, mu.dim()))

    return torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim = axis_to_sum), dim = 0)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Warm-up (annealing) schedule

def compute_kl_weight(epoch : int, warmup_type : str, warmup_epochs : int, warmup_start : float) -> float :
    """
    Compute the multiplier applied to the Kullback-Leibler term at a given epoch.

    A warm-up starts the training with a small KL weight, so the decoder can learn to reconstruct before the regularization kicks in.
    Without it the KL term can dominate from the very first epochs and drive the model towards the posterior collapse, i.e. a latent space that carries no information about the input.

    Parameters
    ----------
    epoch : int
        The current epoch (zero based).
    warmup_type : str
        The schedule. ``'none'`` returns 1 at every epoch, ``'linear'`` ramps the weight up linearly.
    warmup_epochs : int
        The number of epochs the warm-up lasts, i.e. the first epoch at which the weight reaches 1. Ignored when ``warmup_type`` is ``'none'``.
    warmup_start : float
        The weight at epoch 0, in ``[0, 1]``.

    Returns
    -------
    float
        The multiplier, in ``[warmup_start, 1]``.

    Raises
    ------
    ValueError
        If ``warmup_type`` is not a known schedule.
    """

    warmup_type = str(warmup_type).lower()

    # No warm-up: the KL term always has its full weight.
    if warmup_type == "none" :
        return 1.0

    if warmup_type == "linear" :
        # A non positive duration would divide by zero. The config forbids it, but keep the function safe when called directly.
        if warmup_epochs <= 0 :
            return 1.0

        return float(warmup_start + (1.0 - warmup_start) * min(1.0, epoch / warmup_epochs))

    raise ValueError("unknown KL warm-up type {}; expected 'none' or 'linear'".format(repr(warmup_type)))
