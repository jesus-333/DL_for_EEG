"""
Support functions for the custom loss functions.

This module holds the small, stateless pieces the custom losses are built from: the reconstruction terms, the Kullback-Leibler divergence and the warm-up schedule.
The only stateful piece is :class:`dtw_reconstruction_loss`, which wraps the SoftDTW / block DTW of the optional package ``dtw-loss-functions``.
They are kept apart from the losses themselves (:mod:`jade.training.custom_loss_function.elbo`) because they are pure functions of their inputs, so they can be tested, reused and reasoned about without building a model.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Standard library imports
import importlib.metadata

# Third party imports
import torch
from torch import nn
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

# Oldest version of dtw-loss-functions with the API used by dtw_reconstruction_loss (soft_dtw / block_dtw built from implementation + sdtw_config).
MIN_DTW_LOSS_FUNCTIONS_VERSION = "1.0.4"

class dtw_reconstruction_loss(nn.Module) :
    """
    Reconstruction term based on the SoftDTW (``'soft_dtw'``) or on the block DTW (``'block_dtw'``), i.e. the reconstruction term used to train hvEEGNet.

    The DTW compares two 1D sequences, so it is computed independently for every channel of every trial: an input of shape ``(batch, depth, C, T)`` is reshaped into ``batch * C`` sequences of ``T`` samples with ``depth`` features each, and all of them are processed in a single batched call.
    The ``C`` values of a trial are then summed (or averaged, if ``config.dtw_average_channels``), optionally divided by ``T`` (``config.dtw_average_time_samples``), and finally reduced over the batch with ``config.reduction``.
    With the default settings this is exactly the reconstruction term of the original hvEEGNet implementation (sum over the channels, mean over the batch).

    The actual DTW computation is delegated to :class:`dtw_loss_functions.soft_dtw.soft_dtw` and :class:`dtw_loss_functions.block_dtw.block_dtw`.
    The package is an optional dependency (``pip install "jade[extra]"``), so it is imported only here, when the loss is built.

    Note that some implementations (``'mag'``, ``'pysdtw'``) decide at construction time whether to run the CUDA kernel.
    So the underlying DTW module is created lazily, once per device type, the first time a tensor on that device is received: the same loss object works on CPU and GPU without any extra flag.

    Parameters
    ----------
    config : elbo_config
        The loss configuration. ``reconstruction_type``, ``reduction`` and the ``dtw_*`` fields are read.

    Raises
    ------
    ImportError
        If ``dtw-loss-functions`` is not installed, or if the installed version is older than ``MIN_DTW_LOSS_FUNCTIONS_VERSION`` (1.0.4).
    """

    def __init__(self, config) :
        super().__init__()

        # Fail early (when the loss is built, not at the first batch) if the optional dependency is missing or too old.
        # Note that the versions before 1.0.4 are installable but have a different API (e.g. 1.0.2 has no soft_dtw module), so the two cases get different messages.
        try :
            from dtw_loss_functions import soft_dtw, block_dtw
        except ImportError as error :
            try :
                installed_version = importlib.metadata.version("dtw-loss-functions")
            except importlib.metadata.PackageNotFoundError :
                installed_version = None

            if installed_version is None :
                raise ImportError(f"the reconstruction term {config.reconstruction_type!r} needs the optional package dtw-loss-functions (>= {MIN_DTW_LOSS_FUNCTIONS_VERSION}). Install it with: pip install \"dtw-loss-functions>={MIN_DTW_LOSS_FUNCTIONS_VERSION}\" (or pip install \"jade[extra]\")") from error
            else :
                raise ImportError(f"the reconstruction term {config.reconstruction_type!r} needs dtw-loss-functions >= {MIN_DTW_LOSS_FUNCTIONS_VERSION}, but version {installed_version} is installed. Update it with: pip install --upgrade \"dtw-loss-functions>={MIN_DTW_LOSS_FUNCTIONS_VERSION}\"") from error

        self.soft_dtw_module  = soft_dtw
        self.block_dtw_module = block_dtw
        self.config = config

        # DTW modules already built, keyed by device type (see the class docstring).
        # A plain dict and not a ModuleDict: they hold no parameter, and they must not appear in a state dict.
        self.dtw_functions = {}

    def get_dtw_function(self, device : torch.device) :
        """
        Return the DTW module for the given device, building it on first use.
        """

        if device.type not in self.dtw_functions :
            sdtw_config = {
                "gamma"     : self.config.dtw_gamma,
                "normalize" : self.config.dtw_normalize,
                "bandwidth" : self.config.dtw_bandwidth,
            }

            # Only these two implementations have the use_cuda option.
            if self.config.dtw_implementation in ("mag", "pysdtw") : sdtw_config["use_cuda"] = device.type == "cuda"

            if self.config.reconstruction_type == "block_dtw" :
                dtw_function = self.block_dtw_module.block_dtw(self.config.dtw_block_size, implementation = self.config.dtw_implementation, sdtw_config = sdtw_config)
            else :
                dtw_function = self.soft_dtw_module.soft_dtw(implementation = self.config.dtw_implementation, sdtw_config = sdtw_config)

            self.dtw_functions[device.type] = dtw_function

        return self.dtw_functions[device.type]

    def forward(self, x_reconstructed : torch.Tensor, x : torch.Tensor, config = None) -> torch.Tensor :
        """
        Compute the DTW reconstruction term.

        Parameters
        ----------
        x_reconstructed : torch.Tensor
            The reconstructed signal, of shape ``(batch, depth, C, T)``.
        x : torch.Tensor
            The original signal (same shape).
        config : elbo_config, optional
            Accepted for signature compatibility with the other reconstruction terms and ignored: the config given at construction is used.

        Returns
        -------
        torch.Tensor
            A scalar tensor.

        Raises
        ------
        ValueError
            If the two tensors do not have the same 4D shape.
        """

        if x.dim() != 4 or x_reconstructed.shape != x.shape :
            raise ValueError(f"the DTW reconstruction term expects two tensors with the same (batch, depth, C, T) shape, got {tuple(x_reconstructed.shape)} and {tuple(x.shape)}")

        batch_size, depth, n_channels, n_time_samples = x.shape

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # One sequence per (trial, channel), in the (sequences, time, features) layout of dtw_loss_functions

        # Note the contiguous copy: the block DTW of dtw_loss_functions reshapes its inputs with view, which fails on a non-contiguous tensor.
        x_sequences               = x.permute(0, 2, 3, 1).reshape(batch_size * n_channels, n_time_samples, depth).contiguous()
        x_reconstructed_sequences = x_reconstructed.permute(0, 2, 3, 1).reshape(batch_size * n_channels, n_time_samples, depth).contiguous()

        dtw_per_channel = self.get_dtw_function(x.device)(x_sequences, x_reconstructed_sequences).view(batch_size, n_channels)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Reduction: channels, time, then batch

        dtw_per_trial = dtw_per_channel.mean(dim = 1) if self.config.dtw_average_channels else dtw_per_channel.sum(dim = 1)
        if self.config.dtw_average_time_samples : dtw_per_trial = dtw_per_trial / n_time_samples

        return dtw_per_trial.mean() if self.config.reduction == "mean" else dtw_per_trial.sum()


# Map the lowercase name of a reconstruction term to what computes it.
# The DTW terms map to a class (they hold state), which get_reconstruction_loss_function instantiates with the loss config.
RECONSTRUCTION_LOSS_REGISTRY = {
    "mse"          : reconstruction_loss_mse,
    "l1"           : reconstruction_loss_l1,
    "gaussian_nll" : reconstruction_loss_gaussian_nll,
    "soft_dtw"     : dtw_reconstruction_loss,
    "block_dtw"    : dtw_reconstruction_loss,
}


def get_reconstruction_loss_function(name : str, config = None) :
    """
    Return the callable that computes a reconstruction term.

    Parameters
    ----------
    name : str
        The name of the reconstruction term (case-insensitive). One of the keys of :data:`RECONSTRUCTION_LOSS_REGISTRY`.
    config : elbo_config, optional
        The loss configuration. Required only for the DTW terms, which are built from it (see :class:`dtw_reconstruction_loss`).

    Returns
    -------
    callable
        A function (or, for the DTW terms, a :class:`torch.nn.Module`) with signature ``(x_reconstructed, x, config) -> torch.Tensor``.

    Raises
    ------
    ValueError
        If ``name`` is not a known reconstruction term, or if a DTW term is requested without a config.
    ImportError
        If a DTW term is requested and ``dtw-loss-functions`` is not installed.
    """

    name = str(name).lower()
    if name not in RECONSTRUCTION_LOSS_REGISTRY :
        raise ValueError("unknown reconstruction loss {}; expected one of {}".format(repr(name), sorted(RECONSTRUCTION_LOSS_REGISTRY)))

    reconstruction_loss = RECONSTRUCTION_LOSS_REGISTRY[name]

    # A stateful term (a class) is instantiated with the loss config.
    if isinstance(reconstruction_loss, type) :
        if config is None :
            raise ValueError(f"the reconstruction term {name!r} is built from the loss config, pass it as config")
        return reconstruction_loss(config)

    return reconstruction_loss

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


def kl_divergence_shifted_normal(prior_log_var : torch.Tensor, delta_mu : torch.Tensor, delta_log_var : torch.Tensor) -> torch.Tensor :
    """
    Kullback-Leibler divergence between an approximate posterior expressed as a shift of its prior and the prior itself (Section 3.2 of the NVAE paper).

    The prior is :math:`p = N(\\mu, \\sigma^{2})` and the posterior is :math:`q = N(\\mu + \\Delta\\mu, \\sigma^{2} \\Delta\\sigma^{2})`, i.e. ``log_var + delta_log_var`` is the log variance of the posterior. The divergence has the closed form

    .. math::

        KL(q || p) = -\\frac{1}{2} \\sum_{i} \\left( 1 + \\log \\Delta\\sigma_{i}^{2} - \\frac{\\Delta\\mu_{i}^{2}}{\\sigma_{i}^{2}} - \\Delta\\sigma_{i}^{2} \\right)

    Note that it does not depend on the mean of the prior. As for :func:`kl_divergence_normal`, the sum runs over every non-batch axis and the result is averaged over the batch.

    Parameters
    ----------
    prior_log_var : torch.Tensor
        The logarithm of the variance of the prior.
    delta_mu : torch.Tensor
        The shift of the mean (same shape as ``prior_log_var``).
    delta_log_var : torch.Tensor
        The shift of the log variance (same shape as ``prior_log_var``).

    Returns
    -------
    torch.Tensor
        A scalar tensor, the batch-averaged divergence.
    """

    axis_to_sum = list(range(1, delta_mu.dim()))

    return torch.mean(-0.5 * torch.sum(1 + delta_log_var - delta_mu.pow(2) / prior_log_var.exp() - delta_log_var.exp(), dim = axis_to_sum), dim = 0)


def kl_divergence_hierarchical(mu_list, log_var_list, delta_mu_list, delta_log_var_list) -> torch.Tensor :
    """
    Kullback-Leibler term of a hierarchical VAE (e.g. :class:`~jade.model.hvEEGNet.hvEEGNet`), i.e. the sum of the divergence of each latent space.

    The deepest latent space has a standard normal prior, so its divergence is :func:`kl_divergence_normal`.
    Every other latent space has a conditional prior and a posterior expressed as a shift of it, so its divergence is :func:`kl_divergence_shifted_normal`.

    Parameters
    ----------
    mu_list : list of torch.Tensor
        The mean of the posterior of the deepest latent space, followed by the means of the priors of the other ones.
    log_var_list : list of torch.Tensor
        The matching log variances.
    delta_mu_list : list of torch.Tensor
        The shifts of the means, one for each latent space but the deepest (so one element less than ``mu_list``).
    delta_log_var_list : list of torch.Tensor
        The matching shifts of the log variances.

    Returns
    -------
    torch.Tensor
        A scalar tensor, the sum of the batch-averaged divergences of every latent space.

    Raises
    ------
    ValueError
        If the lengths of the lists are not consistent.
    """

    if len(log_var_list) != len(mu_list) or len(delta_mu_list) != len(mu_list) - 1 or len(delta_log_var_list) != len(mu_list) - 1 :
        raise ValueError(f"inconsistent hierarchical latent parameters: {len(mu_list)} mu, {len(log_var_list)} log_var, {len(delta_mu_list)} delta_mu and {len(delta_log_var_list)} delta_log_var (expected L, L, L - 1, L - 1)")

    # Deepest latent space, with the standard normal prior.
    kl_loss = kl_divergence_normal(mu_list[0], log_var_list[0])

    # The other latent spaces. Note the shift of one in the index: the deepest latent space has no shift term.
    for level in range(1, len(mu_list)) :
        kl_loss = kl_loss + kl_divergence_shifted_normal(log_var_list[level], delta_mu_list[level - 1], delta_log_var_list[level - 1])

    return kl_loss

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
