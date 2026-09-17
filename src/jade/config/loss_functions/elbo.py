"""
ELBO (Evidence Lower BOund) loss configuration.

:class:`~jade.config.loss_functions.elbo.elbo_config` describes the loss used to train a variational autoencoder, i.e. :class:`~jade.training.custom_loss_function.elbo.elbo_loss`.
Contrary to the other members of this subpackage the ELBO has no :mod:`torch.nn` counterpart, so the loss itself is implemented inside :mod:`jade.training.custom_loss_function`.
The config lives here anyway, together with the other loss configs, because it is the ``[loss]`` section of a :class:`~jade.config.training.training_config` and it must be reachable through :data:`~jade.config.loss_functions.LOSS_CONFIG_REGISTRY` to be loadable from a TOML file.

The loss it describes is

.. math::

    L = \\alpha \\, L_{R} + \\beta \\, w(e) \\, L_{KL}

where :math:`L_{R}` is the reconstruction term (selected by ``reconstruction_type``), :math:`L_{KL}` is the Kullback-Leibler divergence between the approximate posterior and the standard normal prior, and :math:`w(e)` is the warm-up (annealing) multiplier at epoch :math:`e`.

Note that this loss needs a *generative* model, i.e. a model whose forward returns the reconstruction and the parameters of the latent distribution (e.g. :class:`~jade.model.vEEGNet.vEEGNet`).
The pairing is checked by :func:`~jade.training.support_training.cross_check_config`.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from dataclasses import dataclass

# Internal imports
from jade.config.loss_functions.generic import loss_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module constants

# Reconstruction terms implemented by jade.training.custom_loss_function.support_loss.
RECONSTRUCTION_TYPES = ("mse", "l1", "gaussian_nll")

# Warm-up (annealing) schedules of the KL term.
KL_WARMUP_TYPES = ("none", "linear")

# Reduction modes accepted by the loss.
# Note that, contrary to the classification losses, 'none' is not accepted: a non-reduced ELBO is not a scalar and cannot be backpropagated.
REDUCTIONS = ("mean", "sum")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Helper functions

def _is_int(value) -> bool :
    """
    Return True only for a genuine integer.

    ``bool`` is a subclass of ``int`` in Python, so it is excluded explicitly (a flag is never a valid count).
    """

    return isinstance(value, int) and not isinstance(value, bool)


def _check_non_negative_float(value, field_name : str) -> float :
    """
    Validate (and normalize) a non-negative float.

    Raize ``ValueError`` if ``value`` is not a real number greater than or equal to zero.
    """

    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0 :
        raise ValueError(f"{field_name} must be a number >= 0, got {value!r}")

    return float(value)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# ELBO config class

@dataclass
class elbo_config(loss_config) :
    """
    Configuration for :class:`~jade.training.custom_loss_function.elbo.elbo_loss`.

    Parameters
    ----------
    reconstruction_type : str, default 'mse'
        Reconstruction term. One of ``RECONSTRUCTION_TYPES``, i.e. ``'mse'`` (the term used in the original vEEGNet paper), ``'l1'`` or ``'gaussian_nll'``. Case-insensitive.
    alpha : float, default 1.0
        Weight of the reconstruction term. It must be greater than or equal to zero.
    beta : float, default 1.0
        Weight of the Kullback-Leibler term. It must be greater than or equal to zero.
        ``beta = 0`` removes the regularization, i.e. it turns the model into a plain (non variational) autoencoder.
        Values different from 1 correspond to the so called beta-VAE.
    reduction : str, default 'mean'
        How both terms are reduced over the batch. One of ``REDUCTIONS``.
    kl_warmup_type : str, default 'none'
        Warm-up (annealing) schedule of the KL weight. One of ``KL_WARMUP_TYPES``.
        A warm-up starts the training with a small KL weight, so the decoder can learn to reconstruct before the regularization kicks in (it mitigates the posterior collapse).
        With ``'none'`` the weight is constantly 1 and the loss is the plain (beta-)ELBO.
    kl_warmup_epochs : int, default 0
        Number of epochs the warm-up lasts, i.e. the first epoch at which the KL weight reaches 1.
        Ignored when ``kl_warmup_type = 'none'``, required (and greater than zero) otherwise.
    kl_warmup_start : float, default 0.0
        KL weight at epoch 0, linearly ramped up to 1 over ``kl_warmup_epochs``. It must be in ``[0, 1]``.
    gaussian_nll_eps : float, default 1e-6
        Lower bound on the variance, used only when ``reconstruction_type = 'gaussian_nll'``. It must be greater than zero.
    gaussian_nll_full : bool, default False
        If True the constant term of the Gaussian negative log-likelihood is included. Used only when ``reconstruction_type = 'gaussian_nll'``.

    Notes
    -----
    Contrary to :class:`~jade.config.loss_functions.cross_entropy.cross_entropy_config` and :class:`~jade.config.loss_functions.nll.nll_config` there is no ``class_weight`` and no ``ignore_index`` field: they describe a per-class rescaling, which has no meaning for a reconstruction objective.
    """

    name : str = "elbo"

    # Reconstruction term
    reconstruction_type : str = "mse"
    alpha               : float = 1.0

    # Kullback-Leibler term
    beta                : float = 1.0

    # Reduction over the batch (shared by both terms)
    reduction           : str = "mean"

    # KL warm-up (annealing)
    kl_warmup_type      : str = "none"
    kl_warmup_epochs    : int = 0
    kl_warmup_start     : float = 0.0

    # Used only when reconstruction_type = 'gaussian_nll'
    gaussian_nll_eps    : float = 1e-6
    gaussian_nll_full   : bool = False


    def __post_init__(self) -> None :
        """
        Validate the ELBO parameters and freeze the loss name.
        """

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Discriminator

        # Freeze the discriminator so the config always serializes as an ELBO config.
        self.name = "elbo"

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Reconstruction term and reduction

        self.reconstruction_type = str(self.reconstruction_type).lower()
        if self.reconstruction_type not in RECONSTRUCTION_TYPES :
            raise ValueError(f"reconstruction_type must be one of {RECONSTRUCTION_TYPES}, got {self.reconstruction_type!r}")

        self.reduction = str(self.reduction).lower()
        if self.reduction not in REDUCTIONS :
            raise ValueError(f"reduction must be one of {REDUCTIONS}, got {self.reduction!r}")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Weights of the two terms

        # Note that zero is allowed for both: alpha = 0 and beta = 0 are degenerate but legitimate ablations.
        self.alpha = _check_non_negative_float(self.alpha, "alpha")
        self.beta  = _check_non_negative_float(self.beta, "beta")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # KL warm-up

        self.kl_warmup_type = str(self.kl_warmup_type).lower()
        if self.kl_warmup_type not in KL_WARMUP_TYPES :
            raise ValueError(f"kl_warmup_type must be one of {KL_WARMUP_TYPES}, got {self.kl_warmup_type!r}")

        if not _is_int(self.kl_warmup_epochs) or self.kl_warmup_epochs < 0 :
            raise ValueError(f"kl_warmup_epochs must be an integer >= 0, got {self.kl_warmup_epochs!r}")

        # An active warm-up with no duration is raized, and not silently downgraded to 'none', so a typo in a config file cannot quietly train a different objective.
        if self.kl_warmup_type != "none" and self.kl_warmup_epochs <= 0 :
            raise ValueError(f"a {self.kl_warmup_type!r} KL warm-up needs kl_warmup_epochs > 0, got {self.kl_warmup_epochs!r}")

        if not isinstance(self.kl_warmup_start, (int, float)) or isinstance(self.kl_warmup_start, bool) or not 0.0 <= self.kl_warmup_start <= 1.0 :
            raise ValueError(f"kl_warmup_start must be a float in [0, 1], got {self.kl_warmup_start!r}")
        self.kl_warmup_start = float(self.kl_warmup_start)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Gaussian negative log-likelihood parameters

        if not isinstance(self.gaussian_nll_eps, (int, float)) or isinstance(self.gaussian_nll_eps, bool) or self.gaussian_nll_eps <= 0 :
            raise ValueError(f"gaussian_nll_eps must be a number > 0, got {self.gaussian_nll_eps!r}")
        self.gaussian_nll_eps = float(self.gaussian_nll_eps)

        if not isinstance(self.gaussian_nll_full, bool) :
            raise ValueError(f"gaussian_nll_full must be a bool, got {self.gaussian_nll_full!r}")
