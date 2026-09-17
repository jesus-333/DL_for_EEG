"""
ELBO (Evidence Lower BOund) loss, i.e. the loss used to train a variational autoencoder.

A variational autoencoder is trained by maximizing the evidence lower bound of the data likelihood, which is equivalent to minimizing

.. math::

    L = \\alpha \\, L_{R} + \\beta \\, w(e) \\, L_{KL}

where :math:`L_{R}` is the reconstruction term (how far the output is from the input), :math:`L_{KL}` is the Kullback-Leibler divergence between the approximate posterior and the prior (a regularizer that keeps the latent space close to a standard normal), and :math:`w(e)` is the warm-up multiplier at epoch :math:`e`.

Every hyper-parameter is described by :class:`~jade.config.loss_functions.elbo.elbo_config` and the building blocks live in :mod:`jade.training.custom_loss_function.support_loss`.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from collections import namedtuple

# Third party imports
import torch
from torch import nn

# Internal imports
from jade.config.loss_functions import get_loss_config
from jade.training.custom_loss_function import support_loss

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Loss output

# The output of an ELBO evaluation.
# A named tuple (and not a single scalar) because the loss is intrinsically made of several terms and all of them are worth logging.
# Keeping them in the return value, instead of stashing them on the loss object, means the loss stays a pure function of its inputs: it can be called for the training set and for the validation set without one call overwriting the results of the other.
#
# * total         : the value to backpropagate, i.e. alpha * reconstruction + beta * kl_weight * kl
# * reconstruction: the raw (unweighted) reconstruction term
# * kl            : the raw (unweighted) Kullback-Leibler term
# * kl_weight     : the warm-up multiplier applied to the KL term at the current epoch
# * total_full_kl : the same objective computed with the FINAL KL weight, i.e. alpha * reconstruction + beta * kl.
#                   While the warm-up runs, `total` changes meaning at every epoch, so comparing it across epochs (e.g. to pick the best model) would systematically favour the first epochs, where the KL term is cheap.
#                   `total_full_kl` is comparable from the very first epoch, and it is identical to `total` when the warm-up is disabled.
elbo_terms = namedtuple("elbo_terms", ["total", "reconstruction", "kl", "kl_weight", "total_full_kl"])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Loss declaration

class elbo_loss(nn.Module) :
    """
    ELBO loss for a variational autoencoder.

    It is a :class:`torch.nn.Module` (and not a plain class) so it behaves like every other loss of the package: the training cycle can move it to the training device with ``loss_function.to(device)``, wandb accepts it as a criterion, and a future reconstruction term carrying buffers or parameters (e.g. a learned output variance) needs no change here.

    Parameters
    ----------
    config : elbo_config or dict
        The loss configuration, an instance of :class:`~jade.config.loss_functions.elbo.elbo_config` (a dict is accepted too and promoted automatically through :func:`~jade.config.loss_functions.get_loss_config`).
        Every field is documented on that config class; the validation (and defaulting) happens there, so this constructor just reads the already-validated attributes.

    Notes
    -----
    The loss keeps track of the current epoch (see :meth:`set_epoch`) because the weight of the Kullback-Leibler term may depend on it.
    This is why the epoch is stored here instead of being passed at every call: the epoch functions keep the same signature they have for a classifier, and the validation epoch automatically uses the same KL weight as the training epoch it follows (which is what makes the two losses comparable).
    """

    def __init__(self, config) :
        super().__init__()

        # Accept a dict (or an already-built config) and promote it to a validated elbo_config.
        config = get_loss_config(config)

        self.config = config

        # Resolve the reconstruction term from its name (the KL term has a single closed form, so it needs no factory).
        self.reconstruction_loss_function = support_loss.get_reconstruction_loss_function(config.reconstruction_type)

        # Weights of the two terms.
        self.alpha = config.alpha
        self.beta  = config.beta

        # Current epoch and the matching warm-up multiplier.
        # They are plain python attributes and not buffers: they are training-cycle state, not model parameters, and they must not travel with a state dict nor with a .to(device) call.
        self.current_epoch = 0
        self.kl_weight     = support_loss.compute_kl_weight(0, config.kl_warmup_type, config.kl_warmup_epochs, config.kl_warmup_start)


    def set_epoch(self, epoch : int) -> None :
        """
        Declare the epoch the training is currently in, and update the weight of the Kullback-Leibler term accordingly.

        It is meant to be called once per epoch by the training cycle (see :func:`~jade.training.vae.train_model`), not once per batch: the weight is recomputed here and then reused by every call of the epoch.

        Parameters
        ----------
        epoch : int
            The current epoch (zero based).

        Raises
        ------
        ValueError
            If ``epoch`` is not a non-negative integer.
        """

        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0 :
            raise ValueError(f"epoch must be an integer >= 0, got {epoch!r}")

        self.current_epoch = epoch
        self.kl_weight     = support_loss.compute_kl_weight(epoch, self.config.kl_warmup_type, self.config.kl_warmup_epochs, self.config.kl_warmup_start)


    def forward(self, model_output, x : torch.Tensor) -> elbo_terms :
        """
        Compute the loss from the output of a generative model.

        The signature mirrors the one of a classification loss (``loss_function(prediction, target)``), with the peculiarity that for an autoencoder the target *is* the input.
        Note that the whole output of the model is taken as first argument, so the model can return additional elements (e.g. the sampled latent variable, or the predicted label once the classification branch is implemented) without any change here.

        Parameters
        ----------
        model_output : tuple
            The output of the model. The first three elements must be the reconstruction, the mean and the log variance of the approximate posterior.
        x : torch.Tensor
            The original signal, i.e. the input of the model.

        Returns
        -------
        elbo_terms
            The named tuple with the total loss and each of its terms. The field ``total`` is the one to backpropagate.

        Raises
        ------
        ValueError
            If ``model_output`` does not contain at least three elements.
        """

        # A model trained with this loss must return (at least) the reconstruction and the two parameters of the latent distribution.
        if not isinstance(model_output, (tuple, list)) or len(model_output) < 3 :
            raise ValueError("the ELBO expects a model whose forward returns at least (x_reconstructed, mu, log_var), got {}".format(type(model_output)))

        x_reconstructed, mu, log_var = model_output[0], model_output[1], model_output[2]

        return self.compute_loss(x, x_reconstructed, mu, log_var)


    def compute_loss(self, x : torch.Tensor, x_reconstructed : torch.Tensor, mu : torch.Tensor, log_var : torch.Tensor) -> elbo_terms :
        """
        Compute the loss from its explicit arguments.

        This is the entry point to call when the elements are already unpacked (e.g. in a test); :meth:`forward` simply unpacks the output of the model and delegates here.

        Parameters
        ----------
        x : torch.Tensor
            The original signal.
        x_reconstructed : torch.Tensor
            The signal reconstructed by the decoder (same shape as ``x``).
        mu : torch.Tensor
            The mean of the approximate posterior.
        log_var : torch.Tensor
            The logarithm of the variance of the approximate posterior (same shape as ``mu``).

        Returns
        -------
        elbo_terms
            The named tuple with the total loss and each of its terms.
        """

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # The two terms

        reconstruction_loss = self.reconstruction_loss_function(x_reconstructed, x, self.config)
        kl_loss             = support_loss.kl_divergence_normal(mu, log_var)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Combination

        # The value that is actually optimized, i.e. with the KL term weighted by the current warm-up multiplier.
        total_loss = self.alpha * reconstruction_loss + self.beta * self.kl_weight * kl_loss

        # The same objective with the final KL weight. See the comment on elbo_terms for why it is worth computing.
        total_loss_full_kl = self.alpha * reconstruction_loss + self.beta * kl_loss

        return elbo_terms(
            total          = total_loss,
            reconstruction = reconstruction_loss,
            kl             = kl_loss,
            kl_weight      = self.kl_weight,
            total_full_kl  = total_loss_full_kl,
        )
