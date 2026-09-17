"""
Implementation of vEEGNet model using PyTorch.

vEEGNet (Zancanaro et al.) is a variational autoencoder (VAE) for EEG data: the encoder is an :class:`~jade.model.EEGNet.EEGNet`, the decoder mirrors it, and between the two the signal is compressed into a probabilistic latent space.
A VAE learns two distributions at once: an approximate posterior ``q(z|x)`` (the encoder, which maps a trial to the parameters of a Gaussian) and a generative model ``p(x|z)`` (the decoder, which maps a point of the latent space back to a trial).
It is trained with the ELBO, see :class:`~jade.training.custom_loss_function.elbo.elbo_loss`.

Note that the classification branch of the original model is deliberately **not** implemented: this is the purely generative half of vEEGNet, so the model only reconstructs (see :meth:`vEEGNet.reconstruct`) and generates (see :meth:`vEEGNet.generate`).
The multi-branch (MBEEGNet) encoder of the original implementation is not implemented either: the encoder is always a single EEGNet.

The name of the variables respect the nomeclature of the original paper where it is possible.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

from __future__ import annotations

import torch
from torch import nn

from jade.config.model import get_model_config
from jade.model import support_functions
from jade.model.EEGNet import EEGNet

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Building blocks

class same_padding_conv_transpose_2d(nn.Module) :
    """
    Transposed convolution that preserves the spatial size of its input.

    :class:`torch.nn.ConvTranspose2d` has no ``padding = 'same'`` option, and the usual workaround (``output_padding``) is unavailable here because with ``stride = 1`` torch requires ``output_padding`` to be zero.
    So the size is preserved manually. With a padding of ``(k - 1) // 2`` the output of a stride-1 transposed convolution has size ``n`` when the kernel ``k`` is odd, and ``n + 1`` when it is even; this module therefore drops the last row/column of every axis whose kernel is even.
    Note that this is not a corner case: the default temporal kernels of EEGNet (``(1, 64)`` and ``(1, 16)``) are both even.

    It is used by :class:`eegnet_decoder` to invert the two convolutions that EEGNet builds with ``padding = 'same'``, keeping the decoder made of genuine transposed convolutions.

    Parameters
    ----------
    in_channels : int
        Number of channels of the input.
    out_channels : int
        Number of channels of the output.
    kernel_size : tuple of int
        Kernel size ``(height, width)``.
    groups : int, optional
        Number of blocked connections from input to output channels, by default 1 (as in :class:`torch.nn.ConvTranspose2d`).
    bias : bool, optional
        If True add a bias term, by default False.
    """

    def __init__(self, in_channels : int, out_channels : int, kernel_size, groups : int = 1, bias : bool = False) :
        super().__init__()

        kernel_size = tuple(kernel_size)

        # With this padding a stride-1 transposed convolution grows an axis by 1 if (and only if) its kernel is even.
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)

        self.convolution = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size = kernel_size,
            padding     = padding,
            groups      = groups,
            bias        = bias,
        )

        # Number of elements to remove at the end of each axis to compensate the growth described above.
        self.crop_height = 1 if kernel_size[0] % 2 == 0 else 0
        self.crop_width  = 1 if kernel_size[1] % 2 == 0 else 0

    def forward(self, x : torch.tensor) -> torch.tensor :
        x = self.convolution(x)

        # Crop the trailing samples (the slicing is a no-op when the corresponding kernel is odd).
        if self.crop_height > 0 : x = x[:, :, : -self.crop_height, :]
        if self.crop_width  > 0 : x = x[:, :, :, : -self.crop_width]

        return x


class gaussian_sampling_layer(nn.Module) :
    """
    Map the output of the encoder to the parameters of the latent distribution and sample from it.

    The approximate posterior is assumed Gaussian with diagonal covariance, so the encoder only has to produce its mean and (the logarithm of) its variance.
    The sample is then drawn with the reparametrization trick, i.e. ``z = mu + sigma * eps`` with ``eps`` drawn from a standard normal: written this way the randomness does not depend on the parameters, so the gradient can flow through the sampling.

    Two mappings are available, selected by ``config.parameters_map_type``:

    * ``'linear'`` (the mapping of the original paper): the encoder output is flattened and two feedforward layers produce two vectors of ``hidden_space_dimension`` elements. The latent variable is a vector.
    * ``'convolutional'``: two ``1x1`` convolutions produce two tensors with the same shape as the encoder output. The latent variable keeps the time axis, i.e. the latent space is a sequence of vectors instead of a single one.

    Parameters
    ----------
    config : vEEGNet_config
        The (already validated) network configuration.
    encoder_output_shape : tuple of int
        Shape of the encoder output, without the batch axis, i.e. ``(filter_2, height, time)``.

    Attributes
    ----------
    latent_shape : tuple of int
        Shape of the latent variable, without the batch axis. It is what :meth:`~jade.model.vEEGNet.vEEGNet.generate` samples from the prior.
    """

    def __init__(self, config, encoder_output_shape) :
        super().__init__()

        self.parameters_map_type = config.parameters_map_type
        self.encoder_output_shape = tuple(encoder_output_shape)

        if self.parameters_map_type == "linear" :
            # Number of neurons produced by the encoder, i.e. the size of its flattened output.
            n_encoder_features = 1
            for axis_size in self.encoder_output_shape : n_encoder_features *= axis_size

            self.mu_map      = nn.Linear(n_encoder_features, config.hidden_space_dimension)
            self.log_var_map = nn.Linear(n_encoder_features, config.hidden_space_dimension)

            self.latent_shape = (config.hidden_space_dimension,)
        else :
            # Two separate 1x1 convolutions (instead of a single one that doubles the depth, later split in two halves):
            # the parameter count is the same, but each map is explicit and the layer stays symmetric with the linear branch.
            depth = self.encoder_output_shape[0]

            self.mu_map      = nn.Conv2d(depth, depth, kernel_size = (1, 1))
            self.log_var_map = nn.Conv2d(depth, depth, kernel_size = (1, 1))

            self.latent_shape = self.encoder_output_shape

    def forward(self, x : torch.tensor) :
        """
        Compute the parameters of the latent distribution and draw a sample from it.

        Parameters
        ----------
        x : torch.tensor
            The output of the encoder, of shape ``(batch, filter_2, height, time)``.

        Returns
        -------
        z : torch.tensor
            The sampled latent variable.
        mu : torch.tensor
            The mean of the approximate posterior.
        log_var : torch.tensor
            The logarithm of the variance of the approximate posterior.
        """

        # The feedforward maps work on a vector, the convolutional ones directly on the feature map.
        if self.parameters_map_type == "linear" : x = x.flatten(1)

        mu      = self.mu_map(x)
        log_var = self.log_var_map(x)

        z = self.reparametrize(mu, log_var)

        return z, mu, log_var

    def reparametrize(self, mu : torch.tensor, log_var : torch.tensor) -> torch.tensor :
        """
        Sample from the latent distribution with the reparametrization trick.

        The network predicts the logarithm of the variance (and not the variance, nor the standard deviation) because it is unconstrained: any real value maps to a positive variance, so the training cannot produce an invalid distribution.

        Parameters
        ----------
        mu : torch.tensor
            The mean of the approximate posterior.
        log_var : torch.tensor
            The logarithm of its variance (same shape as ``mu``).

        Returns
        -------
        torch.tensor
            The sample (same shape as ``mu``).
        """

        sigma = torch.exp(0.5 * log_var)

        # randn_like already creates the noise on the device (and with the dtype) of sigma.
        return mu + sigma * torch.randn_like(sigma)


class eegnet_decoder(nn.Module) :
    """
    Decoder of vEEGNet, i.e. a mirrored :class:`~jade.model.EEGNet.EEGNet`.

    Each block of the encoder is inverted by the corresponding block here, in reverse order: the separable convolution first, then the spatial filter, then the temporal filter.
    The convolutions are inverted by transposed convolutions and the pooling stages by an upsampling.

    Note how the pooling stages are inverted: instead of upsampling by the pooling factor, each stage resizes the tensor back to the shape its counterpart in the encoder *received* (``upsample_target_shapes``, measured at build time).
    This is what guarantees that the reconstruction has exactly the shape of the input, for any number of time samples and any pooling factor: a pooling stage outputs ``T // M`` samples, so upsampling by ``M`` gives the input length back only when ``M`` divides ``T``. With the default setup (``T = 1000``, factors 4 and 8) the naive version returns 992 samples instead of 1000.
    It also makes every pooling operator work the same way, including the ones that do not follow an exact floor division (e.g. a max pooling with ``ceil_mode``, or the wavelet pooling) and the disabled stages (the resize is then a no-op).

    Note also that the upsampling is not, and cannot be, the inverse of the pooling: pooling discards information (it low-passes, or drops a sub-band, or is non-linear). The upsampling restores the *geometry* and the transposed convolutions learn to repair the *content*.

    Parameters
    ----------
    config : vEEGNet_config
        The (already validated) network configuration.
    encoder_output_shape : tuple of int
        Shape of the encoder output, without the batch axis.
    upsample_target_shapes : tuple
        The ``(height, time)`` shapes received by the first and the second pooling stage of the encoder, in this order.
    latent_shape : tuple of int
        Shape of the latent variable, without the batch axis.
    """

    def __init__(self, config, encoder_output_shape, upsample_target_shapes, latent_shape) :
        super().__init__()

        use_bias = config.use_bias
        D = config.D
        activation = support_functions.get_activation(config.activation)
        dropout = support_functions.get_dropout(config.prob_dropout, config.use_dropout_2d)

        self.encoder_output_shape = tuple(encoder_output_shape)
        self.parameters_map_type  = config.parameters_map_type

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Latent input stage

        # With the linear mapping the latent variable is a vector, so it must be projected back to the shape of the encoder output before the convolutional blocks.
        # With the convolutional mapping it already has that shape, so there is nothing to do.
        if self.parameters_map_type == "linear" :
            n_encoder_features = 1
            for axis_size in self.encoder_output_shape : n_encoder_features *= axis_size

            self.latent_projection = nn.Sequential(
                nn.Linear(latent_shape[0], n_encoder_features),
                activation,
            )
        else :
            self.latent_projection = nn.Identity()

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Convolutional section

        # Block 2 (inverse of the separable convolution).
        # The pointwise convolution is inverted first (it is the last one of the encoder block), then the depthwise one.
        self.separable_convolution_transpose = nn.Sequential(
            nn.Upsample(size = upsample_target_shapes[1], mode = config.upsample_mode),
            same_padding_conv_transpose_2d(config.filter_2    , config.filter_1 * D, kernel_size = (1, 1)           , groups = 1                 , bias = use_bias),
            same_padding_conv_transpose_2d(config.filter_1 * D, config.filter_1 * D, kernel_size = config.c_kernel_3, groups = config.filter_1 * D, bias = use_bias),
            nn.BatchNorm2d(config.filter_1 * D),
            activation,
            dropout
        )

        # Block 1 - Spatial filters (inverse).
        # The spatial convolution of the encoder is not padded, so it shrinks the channel axis from C to C - c_kernel_2[0] + 1.
        # A transposed convolution with the same kernel and no padding grows it back by c_kernel_2[0] - 1, i.e. exactly to C, for any kernel height.
        self.spatial_filter_transpose = nn.Sequential(
            nn.Upsample(size = upsample_target_shapes[0], mode = config.upsample_mode),
            nn.ConvTranspose2d(config.filter_1 * D, config.filter_1, kernel_size = config.c_kernel_2, groups = config.filter_1, bias = use_bias),
            nn.BatchNorm2d(config.filter_1),
            activation,
            dropout
        )

        # Block 1 - Temporal filters (inverse).
        # Nothing follows this layer: its output is the reconstructed signal, so it must be free to take any value.
        # A normalization or an activation here would constrain the amplitude of the reconstruction, which is not bounded for an EEG signal.
        self.temporal_filter_transpose = same_padding_conv_transpose_2d(config.filter_1, config.depth_first_layer, kernel_size = config.c_kernel_1, groups = 1, bias = use_bias)

    def forward(self, z : torch.tensor) -> torch.tensor :
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # From the latent space back to a feature map
        x = self.latent_projection(z)
        if self.parameters_map_type == "linear" : x = x.view(-1, *self.encoder_output_shape)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Block 2 (separable convolution)
        x = self.separable_convolution_transpose(x)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Block 1 (spatial + temporal filters)
        x = self.spatial_filter_transpose(x)
        x = self.temporal_filter_transpose(x)

        return x

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Network declaration

class vEEGNet(nn.Module) :
    """
    Implementation of vEEGNet in PyTorch.

    The model chains an :class:`~jade.model.EEGNet.EEGNet` encoder, a :class:`gaussian_sampling_layer` and an :class:`eegnet_decoder`.
    Its forward returns the reconstruction together with the parameters of the latent distribution, which is what the ELBO needs (see :class:`~jade.training.custom_loss_function.elbo.elbo_loss`), and it is trained by :mod:`jade.training.vae`.

    Note that only the generative half of the original vEEGNet is implemented: there is no classification branch, so the model has no ``classify`` method and it must not be paired with a classification loss (:func:`~jade.training.support_training.cross_check_config` enforces it).

    Parameters
    ----------
    config : vEEGNet_config or dict
        The network configuration, an instance of :class:`~jade.config.model.vEEGNet_cfg.vEEGNet_config` (a dict is accepted too and promoted automatically through :func:`~jade.config.model.get_model_config`).
        Every field is documented on that config class; the fields validation (and defaulting) happens there, so this constructor just reads the already-validated attributes.

    Attributes
    ----------
    latent_shape : tuple of int
        Shape of the latent variable, without the batch axis.
    encoder_output_shape : tuple of int
        Shape of the encoder output, without the batch axis.
    upsample_target_shapes : tuple
        The ``(height, time)`` shapes the two pooling stages of the encoder receive, used by the decoder to restore them.

    Notes
    -----
    The geometry of the decoder is measured at build time with a dummy forward through the encoder (see ``config.input_size``), so the model works on trials of that geometry only.
    A tensor with a different shape is rejected by :meth:`forward` with a clear error, instead of being silently reconstructed with the wrong number of samples.
    """

    def __init__(self, config) :
        super().__init__()

        # Accept a dict (or an already-built config) and promote it to a validated vEEGNet_config.
        config = get_model_config(config)

        # Geometry of the input the model is built for (used by the guard in forward).
        self.depth_first_layer = config.depth_first_layer
        self.C = config.C
        self.T = config.T

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Encoder

        # The encoder is a plain EEGNet, built from the config derived by the vEEGNet config (which forces the feature-extractor variant and disables the flattening).
        self.encoder = EEGNet(config.to_encoder_config())

        # Measure the shapes the decoder needs. It must be done after the encoder is built and before the decoder is.
        self.encoder_output_shape, self.upsample_target_shapes = self._probe_encoder_geometry(config.input_size)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Latent space and decoder

        self.sampling = gaussian_sampling_layer(config, self.encoder_output_shape)
        self.latent_shape = self.sampling.latent_shape

        self.decoder = eegnet_decoder(config, self.encoder_output_shape, self.upsample_target_shapes, self.latent_shape)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Print additional information after the creation of the network
        if config.print_var :
            print("vEEGNet Created. Number of parameters:")
            print(f"\tNumber of trainable parameters (Encoder)  = {support_functions.count_trainable_parameters(self.encoder)}")
            print(f"\tNumber of trainable parameters (Sampling) = {support_functions.count_trainable_parameters(self.sampling)}")
            print(f"\tNumber of trainable parameters (Decoder)  = {support_functions.count_trainable_parameters(self.decoder)}\n")
            self.debug_shape(config.input_size)

    def _probe_encoder_geometry(self, input_size) :
        """
        Measure the shapes the decoder needs, by pushing a dummy input through the encoder.

        Two things are recorded: the shape of the encoder output (which sizes the latent maps and the projection back) and the shape each pooling stage receives (which the decoder restores).
        They cannot be computed in closed form, because a pooling stage may be any of the operators of :mod:`jade.model.pooling_1d` (or be disabled), so they are measured.

        Note that the probe runs with the encoder in evaluation mode and without gradients.
        Both matter: a random input must not pollute the running statistics of the batch normalization layers, and those layers raise an error in training mode when the probe (of batch size 1) has a single element per channel, which happens for a short input with large pooling factors.

        Parameters
        ----------
        input_size : tuple of int
            Shape ``(batch, depth, C, T)`` of the dummy input.

        Returns
        -------
        encoder_output_shape : tuple of int
            Shape of the encoder output, without the batch axis.
        upsample_target_shapes : tuple
            The ``(height, time)`` shapes received by the two pooling stages, in encoder order.
        """

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Attach the hooks

        shape_before_pooling = {}

        def make_hook(stage_index) :
            # A forward PRE hook, so the recorded shape is the INPUT of the pooling stage, i.e. what the decoder has to restore.
            def hook(module, module_input) :
                shape_before_pooling[stage_index] = tuple(module_input[0].shape[-2:])
            return hook

        hook_handles = [pooling_layer.register_forward_pre_hook(make_hook(i)) for i, pooling_layer in enumerate(self.encoder.pooling_layers)]

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Run the dummy input

        was_training = self.encoder.training
        try :
            self.encoder.eval()
            with torch.no_grad() :
                encoder_output = self.encoder(torch.randn(input_size))
        finally :
            # Always remove the hooks and restore the mode, even if the forward fails.
            for handle in hook_handles : handle.remove()
            self.encoder.train(was_training)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Collect the results

        encoder_output_shape   = tuple(encoder_output.shape[1:])
        upsample_target_shapes = (shape_before_pooling[0], shape_before_pooling[1])

        return encoder_output_shape, upsample_target_shapes

    def forward(self, x : torch.tensor) :
        """
        Encode, sample and decode.

        Parameters
        ----------
        x : torch.tensor
            A batch of EEG trials, of shape ``(batch, depth_first_layer, C, T)``.

        Returns
        -------
        x_reconstructed : torch.tensor
            The reconstructed signal, with exactly the shape of ``x``.
        mu : torch.tensor
            The mean of the approximate posterior.
        log_var : torch.tensor
            The logarithm of its variance.
        z : torch.tensor
            The sampled latent variable.

        Raises
        ------
        ValueError
            If the geometry of ``x`` is not the one the model was built for.

        Notes
        -----
        The latent variable is returned (it is not needed by the loss) because it is what most analyses of a VAE work on, and because the classification branch of the original paper consumes it.
        The order of the elements is append-only by design: a future element is added at the end, so ``output[0]`` is always the reconstruction.
        """

        # The decoder geometry is fixed at build time, so an input of a different shape would be silently reconstructed with the wrong number of channels/samples.
        if tuple(x.shape[1:]) != (self.depth_first_layer, self.C, self.T) :
            raise ValueError("this vEEGNet was built for an input of shape (batch, {}, {}, {}), got {}".format(self.depth_first_layer, self.C, self.T, tuple(x.shape)))

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Encoder and latent space
        x = self.encoder(x)
        z, mu, log_var = self.sampling(x)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Decoder
        x_reconstructed = self.decoder(z)

        return x_reconstructed, mu, log_var, z

    def encode(self, x : torch.tensor) :
        """
        Compute the parameters of the latent distribution of a batch of trials.

        Parameters
        ----------
        x : torch.tensor
            A batch of EEG trials.

        Returns
        -------
        mu : torch.tensor
            The mean of the approximate posterior.
        log_var : torch.tensor
            The logarithm of its variance.
        """

        z, mu, log_var = self.sampling(self.encoder(x))

        return mu, log_var

    def decode(self, z : torch.tensor) -> torch.tensor :
        """
        Map a point of the latent space back to a signal.

        Parameters
        ----------
        z : torch.tensor
            A batch of latent variables, of shape ``(batch, *latent_shape)``.

        Returns
        -------
        torch.tensor
            The decoded signal.
        """

        return self.decoder(z)

    def reconstruct(self, x : torch.tensor, no_grad : bool = True) -> torch.tensor :
        """
        Reconstruct a batch of trials, i.e. encode it and decode it back.

        Note that the reconstruction is stochastic: the latent variable is *sampled*, so two calls on the same input give two slightly different outputs.
        Note also that this method does not change the mode of the model: if it is in training mode the dropout layers are active and the batch normalization layers use (and update) the batch statistics. Call ``model.eval()`` first to reconstruct deterministically, apart from the sampling.

        Parameters
        ----------
        x : torch.tensor
            A batch of EEG trials.
        no_grad : bool, optional
            If True (the default) the computation is done inside :func:`torch.no_grad`, so the output carries no gradient. Set it to False to keep the reconstruction in the computational graph.

        Returns
        -------
        torch.tensor
            The reconstructed signal, with the shape of ``x``.
        """

        if no_grad :
            with torch.no_grad() :
                output = self.forward(x)
        else :
            output = self.forward(x)

        return output[0]

    def generate(self, n_samples : int = 1, z : torch.tensor = None, device = None, no_grad : bool = True) -> torch.tensor :
        """
        Generate new synthetic EEG trials.

        The latent variables are drawn from the prior (a standard normal) and decoded.
        This is the property that makes a VAE generative: the KL term of the training loss keeps the approximate posterior close to that prior, so a point sampled from it decodes into a plausible signal.

        Parameters
        ----------
        n_samples : int, optional
            Number of trials to generate, by default 1. Ignored when ``z`` is given.
        z : torch.tensor, optional
            The latent variables to decode, of shape ``(batch, *latent_shape)``, by default None (they are sampled from the prior).
            Pass them explicitly to walk the latent space, e.g. to interpolate between two encoded trials.
        device : torch.device or str, optional
            Where the latent variables are created, by default None, i.e. the device of the model. Ignored when ``z`` is given.
        no_grad : bool, optional
            If True (the default) the computation is done inside :func:`torch.no_grad`.

        Returns
        -------
        torch.tensor
            The generated signals, of shape ``(n_samples, depth_first_layer, C, T)``.

        Raises
        ------
        ValueError
            If ``n_samples`` is not a positive integer.
        """

        if z is None :
            if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples < 1 :
                raise ValueError(f"n_samples must be an integer >= 1, got {n_samples!r}")

            # Default to the device of the model, so the sampled tensor never lands on the wrong device.
            if device is None : device = next(self.parameters()).device

            # Note that latent_shape follows parameters_map_type, so this is correct for both mappings.
            z = torch.randn(n_samples, *self.latent_shape, device = device)

        if no_grad :
            with torch.no_grad() :
                x_generated = self.decoder(z)
        else :
            x_generated = self.decoder(z)

        return x_generated

    def debug_shape(self, input_size) :
        """
        Method that create a fake input and pass it through the network, showing after each step the shape
        """

        print("Shape tracking across vEEGNet")

        x = torch.randn(input_size)
        print("\tInput shape :\t\t\t", x.shape)

        x = self.encoder(x)
        print("\tEncoder output :\t\t", x.shape)

        z, mu, log_var = self.sampling(x)
        print("\tLatent parameters (mu) :\t", mu.shape)
        print("\tLatent sample (z) :\t\t", z.shape)

        x = self.decoder.latent_projection(z)
        if self.decoder.parameters_map_type == "linear" : x = x.view(-1, *self.encoder_output_shape)
        print("\tDecoder (latent projection) :\t", x.shape)

        x = self.decoder.separable_convolution_transpose(x)
        print("\tDecoder (separable conv) :\t", x.shape)

        x = self.decoder.spatial_filter_transpose(x)
        print("\tDecoder (spatial filter) :\t", x.shape)

        x = self.decoder.temporal_filter_transpose(x)
        print("\tDecoder (temporal filter) :\t", x.shape)

        # The whole point of the decoder geometry: the reconstruction must have exactly the shape of the input.
        print(f"\tInput shape == output shape :\t {tuple(x.shape) == tuple(input_size)}\n")
