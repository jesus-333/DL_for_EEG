"""
Implementation of hvEEGNet model using PyTorch.

hvEEGNet (Cisotto, Zancanaro, Zoppis & Manzoni, 2024) is the hierarchical version of :class:`~jade.model.vEEGNet.vEEGNet`, built following the hierarchical VAE of NVAE (Vahdat & Kautz, 2020).
The encoder is an :class:`~jade.model.EEGNet.EEGNet` and the decoder mirrors it, as in vEEGNet, but instead of a single latent space there is one after each of the three blocks of the encoder:

* ``z_0`` (the deepest), after the separable convolution;
* ``z_1``, after the spatial filter;
* ``z_2`` (the shallowest), after the temporal filter.

Note that the paper numbers use ``z1`` for the deepest and ``z3`` for the shallowest. Here the index follows the order in which the decoder visits them, which is also the order of the lists returned by :meth:`hvEEGNet.forward`.

The deepest latent space works as in a standard VAE: the encoder output is mapped to the parameters of a Gaussian, whose prior is a standard normal.
The other two are *conditional*: the decoder computes, from what it has decoded so far, the parameters of a prior ``p(z_l | z_<l)``; during training the encoder features of the same level are used to compute a *shift* ``(delta_mu, delta_log_var)`` of that prior, which gives the approximate posterior ``q(z_l | x, z_<l) = N(mu + delta_mu, exp(log_var + delta_log_var))`` (Section 3.2 of the NVAE paper).
The KL term of the ELBO is then the sum of the divergence of each level, see :func:`~jade.training.custom_loss_function.support_loss.kl_divergence_hierarchical`.

This is a direct implementation of hvEEGNet, not a generic hierarchical VAE: the number of latent spaces is fixed to three.
By default every latent space is convolutional (i.e. it keeps the shape of the feature map it is computed from), as in the paper; the vector mappings of the original implementation (feedforward, plain or LoRA) are available through ``parameters_map_type``.
The deepest decoder level also receives the extra input ``h`` of the NVAE architecture (trainable, or zeros, or given by the user).
The classification branch of the original implementation is not implemented.

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
from jade.model.vEEGNet import same_padding_conv_transpose_2d

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Building blocks

def reparametrize(mu : torch.tensor, log_var : torch.tensor) -> torch.tensor :
    """
    Sample from a Gaussian with diagonal covariance through the reparametrization trick, i.e. ``z = mu + sigma * eps`` with ``eps`` drawn from a standard normal.

    Parameters
    ----------
    mu : torch.tensor
        The mean of the distribution.
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


def _build_feedforward_layer(in_features : int, out_features : int, config) -> nn.Module :
    """
    Build the feedforward layer of a vector latent mapping: a :class:`torch.nn.Linear` for ``'linear'``, a :class:`~jade.model.support_functions.lora_linear` for ``'lora'``.
    """

    if config.parameters_map_type == "lora" :
        return support_functions.lora_linear(in_features, out_features, config.lora_rank, use_spectral_norm = config.use_spectral_norm)

    layer = nn.Linear(in_features, out_features)
    if config.use_spectral_norm : layer = nn.utils.parametrizations.spectral_norm(layer)

    return layer


class gaussian_parameters_map(nn.Module) :
    """
    Map a feature map to the parameters (mean and log variance) of a Gaussian over a latent space.

    The mapping depends on ``config.parameters_map_type``:

    * ``'convolutional'`` (the hvEEGNet of the paper): a ``1x1`` convolution doubles the depth of the tensor, the first half of the depth maps is the mean and the second half the log variance. The latent space has the shape of the feature map.
    * ``'linear'`` / ``'lora'``: the feature map is flattened and two feedforward layers (plain or low-rank) produce two vectors of ``latent_dimension`` elements.

    If ``config.use_activation_in_sampling`` the activation is applied first, as in the original implementation.

    Parameters
    ----------
    feature_shape : tuple of int
        Shape of the input feature map, without the batch axis.
    latent_dimension : int
        Dimension of the latent space. Ignored with the convolutional mapping.
    config : hvEEGNet_config
        The (already validated) network configuration.
    """

    def __init__(self, feature_shape, latent_dimension : int, config) :
        super().__init__()

        self.is_convolutional = config.parameters_map_type == "convolutional"

        self.activation = support_functions.get_activation(config.activation) if config.use_activation_in_sampling else nn.Identity()

        if self.is_convolutional :
            depth = feature_shape[0]
            self.parameters_map = nn.Conv2d(depth, 2 * depth, kernel_size = (1, 1))
        else :
            n_features = 1
            for axis_size in feature_shape : n_features *= axis_size

            self.mu_map      = _build_feedforward_layer(n_features, latent_dimension, config)
            self.log_var_map = _build_feedforward_layer(n_features, latent_dimension, config)

    def forward(self, x : torch.tensor) :
        """
        Return the mean and the log variance of the latent distribution.
        """

        x = self.activation(x)

        if self.is_convolutional :
            mu, log_var = self.parameters_map(x).chunk(2, dim = 1)
        else :
            x = x.flatten(1)
            mu, log_var = self.mu_map(x), self.log_var_map(x)

        return mu, log_var


class latent_projection(nn.Module) :
    """
    Map a sample of a latent space back to the feature map the latent space is attached to.

    With the convolutional mapping the sample already has the shape of the feature map, so this is an identity.
    With the vector mappings a feedforward layer (plain or low-rank) followed by the activation (if ``config.use_activation_in_sampling``) produces the flattened feature map, which is then reshaped.

    Note that there is a single projection per latent space, used both for the samples of the posterior (training) and of the prior (generation).
    In the original implementation the prior and the posterior sampling layers each had their own projection, so the one used during generation was never trained.

    Parameters
    ----------
    feature_shape : tuple of int
        Shape of the feature map, without the batch axis.
    latent_dimension : int
        Dimension of the latent space. Ignored with the convolutional mapping.
    config : hvEEGNet_config
        The (already validated) network configuration.
    """

    def __init__(self, feature_shape, latent_dimension : int, config) :
        super().__init__()

        self.feature_shape    = tuple(feature_shape)
        self.is_convolutional = config.parameters_map_type == "convolutional"

        if self.is_convolutional :
            self.projection = nn.Identity()
        else :
            n_features = 1
            for axis_size in self.feature_shape : n_features *= axis_size

            self.projection = nn.Sequential(
                _build_feedforward_layer(latent_dimension, n_features, config),
                support_functions.get_activation(config.activation) if config.use_activation_in_sampling else nn.Identity(),
            )

    def forward(self, z : torch.tensor) -> torch.tensor :
        x = self.projection(z)
        if not self.is_convolutional : x = x.view(-1, *self.feature_shape)

        return x


class features_combination(nn.Module) :
    """
    Combine two feature maps with the same height and width.

    The two tensors are concatenated along the depth axis and mixed by a ``1x1`` convolution, i.e. the output is a learned weighted sum of their depth maps.
    It is used both to merge the decoder features with the encoder features (to compute the posterior shift) and to merge the decoder features with the sampled latent variable (to feed the next decoder block).

    Parameters
    ----------
    depth_1 : int
        Depth of the first tensor.
    depth_2 : int
        Depth of the second tensor.
    depth_output : int
        Depth of the output.
    """

    def __init__(self, depth_1 : int, depth_2 : int, depth_output : int) :
        super().__init__()

        self.mix = nn.Conv2d(depth_1 + depth_2, depth_output, kernel_size = (1, 1))

    def forward(self, x_1 : torch.tensor, x_2 : torch.tensor) -> torch.tensor :
        return self.mix(torch.cat([x_1, x_2], dim = 1))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Network declaration

class hvEEGNet(nn.Module) :
    """
    Implementation of hvEEGNet in PyTorch.

    The model is made of:

    * an :class:`~jade.model.EEGNet.EEGNet` encoder, whose three blocks (temporal filter, spatial filter, separable convolution) are run one by one so that the output of each one is available;
    * the deepest latent space ``z_0``, computed from the encoder output, and the combination of its sample with the extra input ``h``;
    * three decoder blocks, mirroring the encoder ones in reverse order (same layers as :class:`~jade.model.vEEGNet.eegnet_decoder`);
    * for the two shallower levels, a prior map, a posterior (shift) map and two :class:`features_combination` modules.

    Its forward returns the reconstruction together with the parameters of all the latent distributions, which is what the ELBO needs (see :class:`~jade.training.custom_loss_function.elbo.elbo_loss`, that recognizes the hierarchical output), and it is trained by :mod:`jade.training.vae`.

    Parameters
    ----------
    config : hvEEGNet_config or dict
        The network configuration, an instance of :class:`~jade.config.model.hvEEGNet_cfg.hvEEGNet_config` (a dict is accepted too and promoted automatically through :func:`~jade.config.model.get_model_config`).

    Attributes
    ----------
    latent_shapes : tuple
        Shapes of the three latent variables, without the batch axis, from the deepest (``z_0``) to the shallowest (``z_2``). With the vector mappings each one is ``(hidden_space_dimension,)``.
    encoder_outputs_shapes : tuple
        Shapes of the outputs of the three encoder blocks, without the batch axis, in encoder order (temporal filter, spatial filter, separable convolution).
    upsample_target_shapes : tuple
        The ``(height, time)`` shapes the two pooling stages of the encoder receive, used by the decoder to restore them.

    Notes
    -----
    The geometry of the decoder is measured at build time with a dummy forward through the encoder (see ``config.input_size``), so the model works on trials of that geometry only.
    """

    def __init__(self, config) :
        super().__init__()

        # Accept a dict (or an already-built config) and promote it to a validated hvEEGNet_config.
        config = get_model_config(config)

        # Geometry of the input the model is built for (used by the guard in forward).
        self.depth_first_layer = config.depth_first_layer
        self.C = config.C
        self.T = config.T

        use_bias   = config.use_bias
        activation = support_functions.get_activation(config.activation)
        dropout    = support_functions.get_dropout(config.prob_dropout, config.use_dropout_2d)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Encoder

        self.encoder = EEGNet(config.to_encoder_config())

        # Measure the shapes the decoder needs. It must be done after the encoder is built and before the decoder is.
        self.encoder_outputs_shapes, self.upsample_target_shapes = self._probe_encoder_geometry(config.input_size)

        # Depth of the output of each encoder block (temporal filter, spatial filter, separable convolution).
        depth_temporal, depth_spatial, depth_separable = [shape[0] for shape in self.encoder_outputs_shapes]

        # Shape of the feature map each latent space is attached to. Deepest first (z_0, z_1, z_2).
        feature_shapes = (self.encoder_outputs_shapes[2], self.encoder_outputs_shapes[1], self.encoder_outputs_shapes[0])

        # Shape of the latent variables, deepest first: the feature map itself with the convolutional mapping, a vector otherwise.
        self.parameters_map_type = config.parameters_map_type
        if self.parameters_map_type == "convolutional" :
            self.latent_shapes = feature_shapes
        else :
            self.latent_shapes = tuple((hidden_space_dimension,) for hidden_space_dimension in config.hidden_space_dimension_list)
        latent_dimensions = config.hidden_space_dimension_list

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Deepest latent space (z_0)

        # Approximate posterior q(z_0 | x). Its prior is a standard normal.
        self.posterior_map_0 = gaussian_parameters_map(feature_shapes[0], latent_dimensions[0], config)

        # Input stage of the decoder: the sample of z_0 (as a feature map) is combined with h (Fig. 2 of the NVAE paper).
        # h is a trainable tensor if use_h_in_decoder, otherwise a tensor of zeros (the hvEEGNet of the paper), and it can always be overridden by passing h to forward / generate.
        # Note that the batch axis of h is 1, it is broadcasted over the batch.
        self.use_h_in_decoder = config.use_h_in_decoder
        if self.use_h_in_decoder : self.h = nn.Parameter(torch.zeros(1, *feature_shapes[0]))
        self.h_decoder_combination = features_combination(depth_separable, depth_separable, depth_separable)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Decoder blocks (same layers of jade.model.vEEGNet.eegnet_decoder)

        # Inverse of the separable convolution. Its output has the shape of the spatial filter output.
        self.separable_convolution_transpose = nn.Sequential(
            nn.Upsample(size = self.upsample_target_shapes[1], mode = config.upsample_mode),
            same_padding_conv_transpose_2d(depth_separable, depth_spatial, kernel_size = (1, 1)           , groups = 1            , bias = use_bias),
            same_padding_conv_transpose_2d(depth_spatial  , depth_spatial, kernel_size = config.c_kernel_3, groups = depth_spatial, bias = use_bias),
            nn.BatchNorm2d(depth_spatial),
            activation,
            dropout,
        )

        # Inverse of the spatial filter. Its output has the shape of the temporal filter output.
        # The unpadded transposed convolution grows the channel axis back from C - c_kernel_2[0] + 1 to C.
        self.spatial_filter_transpose = nn.Sequential(
            nn.Upsample(size = self.upsample_target_shapes[0], mode = config.upsample_mode),
            nn.ConvTranspose2d(depth_spatial, depth_temporal, kernel_size = config.c_kernel_2, groups = config.filter_1, bias = use_bias),
            nn.BatchNorm2d(depth_temporal),
            activation,
            dropout,
        )

        # Inverse of the temporal filter. Nothing follows it: its output is the reconstructed signal, so it must be free to take any value.
        self.temporal_filter_transpose = same_padding_conv_transpose_2d(depth_temporal, config.depth_first_layer, kernel_size = config.c_kernel_1, groups = 1, bias = use_bias)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Shallower latent spaces (z_1 and z_2)
        # Index 0 of each ModuleList refers to z_1 (after the spatial filter), index 1 to z_2 (after the temporal filter).

        # Prior p(z_l | z_<l), computed from the decoder features.
        self.prior_maps = nn.ModuleList([
            gaussian_parameters_map(feature_shapes[1], latent_dimensions[1], config),
            gaussian_parameters_map(feature_shapes[2], latent_dimensions[2], config),
        ])

        # Merge the decoder features with the encoder features of the same level, to compute the posterior shift.
        self.encoder_decoder_combinations = nn.ModuleList([
            features_combination(depth_spatial , depth_spatial , depth_spatial),
            features_combination(depth_temporal, depth_temporal, depth_temporal),
        ])

        # Shift (delta_mu, delta_log_var) of the prior, i.e. the approximate posterior q(z_l | x, z_<l).
        self.posterior_shift_maps = nn.ModuleList([
            gaussian_parameters_map(feature_shapes[1], latent_dimensions[1], config),
            gaussian_parameters_map(feature_shapes[2], latent_dimensions[2], config),
        ])

        # From the latent space back to the feature map, one per latent space (deepest first). Identities with the convolutional mapping.
        self.latent_projections = nn.ModuleList([latent_projection(feature_shapes[level], latent_dimensions[level], config) for level in range(3)])

        # Merge the decoder features with the sampled latent variable, to feed the next decoder block.
        self.latent_decoder_combinations = nn.ModuleList([
            features_combination(depth_spatial , depth_spatial , depth_spatial),
            features_combination(depth_temporal, depth_temporal, depth_temporal),
        ])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Print additional information after the creation of the network
        if config.print_var :
            print("hvEEGNet Created. Number of parameters:")
            print(f"\tNumber of trainable parameters (Encoder) = {support_functions.count_trainable_parameters(self.encoder)}")
            print(f"\tNumber of trainable parameters (Total)   = {support_functions.count_trainable_parameters(self)}\n")
            self.debug_shape(config.input_size)

    def _probe_encoder_geometry(self, input_size) :
        """
        Measure the shapes the decoder needs, by pushing a dummy input through the encoder.

        Two things are recorded: the output shape of each encoder block (which sizes the latent spaces) and the shape each pooling stage receives (which the decoder restores).
        As in :class:`~jade.model.vEEGNet.vEEGNet`, the probe runs in evaluation mode and without gradients, so it does not pollute the batch normalization statistics.

        Parameters
        ----------
        input_size : tuple of int
            Shape ``(batch, depth, C, T)`` of the dummy input.

        Returns
        -------
        encoder_outputs_shapes : tuple
            Shapes of the outputs of the three encoder blocks, without the batch axis, in encoder order.
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
                encoder_outputs = self._encoder_blocks_forward(torch.randn(input_size))
        finally :
            # Always remove the hooks and restore the mode, even if the forward fails.
            for handle in hook_handles : handle.remove()
            self.encoder.train(was_training)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Collect the results

        encoder_outputs_shapes = tuple(tuple(output.shape[1:]) for output in encoder_outputs)
        upsample_target_shapes = (shape_before_pooling[0], shape_before_pooling[1])

        return encoder_outputs_shapes, upsample_target_shapes

    def _encoder_blocks_forward(self, x : torch.tensor) :
        """
        Run the three blocks of the encoder and return the output of each of them, in encoder order (temporal filter, spatial filter, separable convolution).
        """

        x_temporal  = self.encoder.temporal_filter(x)
        x_spatial   = self.encoder.spatial_filter(x_temporal)
        x_separable = self.encoder.separable_convolution(x_spatial)

        return x_temporal, x_spatial, x_separable

    def _check_input_shape(self, x : torch.tensor) -> None :
        """
        Raize ``ValueError`` if the geometry of ``x`` is not the one the model was built for.
        """

        if tuple(x.shape[1:]) != (self.depth_first_layer, self.C, self.T) :
            raise ValueError(f"this hvEEGNet was built for an input of shape (batch, {self.depth_first_layer}, {self.C}, {self.T}), got {tuple(x.shape)}")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Decoder

    def _get_h(self, h : torch.tensor, z_0_feature_map : torch.tensor) -> torch.tensor :
        """
        Return the extra input ``h`` of the deepest decoder level, i.e. the tensor given by the user, else the trainable one (if ``use_h_in_decoder``), else a tensor of zeros.
        A tensor with a batch axis of 1 is broadcasted over the batch.
        """

        if h is None :
            if self.use_h_in_decoder : h = self.h
            else : return torch.zeros_like(z_0_feature_map)

        if tuple(h.shape[1:]) != tuple(z_0_feature_map.shape[1:]) :
            raise ValueError(f"h must have shape (batch, {', '.join(str(axis_size) for axis_size in z_0_feature_map.shape[1:])}) (or a batch axis of 1), got {tuple(h.shape)}")

        return h.expand(z_0_feature_map.shape[0], *h.shape[1:]) if h.shape[0] == 1 else h

    def _decode(self, z_0 : torch.tensor, encoder_outputs = None, use_posterior = (True, True), h : torch.tensor = None) :
        """
        Run the decoder from the deepest latent variable, sampling the two shallower latent spaces on the way.

        Parameters
        ----------
        z_0 : torch.tensor
            The deepest latent variable, of shape ``(batch, *latent_shapes[0])``.
        encoder_outputs : tuple of torch.tensor, optional
            The outputs of the three encoder blocks, in encoder order, by default None.
            When given, ``z_1`` and ``z_2`` are sampled from the approximate posterior (training / reconstruction); when None they are sampled from the prior (generation).
        use_posterior : tuple of bool, optional
            For ``z_1`` and ``z_2`` respectively, whether to use the approximate posterior (True) or the prior (False), by default ``(True, True)``. Ignored when ``encoder_outputs`` is None.
        h : torch.tensor, optional
            Extra input of the deepest decoder level, with the shape of the encoder output. By default None, see :meth:`_get_h`.

        Returns
        -------
        x_reconstructed : torch.tensor
            The decoded signal.
        prior_mu_list, prior_log_var_list : list of torch.tensor
            The parameters of the priors of ``z_1`` and ``z_2``.
        delta_mu_list, delta_log_var_list : list of torch.tensor
            The posterior shifts of the levels where the posterior was used (empty during generation).
        z_list : list of torch.tensor
            The sampled ``z_1`` and ``z_2``.
        """

        prior_mu_list, prior_log_var_list = [], []
        delta_mu_list, delta_log_var_list = [], []
        z_list = []

        # The decoder blocks that follow z_1 and z_2.
        next_decoder_blocks = (self.spatial_filter_transpose, self.temporal_filter_transpose)

        # Encoder features of the same level of z_1 (spatial filter output) and z_2 (temporal filter output).
        if encoder_outputs is not None : encoder_features = (encoder_outputs[1], encoder_outputs[0])

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Deepest level

        # From the latent space to a feature map (an identity with the convolutional mapping), then combined with h.
        x = self.latent_projections[0](z_0)
        x = self.h_decoder_combination(self._get_h(h, x), x)
        x = self.separable_convolution_transpose(x)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Shallower levels (z_1, then z_2)

        for level in range(2) :
            # Prior p(z_l | z_<l), from what has been decoded so far.
            prior_mu, prior_log_var = self.prior_maps[level](x)
            prior_mu_list.append(prior_mu)
            prior_log_var_list.append(prior_log_var)

            if encoder_outputs is not None and use_posterior[level] :
                # Approximate posterior: the prior shifted by a term that also sees the encoder features (Section 3.2 of the NVAE paper).
                delta_mu, delta_log_var = self.posterior_shift_maps[level](self.encoder_decoder_combinations[level](x, encoder_features[level]))
                delta_mu_list.append(delta_mu)
                delta_log_var_list.append(delta_log_var)

                z = reparametrize(prior_mu + delta_mu, prior_log_var + delta_log_var)
            else :
                z = reparametrize(prior_mu, prior_log_var)
            z_list.append(z)

            # Merge the sample (as a feature map) with the decoder features and run the next decoder block.
            x = self.latent_decoder_combinations[level](x, self.latent_projections[level + 1](z))
            x = next_decoder_blocks[level](x)

        return x, prior_mu_list, prior_log_var_list, delta_mu_list, delta_log_var_list, z_list

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Public API

    def forward(self, x : torch.tensor, h : torch.tensor = None) :
        """
        Encode, sample the three latent spaces and decode.

        Parameters
        ----------
        x : torch.tensor
            A batch of EEG trials, of shape ``(batch, depth_first_layer, C, T)``.
        h : torch.tensor, optional
            Extra input of the deepest decoder level (Fig. 2 of the NVAE paper), with the shape of the encoder output. By default None, i.e. the trainable ``h`` if ``use_h_in_decoder``, else zeros.

        Returns
        -------
        x_reconstructed : torch.tensor
            The reconstructed signal, with exactly the shape of ``x``.
        mu_list : list of torch.tensor
            ``[mu_0, mu_1, mu_2]``: the mean of the approximate posterior of ``z_0``, then the means of the priors of ``z_1`` and ``z_2``.
        log_var_list : list of torch.tensor
            The matching log variances.
        delta_mu_list : list of torch.tensor
            ``[delta_mu_1, delta_mu_2]``: the shift of the mean of the priors of ``z_1`` and ``z_2`` that gives the approximate posterior.
        delta_log_var_list : list of torch.tensor
            The matching shifts of the log variances.
        z_list : list of torch.tensor
            ``[z_0, z_1, z_2]``: the sampled latent variables.

        Notes
        -----
        The first five elements follow the output of the original implementation. The order is append-only by design, so ``output[0]`` is always the reconstruction.
        """

        self._check_input_shape(x)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Encoder and deepest latent space
        encoder_outputs = self._encoder_blocks_forward(x)

        mu_0, log_var_0 = self.posterior_map_0(encoder_outputs[2])
        z_0 = reparametrize(mu_0, log_var_0)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Decoder (and shallower latent spaces)
        x_reconstructed, prior_mu_list, prior_log_var_list, delta_mu_list, delta_log_var_list, z_list = self._decode(z_0, encoder_outputs, h = h)

        return x_reconstructed, [mu_0] + prior_mu_list, [log_var_0] + prior_log_var_list, delta_mu_list, delta_log_var_list, [z_0] + z_list

    def encode(self, x : torch.tensor) :
        """
        Compute the parameters of the deepest latent distribution ``q(z_0 | x)`` of a batch of trials.

        Note that the shallower latent spaces are not returned: they depend on the decoder (through their prior), so they are not a function of the encoder alone. Use :meth:`forward` to get all of them.

        Parameters
        ----------
        x : torch.tensor
            A batch of EEG trials.

        Returns
        -------
        mu : torch.tensor
            The mean of the approximate posterior of ``z_0``.
        log_var : torch.tensor
            The logarithm of its variance.
        """

        self._check_input_shape(x)

        return self.posterior_map_0(self._encoder_blocks_forward(x)[2])

    def reconstruct(self, x : torch.tensor, no_grad : bool = True) -> torch.tensor :
        """
        Reconstruct a batch of trials, i.e. encode it and decode it back.

        Note that the reconstruction is stochastic (all the latent variables are *sampled*) and that this method does not change the mode of the model: call ``model.eval()`` first to disable dropout and use the running statistics of the batch normalization.

        Parameters
        ----------
        x : torch.tensor
            A batch of EEG trials.
        no_grad : bool, optional
            If True (the default) the computation is done inside :func:`torch.no_grad`.

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

    def reconstruct_ignoring_latent_spaces(self, x : torch.tensor, latent_spaces_to_ignore) -> torch.tensor :
        """
        Reconstruct a batch of trials ignoring the information that the encoder sends to some latent spaces.

        An ignored latent space is sampled from its prior instead of from the approximate posterior, so the decoder receives no information about ``x`` through it.
        For ``z_0`` the prior is a standard normal; for ``z_1`` and ``z_2`` it is the conditional prior computed by the decoder.
        It is useful to study what each level of the hierarchy encodes (e.g. ignoring ``z_1`` and ``z_2`` shows what the deepest level alone carries).

        Note that the original implementation took the list in encoder order (temporal, spatial, separable) and had no effect for the deepest level. Here the list follows the order of the latent spaces (deepest first) and ignoring ``z_0`` is supported.

        Parameters
        ----------
        x : torch.tensor
            A batch of EEG trials.
        latent_spaces_to_ignore : list of bool
            Three flags, for ``z_0``, ``z_1`` and ``z_2`` respectively. True means that the latent space is ignored.

        Returns
        -------
        torch.tensor
            The reconstructed signal, with the shape of ``x``.

        Raises
        ------
        ValueError
            If ``latent_spaces_to_ignore`` does not contain exactly three booleans.
        """

        latent_spaces_to_ignore = list(latent_spaces_to_ignore)
        if len(latent_spaces_to_ignore) != 3 or not all(isinstance(flag, bool) for flag in latent_spaces_to_ignore) :
            raise ValueError(f"latent_spaces_to_ignore must contain 3 booleans (for z_0, z_1 and z_2), got {latent_spaces_to_ignore!r}")

        self._check_input_shape(x)

        with torch.no_grad() :
            encoder_outputs = self._encoder_blocks_forward(x)

            # Deepest level: posterior, or the standard normal prior.
            if latent_spaces_to_ignore[0] :
                z_0 = torch.randn(x.shape[0], *self.latent_shapes[0], device = x.device, dtype = x.dtype)
            else :
                z_0 = reparametrize(*self.posterior_map_0(encoder_outputs[2]))

            # Shallower levels: the posterior is used only for the latent spaces that are not ignored.
            use_posterior = (not latent_spaces_to_ignore[1], not latent_spaces_to_ignore[2])
            x_reconstructed = self._decode(z_0, encoder_outputs, use_posterior)[0]

        return x_reconstructed

    def generate(self, n_samples : int = 1, z : torch.tensor = None, device = None, no_grad : bool = True, h : torch.tensor = None) -> torch.tensor :
        """
        Generate new synthetic EEG trials.

        The deepest latent variable is drawn from its prior (a standard normal), then the decoder samples ``z_1`` and ``z_2`` from their conditional priors.

        Parameters
        ----------
        n_samples : int, optional
            Number of trials to generate, by default 1. Ignored when ``z`` is given.
        z : torch.tensor, optional
            The deepest latent variables to decode, of shape ``(batch, *latent_shapes[0])``, by default None (they are sampled from the prior).
        device : torch.device or str, optional
            Where the latent variables are created, by default None, i.e. the device of the model. Ignored when ``z`` is given.
        no_grad : bool, optional
            If True (the default) the computation is done inside :func:`torch.no_grad`.
        h : torch.tensor, optional
            Extra input of the deepest decoder level, i.e. a "seed" of the generation, with the shape of the encoder output (a batch axis of 1 is broadcasted). By default None, i.e. the trainable ``h`` if ``use_h_in_decoder``, else zeros.

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

            z = torch.randn(n_samples, *self.latent_shapes[0], device = device)

        if no_grad :
            with torch.no_grad() :
                x_generated = self._decode(z, h = h)[0]
        else :
            x_generated = self._decode(z, h = h)[0]

        return x_generated

    def debug_shape(self, input_size) :
        """
        Method that create a fake input and pass it through the network, showing after each step the shape
        """

        print("Shape tracking across hvEEGNet")

        x = torch.randn(input_size)
        print("\tInput shape :\t\t\t\t", x.shape)

        x_temporal, x_spatial, x_separable = self._encoder_blocks_forward(x)
        print("\tEncoder (temporal filter) :\t\t", x_temporal.shape)
        print("\tEncoder (spatial filter) :\t\t", x_spatial.shape)
        print("\tEncoder (separable conv) :\t\t", x_separable.shape)

        x_reconstructed, mu_list, _, _, _, _ = self.forward(torch.randn(input_size))
        for level, mu in enumerate(mu_list) : print(f"\tLatent space z_{level} :\t\t\t", mu.shape)
        print("\tOutput shape :\t\t\t\t", x_reconstructed.shape)

        # The reconstruction must have exactly the shape of the input.
        print(f"\tInput shape == output shape :\t\t {tuple(x_reconstructed.shape) == tuple(input_size)}\n")
