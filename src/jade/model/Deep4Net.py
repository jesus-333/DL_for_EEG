"""
Implementation of the Deep4Net (Deep ConvNet) model using PyTorch.

The model is the one presented as ``Deep ConvNet`` in ``Deep learning with convolutional neural networks for EEG decoding and visualization`` (Schirrmeister et al., Human Brain Mapping, 2017, https://doi.org/10.1002/hbm.23730), better known through the ``Deep4Net`` name of its reference implementation.
It is the deep counterpart of :class:`~jade.model.ShallowFBCSPNet.ShallowFBCSPNet`: a split temporal + spatial convolution closed by a pooling stage, followed by three convolution + pooling blocks with a growing number of filters (50, 100 and 200).

This implementation follows the structure of the reference code, but is rewritten to fit the ``jade`` conventions:
- it is driven by a validated :class:`~jade.config.model.Deep4Net_cfg.Deep4Net_config` dataclass (like :class:`~jade.model.EEGNet.EEGNet`), so all the defaulting/validation lives in the config;
- it works on the ``jade`` tensor layout ``(batch, depth, C, T)`` (the reference implementation transposes the input to put the time axis second), so the four pooling stages support the custom anti-aliased operators of :mod:`jade.model.pooling_1d` (through :func:`get_deep4net_pooling`) without any axis juggling;
- the classification head is the usual ``jade`` flatten + feedforward layer of :class:`~jade.model.EEGNet.EEGNet_Classifier` instead of the convolutional classifier of the reference implementation (the two are equivalent: a convolution spanning the whole remaining time axis is a linear map of the flattened features);
- it depends only on ``torch`` (no ``einops``, no ``braindecode``).

Two options of the reference implementation are deliberately **not** reproduced.
The first is ``stride_before_pool``, which moves the decimation from the pooling to the convolution and leaves the pooling with stride 1: an overlapping, non decimating pooling has no counterpart in :mod:`jade.model.pooling_1d`, whose operators are decimators by construction, so supporting it would make half of the pooling ablation meaningless.
The second is the ``mapping`` dictionary, which only exists to load the state dict of an older version of the reference code.

Note that, unlike the reference implementation, the weights are left to the PyTorch defaults (the reference applies a Xavier uniform initialization), consistently with the rest of :mod:`jade.model`.

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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Pooling factory

def get_deep4net_pooling(config) -> nn.Module :
    """
    Build the pooling layer of one of the four Deep4Net blocks.

    This is the bridge between a single stage of the ``[pooling]`` section of :class:`~jade.config.model.Deep4Net_cfg.Deep4Net_config` and the 1D pooling family in :mod:`jade.model.pooling_1d`.
    Unlike :func:`~jade.model.ShallowFBCSPNet.get_shallowfbcspnet_pooling`, the original pooling here is **non overlapping** (a window of ``decimation_factor`` samples every ``decimation_factor``), so the ``'default'`` type and the 1D operators produce the same output length and the stages stay interchangeable.

    The ``pooling_type`` selects the behaviour:

    - ``'default'`` reproduces the original Deep4Net pooling, an ``nn.MaxPool2d`` (or ``nn.AvgPool2d``, see ``pool_mode``) with kernel ``(1, decimation_factor)`` (pooling along the time axis only).
    - one of the 1D pooling names (``'avg'``, ``'max'``, ``'blur'``, ``'sinc'``, ``'dwt'`` and their aliases) returns the matching ``{name}_multichannel`` wrapper of :mod:`jade.model.pooling_1d`, applied channel-wise (through :func:`~jade.model.support_functions.get_multichannel_pooling`).
    - ``'identity'`` / ``'none'``, or a negative decimation factor, returns ``nn.Identity`` (the pooling stage is disabled).

    Parameters
    ----------
    config : pooling_config or dict
        The configuration of a single pooling stage, as assembled by the ``pooling_layer_N_config`` properties of :class:`~jade.config.model.Deep4Net_cfg.deep4net_pooling_config`:
        a :class:`~jade.config.model.pooling_1d_cfg.pooling_config` dataclass for an active 1D operator, a plain ``{pooling_type, decimation_factor, pool_mode, ...}`` dict otherwise.

    Returns
    -------
    pooling_module : nn.Module
        The pooling layer to drop into the block.
    """

    # A 1D operator arrives as an already-validated pooling_config dataclass, so it can be handed straight
    # to the shared channel-wise factory (which accepts both forms).
    if not isinstance(config, dict) :
        return support_functions.get_multichannel_pooling(config)

    pooling_type = str(config.get("pooling_type", "default")).lower()
    factor       = config.get("decimation_factor", None)

    # Disabled stage: explicit identity / none, or a non-positive decimation factor.
    if pooling_type in ("identity", "none") or (factor is not None and factor < 1) :
        return nn.Identity()

    # Original Deep4Net behaviour: non overlapping 2D pooling along the time axis only.
    if pooling_type == "default" :
        pooling_class = nn.AvgPool2d if config.get("pool_mode", "max") == "mean" else nn.MaxPool2d
        return pooling_class(kernel_size = (1, int(factor)))

    # Any other 1D operator that reached us as a raw dict.
    return support_functions.get_multichannel_pooling(config)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Building blocks

class conv_pool_block(nn.Module) :
    """
    One of the three deeper convolution + pooling blocks of Deep4Net.

    The block chains dropout, a temporal convolution, an optional batch normalization, the activation, the pooling stage and the post-pooling nonlinearity, in this order (the dropout comes first, as in the reference implementation).

    Parameters
    ----------
    config : Deep4Net_config
        The (already validated) model configuration, read for the options shared by every block (activation, post-pooling nonlinearity, batch normalization and dropout).
    n_filters_in : int
        Number of feature maps entering the block.
    n_filters_out : int
        Number of feature maps produced by the convolution of the block.
    filter_length : int
        Length (in time samples) of the convolution kernel of the block, i.e. the kernel is ``(1, filter_length)``.
    pooling_layer : nn.Module
        The pooling layer of the block, already built by :func:`get_deep4net_pooling`.
    """

    def __init__(self, config, n_filters_in : int, n_filters_out : int, filter_length : int, pooling_layer : nn.Module) :
        super().__init__()

        # The batch normalization makes the bias of the convolution that precedes it redundant.
        self.block = nn.Sequential(
            support_functions.get_dropout(config.prob_dropout, config.use_dropout_2d),
            nn.Conv2d(n_filters_in, n_filters_out, (1, filter_length), bias = not config.batch_norm),
            nn.BatchNorm2d(n_filters_out, momentum = config.batch_norm_alpha, affine = True, eps = 1e-5) if config.batch_norm else nn.Identity(),
            support_functions.get_activation(config.activation),
            pooling_layer,
            support_functions.get_nonlinearity(config.pool_nonlin),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        return self.block(x)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Network declaration

class Deep4Net(nn.Module) :
    """
    Deep4Net feature extractor (four convolution + pooling blocks).

    Analogous to :class:`~jade.model.EEGNet.EEGNet`, this is the feature-extractor part of the model: the final classification layer is **not** part of it (see :class:`Deep4Net_Classifier`).
    It turns a raw EEG tensor into a stack of feature maps, which can optionally be flattened into a feature vector.

    Parameters
    ----------
    config : Deep4Net_config or dict
        The network configuration, an instance of :class:`~jade.config.model.Deep4Net_cfg.Deep4Net_config` (a dict is accepted too and promoted automatically through :func:`~jade.config.model.get_model_config`).
        Every field is documented on that config class; the fields validation (and defaulting) happens there, so this constructor just reads the already-validated attributes.
    """

    def __init__(self, config) :
        super().__init__()

        # Accept a dict (or an already-built config) and promote it to a validated Deep4Net_config.
        config = get_model_config(config)

        # - - - - - - - - - - - - - - - - - - - - - - - - -
        # First block (split temporal + spatial convolution)

        # As in ShallowFBCSPNet the first layer is split in two convolutions with no nonlinearity in
        # between: a temporal one and a spatial one, whose (C, 1) kernel collapses the EEG channel axis
        # to 1. When the split is disabled a single (C, filter_time_length) kernel does both at once.
        if config.split_first_layer :
            convolutional_layers = [
                nn.Conv2d(config.depth_first_layer, config.n_filters_time, (1, config.filter_time_length), bias = True),
                nn.Conv2d(config.n_filters_time, config.n_filters_spat, (config.C, 1), bias = not config.batch_norm),
            ]
            n_filters_conv = config.n_filters_spat
        else :
            convolutional_layers = [
                nn.Conv2d(config.depth_first_layer, config.n_filters_time, (config.C, config.filter_time_length), bias = not config.batch_norm),
            ]
            n_filters_conv = config.n_filters_time

        # Note that the first block has no dropout (it is the only difference in shape with the three
        # conv_pool_block that follow it, which is why it is assembled here instead of reusing that class).
        self.first_block = nn.Sequential(
            *convolutional_layers,
            nn.BatchNorm2d(n_filters_conv, momentum = config.batch_norm_alpha, affine = True, eps = 1e-5) if config.batch_norm else nn.Identity(),
            support_functions.get_activation(config.activation),
            get_deep4net_pooling(config.pooling.pooling_layer_1_config),
            support_functions.get_nonlinearity(config.pool_nonlin),
        )

        # - - - - - - - - - - - - - - - - - - - - - - - - -
        # Three deeper convolution + pooling blocks

        self.second_block = conv_pool_block(config, n_filters_conv, config.n_filters_2, config.filter_length_2, get_deep4net_pooling(config.pooling.pooling_layer_2_config))
        self.third_block  = conv_pool_block(config, config.n_filters_2, config.n_filters_3, config.filter_length_3, get_deep4net_pooling(config.pooling.pooling_layer_3_config))
        self.fourth_block = conv_pool_block(config, config.n_filters_3, config.n_filters_4, config.filter_length_4, get_deep4net_pooling(config.pooling.pooling_layer_4_config))

        self.flatten_output = config.flatten_output

        # - - - - - - - - - - - - - - - - - - - - - - - - -
        # Print additional information after the creation of the network
        if config.print_var :
            print("Deep4Net Created. Number of parameters:")
            print(f"\tNumber of trainable parameters (First block)  = {support_functions.count_trainable_parameters(self.first_block)}")
            print(f"\tNumber of trainable parameters (Second block) = {support_functions.count_trainable_parameters(self.second_block)}")
            print(f"\tNumber of trainable parameters (Third block)  = {support_functions.count_trainable_parameters(self.third_block)}")
            print(f"\tNumber of trainable parameters (Fourth block) = {support_functions.count_trainable_parameters(self.fourth_block)}\n")
            self.debug_shape(config.input_size)

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        # - - - - - - - - - - - - - - - - - - - - - - - - -
        # First block : (B, depth, C, T) -> (B, n_filters_conv, 1, T_1)
        x = self.first_block(x)

        # - - - - - - - - - - - - - - - - - - - - - - - - -
        # Three deeper blocks, each one shrinking the time axis further
        x = self.second_block(x)
        x = self.third_block(x)
        x = self.fourth_block(x)

        # (OPTIONAL) Flat the output
        if self.flatten_output : return x.flatten(1)
        else : return x

    def debug_shape(self, input_size) :
        """
        Method that create a fake input and pass it through the network, showing after each step the shape.
        """

        print("Shape tracking across Deep4Net")

        x = torch.randn(input_size)
        print("\tInput shape :\t\t\t", x.shape)

        x = self.first_block(x)
        print("\tFirst block :\t\t\t", x.shape)

        x = self.second_block(x)
        print("\tSecond block :\t\t\t", x.shape)

        x = self.third_block(x)
        print("\tThird block :\t\t\t", x.shape)

        x = self.fourth_block(x)
        print("\tFourth block :\t\t\t", x.shape)

        print("\t(OPTIONAL) Flatten :\t\t", x.flatten(1).shape, "\n")


class Deep4Net_Classifier(nn.Module) :
    """
    Deep4Net feature extractor plus a feedforward classification head.

    The head flattens the feature maps of the last block and maps them to the classes with a single linear layer, ending with a :class:`torch.nn.LogSoftmax` (when ``config.use_softmax`` is True, so the output are log-probabilities) or a :class:`torch.nn.Identity` (when False, so the output are raw logits).
    The choice must stay consistent with the training loss, see :func:`~jade.training.support_training.cross_check_config`.

    This replaces the convolutional classifier of the reference implementation, which convolves the whole remaining time axis with a ``(final_conv_length, 1)`` kernel: the two are the same linear map, written differently.

    Parameters
    ----------
    config : Deep4Net_config or dict
        The network configuration, an instance of :class:`~jade.config.model.Deep4Net_cfg.Deep4Net_config` (a dict is accepted too and promoted automatically).
        The classifier reads ``input_size`` (to size the feedforward layer with a dummy forward pass), ``n_classes`` and ``use_softmax`` on top of the fields used by :class:`Deep4Net`.
    """

    def __init__(self, config) :
        super().__init__()

        # Accept a dict (or an already-built config) and promote it to a validated Deep4Net_config.
        config = get_model_config(config)

        self.deep4net = Deep4Net(config)

        # The head needs flat (batch, features) input, so the wrapped extractor must flatten regardless of
        # the config flatten_output (which only concerns the standalone Deep4Net feature extractor).
        self.deep4net.flatten_output = True

        input_neurons   = self.compute_number_of_neurons(config.input_size)
        self.classifier = nn.Sequential(
            nn.Linear(input_neurons, config.n_classes),
            nn.LogSoftmax(dim = 1) if config.use_softmax else nn.Identity(),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        x = self.deep4net(x)

        x = self.classifier(x)

        return x

    def compute_number_of_neurons(self, input_size) :
        """
        Compute the total number of neurons for the feedforward layer of the head.
        """

        x = torch.rand(input_size)
        x = self.deep4net(x)
        input_neurons = len(x.flatten())

        return input_neurons

    def classify(self, x, return_as_index = True) :
        """
        Directly classify an input by returning the label (return_as_index = True) or the probability distribution on the labels (return_as_index = False).
        """

        label = self.forward(x)

        if return_as_index :
            # Note: do NOT squeeze here. A batch of size 1 has shape (1, n_classes) and torch.squeeze would
            # collapse the batch dimension to (n_classes,), making the following argmax(dim = 1) fail.
            # Keeping the (batch, n_classes) shape works for any batch size (exp is monotonic, so it does not
            # change the argmax result for either the LogSoftmax or the Identity/logits case).
            predict_prob = torch.exp(label).detach()
            label = torch.argmax(predict_prob, dim = 1)

        return label
