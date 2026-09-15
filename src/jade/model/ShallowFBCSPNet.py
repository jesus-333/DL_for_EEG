"""
Implementation of the ShallowFBCSPNet (Shallow ConvNet) model using PyTorch.

The model is the one presented as ``Shallow ConvNet`` in ``Deep learning with convolutional neural networks for EEG decoding and visualization`` (Schirrmeister et al., Human Brain Mapping, 2017, https://doi.org/10.1002/hbm.23730), better known through the ``ShallowFBCSPNet`` name of its reference implementation.
It is a shallow network inspired by the FBCSP pipeline: a temporal convolution and a spatial convolution are followed by a squaring nonlinearity, a pooling stage and a logarithm, which together compute the log-variance of the band-limited, spatially filtered signal.
The two model-specific nonlinearities live in :mod:`jade.model.support_functions` (:class:`~jade.model.support_functions.square` and :class:`~jade.model.support_functions.safe_log`, selected through :func:`~jade.model.support_functions.get_nonlinearity`) because :class:`~jade.model.Deep4Net.Deep4Net` can use them too.

This implementation follows the structure of the reference code, but is rewritten to fit the ``jade`` conventions:
- it is driven by a validated :class:`~jade.config.model.ShallowFBCSPNet_cfg.ShallowFBCSPNet_config` dataclass (like :class:`~jade.model.EEGNet.EEGNet`), so all the defaulting/validation lives in the config;
- it works on the ``jade`` tensor layout ``(batch, depth, C, T)`` (the reference implementation transposes the input to put the time axis second), so the pooling stage supports the custom anti-aliased operators of :mod:`jade.model.pooling_1d` (through :func:`get_shallowfbcspnet_pooling`) without any axis juggling;
- the classification head is the usual ``jade`` flatten + feedforward layer of :class:`~jade.model.EEGNet.EEGNet_Classifier` instead of the convolutional classifier of the reference implementation (the two are equivalent: a convolution spanning the whole remaining time axis is a linear map of the flattened features);
- it depends only on ``torch`` (no ``einops``, no ``braindecode``).

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

def get_shallowfbcspnet_pooling(config) -> nn.Module :
    """
    Build the pooling layer of ShallowFBCSPNet.

    This is the bridge between the ``[pooling]`` section of :class:`~jade.config.model.ShallowFBCSPNet_cfg.ShallowFBCSPNet_config` and the 1D pooling family in :mod:`jade.model.pooling_1d`.
    The stage sits between the squaring nonlinearity and the logarithm, so its operator (and decimation factor) directly control the time resolution of the log-variance features the classifier sees.

    The ``pooling_type`` selects the behaviour:

    - ``'default'`` reproduces the original ShallowFBCSPNet pooling, an **overlapping** ``nn.AvgPool2d`` (or ``nn.MaxPool2d``, see ``pool_mode``) with kernel ``(1, pooling_kernel)`` and stride ``(1, pooling_stride)`` (pooling along the time axis only).
    - one of the 1D pooling names (``'avg'``, ``'max'``, ``'blur'``, ``'sinc'``, ``'dwt'`` and their aliases) returns the matching ``{name}_multichannel`` wrapper of :mod:`jade.model.pooling_1d`, applied channel-wise with decimation factor ``pooling_stride`` (through :func:`~jade.model.support_functions.get_multichannel_pooling`).
    - ``'identity'`` / ``'none'`` returns ``nn.Identity`` (the pooling stage is disabled).

    Parameters
    ----------
    config : shallowfbcspnet_pooling_config or dict
        The pooling section, either a :class:`~jade.config.model.ShallowFBCSPNet_cfg.shallowfbcspnet_pooling_config` dataclass or a plain dict (promoted to one).

    Returns
    -------
    pooling_module : nn.Module
        The pooling layer to drop into the network.
    """

    # Accept a dict (or an already-built config) and promote it to a validated shallowfbcspnet_pooling_config.
    # Lazy import to keep the dependency direction one-way (model -> config) and avoid any import cycle.
    from jade.config.model import shallowfbcspnet_pooling_config
    if isinstance(config, dict) :
        config = shallowfbcspnet_pooling_config.from_dict(config)

    pooling_type = str(config.pooling_type).lower()

    # Disabled stage: explicit identity / none.
    if pooling_type in ("identity", "none") :
        return nn.Identity()

    # Original ShallowFBCSPNet behaviour: overlapping 2D pooling along the time axis only.
    if pooling_type == "default" :
        pooling_class = nn.MaxPool2d if config.pool_mode == "max" else nn.AvgPool2d
        return pooling_class(kernel_size = (1, config.pooling_kernel), stride = (1, config.pooling_stride))

    # 1D anti-aliased operator applied channel-wise (the config exposes it as a ready pooling_config).
    return support_functions.get_multichannel_pooling(config.pooling_layer_config)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Network declaration

class ShallowFBCSPNet(nn.Module) :
    """
    ShallowFBCSPNet feature extractor (convolutional filter + squaring, pooling and logarithm).

    Analogous to :class:`~jade.model.EEGNet.EEGNet`, this is the feature-extractor part of the model: the final classification layer is **not** part of it (see :class:`ShallowFBCSPNet_Classifier`).
    It turns a raw EEG tensor into a stack of log-variance feature maps, which can optionally be flattened into a feature vector.

    Parameters
    ----------
    config : ShallowFBCSPNet_config or dict
        The network configuration, an instance of :class:`~jade.config.model.ShallowFBCSPNet_cfg.ShallowFBCSPNet_config` (a dict is accepted too and promoted automatically through :func:`~jade.config.model.get_model_config`).
        Every field is documented on that config class; the fields validation (and defaulting) happens there, so this constructor just reads the already-validated attributes.
    """

    def __init__(self, config) :
        super().__init__()

        # Accept a dict (or an already-built config) and promote it to a validated ShallowFBCSPNet_config.
        config = get_model_config(config)

        conv_nonlin   = support_functions.get_nonlinearity(config.conv_nonlin)
        pool_nonlin   = support_functions.get_nonlinearity(config.pool_nonlin)
        pooling_layer = get_shallowfbcspnet_pooling(config.pooling)
        dropout       = support_functions.get_dropout(config.prob_dropout, config.use_dropout_2d)

        # - - - - - - - - - - - - - - - - - - - - - - - - -
        # Convolutional filter (temporal + spatial)

        # The original first layer is split in two convolutions with no nonlinearity in between: a temporal
        # one (a filter bank over the time axis) and a spatial one, whose (C, 1) kernel collapses the EEG
        # channel axis to 1 (the learnt analogue of the CSP spatial filters). When the split is disabled a
        # single (C, filter_time_length) kernel does both at once, which is what the reference implementation
        # obtains by convolving the channel axis directly.
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

        # The batch normalization makes the bias of the convolution that precedes it redundant (hence the
        # bias flags above), so it is the last layer before the squaring nonlinearity.
        self.convolutional_filter = nn.Sequential(
            *convolutional_layers,
            nn.BatchNorm2d(n_filters_conv, momentum = config.batch_norm_alpha, affine = True) if config.batch_norm else nn.Identity(),
            conv_nonlin,
        )

        # - - - - - - - - - - - - - - - - - - - - - - - - -
        # Pooling stage (squared samples -> log-variance features)

        self.pooling_block = nn.Sequential(
            pooling_layer,
            pool_nonlin,
            dropout,
        )

        self.flatten_output = config.flatten_output

        # - - - - - - - - - - - - - - - - - - - - - - - - -
        # Print additional information after the creation of the network
        if config.print_var :
            print("ShallowFBCSPNet Created. Number of parameters:")
            print(f"\tNumber of trainable parameters (Convolutional filter) = {support_functions.count_trainable_parameters(self.convolutional_filter)}")
            print(f"\tNumber of trainable parameters (Pooling block)        = {support_functions.count_trainable_parameters(self.pooling_block)}\n")
            self.debug_shape(config.input_size)

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        # - - - - - - - - - - - - - - - - - - - - - - - - -
        # Convolutional filter : (B, depth, C, T) -> (B, n_filters_conv, 1, T_conv)
        x = self.convolutional_filter(x)

        # - - - - - - - - - - - - - - - - - - - - - - - - -
        # Pooling stage : (B, n_filters_conv, 1, T_conv) -> (B, n_filters_conv, 1, T_pool)
        x = self.pooling_block(x)

        # (OPTIONAL) Flat the output
        if self.flatten_output : return x.flatten(1)
        else : return x

    def debug_shape(self, input_size) :
        """
        Method that create a fake input and pass it through the network, showing after each step the shape.
        """

        print("Shape tracking across ShallowFBCSPNet")

        x = torch.randn(input_size)
        print("\tInput shape :\t\t\t", x.shape)

        x = self.convolutional_filter(x)
        print("\tConvolutional filter :\t\t", x.shape)

        x = self.pooling_block(x)
        print("\tPooling block :\t\t\t", x.shape)

        print("\t(OPTIONAL) Flatten :\t\t", x.flatten(1).shape, "\n")


class ShallowFBCSPNet_Classifier(nn.Module) :
    """
    ShallowFBCSPNet feature extractor plus a feedforward classification head.

    The head flattens the log-variance feature maps and maps them to the classes with a single linear layer, ending with a :class:`torch.nn.LogSoftmax` (when ``config.use_softmax`` is True, so the output are log-probabilities) or a :class:`torch.nn.Identity` (when False, so the output are raw logits).
    The choice must stay consistent with the training loss, see :func:`~jade.training.support_training.cross_check_config`.

    This replaces the convolutional classifier of the reference implementation, which convolves the whole remaining time axis with a ``(final_conv_length, 1)`` kernel: the two are the same linear map, written differently.

    Parameters
    ----------
    config : ShallowFBCSPNet_config or dict
        The network configuration, an instance of :class:`~jade.config.model.ShallowFBCSPNet_cfg.ShallowFBCSPNet_config` (a dict is accepted too and promoted automatically).
        The classifier reads ``input_size`` (to size the feedforward layer with a dummy forward pass), ``n_classes`` and ``use_softmax`` on top of the fields used by :class:`ShallowFBCSPNet`.
    """

    def __init__(self, config) :
        super().__init__()

        # Accept a dict (or an already-built config) and promote it to a validated ShallowFBCSPNet_config.
        config = get_model_config(config)

        self.shallowfbcspnet = ShallowFBCSPNet(config)

        # The head needs flat (batch, features) input, so the wrapped extractor must flatten regardless of
        # the config flatten_output (which only concerns the standalone ShallowFBCSPNet feature extractor).
        self.shallowfbcspnet.flatten_output = True

        input_neurons   = self.compute_number_of_neurons(config.input_size)
        self.classifier = nn.Sequential(
            nn.Linear(input_neurons, config.n_classes),
            nn.LogSoftmax(dim = 1) if config.use_softmax else nn.Identity(),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        x = self.shallowfbcspnet(x)

        x = self.classifier(x)

        return x

    def compute_number_of_neurons(self, input_size) :
        """
        Compute the total number of neurons for the feedforward layer of the head.
        """

        x = torch.rand(input_size)
        x = self.shallowfbcspnet(x)
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
