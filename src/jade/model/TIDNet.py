"""
Implementation of the TIDNet (Thinker Invariance DenseNet) model using PyTorch.

The model is the one presented in ``Thinker invariance: enabling deep neural networks for BCI across more people`` (Kostas & Rudzicz, Journal of Neural Engineering, 2020, https://doi.org/10.1088/1741-2552/abb7a7).
It chains a residual dilated temporal filter, a decimating pooling stage, a densely connected spatial filter that collapses the EEG-channel axis, and an adaptive summary of the time axis.

This implementation follows the structure of the authors' reference code, but is rewritten to fit the ``jade`` conventions:
- it is driven by a validated :class:`~jade.config.model.TIDNet_cfg.TIDNet_config` dataclass (like :class:`~jade.model.EEGNet.EEGNet`), so all the defaulting/validation lives in the config;
- it works on the ``jade`` tensor layout ``(batch, depth, C, T)`` (the reference implementation expands a ``(batch, C, T)`` tensor internally), so the decimating pooling stage supports the custom anti-aliased operators of :mod:`jade.model.pooling_1d` (through :func:`get_tidnet_pooling`);
- the auxiliary subject and run classification heads of the original (used there for the domain-adaptation experiments) are dropped, so :class:`TIDNet_Classifier` returns a plain tensor like every other ``jade`` model and plugs into :mod:`jade.training` unchanged.

Note that the adaptive average pooling that summarizes the time axis before the head is **not** part of the ``[pooling]`` section: it is an adaptive resize, not a decimation, so it has no anti-aliasing counterpart in :mod:`jade.model.pooling_1d`.
Note also that, unlike the reference implementation, the weights are left to the PyTorch defaults (the reference applies a Xavier normal initialization to the classifier), consistently with the rest of :mod:`jade.model`.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

from __future__ import annotations

# Full module imports
import math

# Third party imports
import torch
from torch import nn

# Internal imports
from jade.config.model import get_model_config
from jade.model import support_functions

# The parametrization-based weight normalization replaces the (deprecated) torch.nn.utils.weight_norm.
try :
    from torch.nn.utils.parametrizations import weight_norm
except ImportError :
    from torch.nn.utils import weight_norm

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Pooling factory

def get_tidnet_pooling(config) -> nn.Module :
    """
    Build the decimating pooling layer that closes the temporal block of TIDNet.

    This is the bridge between the ``[pooling]`` section of :class:`~jade.config.model.TIDNet_cfg.TIDNet_config` and the 1D pooling family in :mod:`jade.model.pooling_1d`.
    The original pooling is **non overlapping** (a window of ``decimation_factor`` samples every ``decimation_factor``), so the ``'default'`` type and the 1D operators produce the same output length.

    The ``pooling_type`` selects the behaviour:

    - ``'default'`` reproduces the original TIDNet pooling, an ``nn.MaxPool2d`` (or ``nn.AvgPool2d``, see ``pool_mode``) with kernel ``(1, decimation_factor)`` (pooling along the time axis only).
    - one of the 1D pooling names (``'avg'``, ``'max'``, ``'blur'``, ``'sinc'``, ``'dwt'`` and their aliases) returns the matching ``{name}_multichannel`` wrapper of :mod:`jade.model.pooling_1d`, applied channel-wise (through :func:`~jade.model.support_functions.get_multichannel_pooling`).
    - ``'identity'`` / ``'none'`` returns ``nn.Identity`` (the pooling stage is disabled).

    Parameters
    ----------
    config : pooling_config or dict
        The pooling stage, as assembled by :attr:`~jade.config.model.TIDNet_cfg.tidnet_pooling_config.pooling_layer_config`:
        a :class:`~jade.config.model.pooling_1d_cfg.pooling_config` dataclass for an active 1D operator, a plain ``{pooling_type, decimation_factor, pool_mode, ...}`` dict otherwise.

    Returns
    -------
    pooling_module : nn.Module
        The pooling layer to drop into the temporal block.
    """

    # A 1D operator arrives as an already-validated pooling_config dataclass, so it can be handed straight
    # to the shared channel-wise factory (which accepts both forms).
    if not isinstance(config, dict) :
        return support_functions.get_multichannel_pooling(config)

    pooling_type = str(config.get("pooling_type", "default")).lower()
    factor       = config.get("decimation_factor", None)

    # Disabled stage: explicit identity / none.
    if pooling_type in ("identity", "none") :
        return nn.Identity()

    # Original TIDNet behaviour: non overlapping 2D pooling along the time axis only.
    if pooling_type == "default" :
        pooling_class = nn.AvgPool2d if config.get("pool_mode", "max") == "mean" else nn.MaxPool2d
        return pooling_class(kernel_size = (1, int(factor)))

    # Any other 1D operator that reached us as a raw dict.
    return support_functions.get_multichannel_pooling(config)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Building blocks

class batch_norm_zg(nn.BatchNorm2d) :
    """
    Batch normalization whose affine parameters are initialized to zero ("zero gamma").

    It is used by the residual variant of :class:`conv_block_2d`: starting with a zero scale makes the block an identity at initialization, so the residual path carries the whole signal and the training of a deep stack starts from a well behaved point.
    """

    def reset_parameters(self) -> None :
        if self.track_running_stats :
            self.running_mean.zero_()
            self.running_var.fill_(1)

        if self.affine :
            self.weight.data.zero_()
            self.bias.data.zero_()


class conv_block_2d(nn.Module) :
    """
    Convolutional block of TIDNet, in the order convolution, dropout, activation, batch normalization.

    That order is the one of the reference implementation (the normalization comes **after** the activation, not before it) and it is kept as-is to reproduce the original model.

    Parameters
    ----------
    n_filters_in : int
        Number of feature maps entering the block.
    n_filters_out : int
        Number of feature maps produced by the convolution.
    kernel : tuple of int
        Convolution kernel, as a ``(height, width)`` pair on the ``(C, T)`` axes.
    activation : str, default 'leaky_relu'
        Name of the activation function, resolved by :func:`~jade.model.support_functions.get_activation`.
    prob_dropout : float, default 0.5
        Dropout probability of the :class:`torch.nn.Dropout2d` that follows the convolution.
    batch_norm : bool, default True
        If True a batch normalization closes the block (and the convolution drops its bias term, which the normalization would make redundant).
    residual : bool, default False
        If True the input is added to the output of the block, and the batch normalization is the zero-initialized :class:`batch_norm_zg`.
        The convolution must then preserve both the shape and the number of feature maps.
    """

    def __init__(self, n_filters_in : int, n_filters_out : int, kernel, activation : str = "leaky_relu", prob_dropout : float = 0.5, batch_norm : bool = True, residual : bool = False) :
        super().__init__()

        self.residual = residual

        if residual      : normalization = batch_norm_zg(n_filters_out)
        elif batch_norm  : normalization = nn.BatchNorm2d(n_filters_out)
        else             : normalization = nn.Identity()

        self.block = nn.Sequential(
            nn.Conv2d(n_filters_in, n_filters_out, kernel, bias = not batch_norm),
            nn.Dropout2d(prob_dropout),
            support_functions.get_activation(activation),
            normalization,
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        y = self.block(x)

        return y + x if self.residual else y


class dense_filter(nn.Module) :
    """
    Single layer of the densely connected spatial filter of TIDNet.

    Following the DenseNet recipe the layer normalizes and activates its input, squeezes it through a ``1x1`` bottleneck convolution, convolves along the EEG-channel axis and **concatenates** the ``growth_rate`` feature maps it produces to its own input, so every layer sees every previous one.

    Parameters
    ----------
    n_features_in : int
        Number of feature maps entering the layer.
    growth_rate : int
        Number of feature maps the layer appends to its input.
    filter_len : int, default 5
        Length of the convolution kernel along the EEG-channel axis (the kernel is ``(filter_len, 1)``, padded to preserve that axis).
    prob_dropout : float, default 0.5
        Dropout probability of the :class:`torch.nn.Dropout2d` that closes the layer.
    bottleneck : int, default 2
        Bottleneck factor: the ``1x1`` convolution expands to ``bottleneck * growth_rate`` feature maps before the spatial convolution reduces them to ``growth_rate``.
    activation : str, default 'leaky_relu'
        Name of the activation function, resolved by :func:`~jade.model.support_functions.get_activation`.
    """

    def __init__(self, n_features_in : int, growth_rate : int, filter_len : int = 5, prob_dropout : float = 0.5, bottleneck : int = 2, activation : str = "leaky_relu") :
        super().__init__()

        # The kernel spans the EEG-channel axis only, and is padded so that axis keeps its size (the
        # concatenation below needs the two tensors to agree on every axis but the feature one).
        kernel = (filter_len, 1)

        self.filter = nn.Sequential(
            nn.BatchNorm2d(n_features_in),
            support_functions.get_activation(activation),
            nn.Conv2d(n_features_in, bottleneck * growth_rate, 1),
            nn.BatchNorm2d(bottleneck * growth_rate),
            support_functions.get_activation(activation),
            nn.Conv2d(bottleneck * growth_rate, growth_rate, kernel, padding = tuple(k // 2 for k in kernel)),
            nn.Dropout2d(prob_dropout),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        return torch.cat((x, self.filter(x)), dim = 1)


class dense_spatial_filter(nn.Module) :
    """
    Densely connected spatial filter of TIDNet.

    A stack of :class:`dense_filter` layers mixes the EEG channels while growing the number of feature maps, then a convolution with a ``(C, 1)`` kernel collapses the EEG-channel axis to a single row, which is squeezed away.

    Parameters
    ----------
    C : int
        Number of EEG channels of the input (the height of the collapsing kernel).
    growth : int
        Growth rate of every dense layer.
    depth : int
        Number of dense layers of the stack. The block outputs ``n_features_in + growth * depth`` feature maps.
    n_features_in : int, default 1
        Number of feature maps entering the block.
    bottleneck : int, default 4
        Bottleneck factor of every dense layer.
    prob_dropout : float, default 0.0
        Dropout probability of every dense layer.
    activation : str, default 'leaky_relu'
        Name of the activation function, resolved by :func:`~jade.model.support_functions.get_activation`.
    """

    def __init__(self, C : int, growth : int, depth : int, n_features_in : int = 1, bottleneck : int = 4, prob_dropout : float = 0.0, activation : str = "leaky_relu") :
        super().__init__()

        self.filter = nn.Sequential(*[
            dense_filter(n_features_in + growth * i, growth, prob_dropout = prob_dropout, bottleneck = bottleneck, activation = activation)
            for i in range(depth)
        ])

        n_filters = n_features_in + growth * depth

        self.channel_collapse = conv_block_2d(n_filters, n_filters, (C, 1), activation = activation, prob_dropout = 0.0)

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        x = self.filter(x)

        # The collapsing kernel spans the whole EEG-channel axis, which therefore becomes 1 and is dropped:
        # (B, F, C, T) -> (B, F, 1, T) -> (B, F, T).
        return self.channel_collapse(x).squeeze(-2)


class temporal_filter(nn.Module) :
    """
    Residual dilated temporal filter of TIDNet.

    A stack of weight-normalized convolutions along the time axis, with a decreasing dilation (the first one is the most dilated), so the receptive field spans a long context without downsampling.
    The whole stack is wrapped in a residual connection whose ``1x1`` convolution only adapts the number of feature maps (the ``netwise`` residual style of the reference implementation).

    Parameters
    ----------
    n_features_in : int
        Number of feature maps entering the block.
    n_filters : int
        Number of filters of every convolution of the stack.
    depth : int
        Number of stacked convolutions. The dilation of the ``i``-th one is ``depth - i``.
    temp_len : int
        Length of the convolution kernel along the time axis. It is rounded up to the next odd value so that, with the dilation-aware padding, the time axis keeps its size.
    prob_dropout : float, default 0.0
        Dropout probability of the :class:`torch.nn.Dropout2d` that closes every convolution.
    activation : str, default 'leaky_relu'
        Name of the activation function, resolved by :func:`~jade.model.support_functions.get_activation`.
    """

    def __init__(self, n_features_in : int, n_filters : int, depth : int, temp_len : int, prob_dropout : float = 0.0, activation : str = "leaky_relu") :
        super().__init__()

        # An even kernel cannot be padded symmetrically, so it is rounded up to the next odd value.
        temp_len = temp_len + 1 - temp_len % 2

        convolutions = []
        for i in range(depth) :
            dilation = depth - i

            # The padding compensates the dilated kernel, so every convolution preserves the time axis.
            convolution = weight_norm(nn.Conv2d(
                n_features_in if i == 0 else n_filters, n_filters,
                kernel_size = (1, temp_len), dilation = dilation, padding = (0, dilation * (temp_len - 1) // 2),
            ))

            convolutions.append(nn.Sequential(
                convolution,
                support_functions.get_activation(activation),
                nn.Dropout2d(prob_dropout),
            ))

        self.filter = nn.Sequential(*convolutions)

        # The residual path only has to match the number of feature maps (the time axis is preserved above).
        self.residual = nn.Conv2d(n_features_in, n_filters, (1, 1))

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        return self.filter(x) + self.residual(x)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Network declaration

class TIDNet(nn.Module) :
    """
    TIDNet feature extractor (temporal filter, pooling, dense spatial filter and adaptive summary).

    Analogous to :class:`~jade.model.EEGNet.EEGNet`, this is the feature-extractor part of the model: the final classification layer is **not** part of it (see :class:`TIDNet_Classifier`).
    It turns a raw EEG tensor into ``t_filters + s_growth * spat_layers`` feature sequences of length ``summary``, which can optionally be flattened into a feature vector.

    Parameters
    ----------
    config : TIDNet_config or dict
        The network configuration, an instance of :class:`~jade.config.model.TIDNet_cfg.TIDNet_config` (a dict is accepted too and promoted automatically through :func:`~jade.config.model.get_model_config`).
        Every field is documented on that config class; the fields validation (and defaulting) happens there, so this constructor just reads the already-validated attributes.
    """

    def __init__(self, config) :
        super().__init__()

        # Accept a dict (or an already-built config) and promote it to a validated TIDNet_config.
        config = get_model_config(config)

        # The temporal kernel is expressed as a fraction of the input window, so it scales with it.
        self.temp_len = math.ceil(config.temp_span * config.T)

        # - - - - - - - - - - - - - - - - - - - - - - - - -
        # Temporal block (residual dilated filter + decimating pooling)

        self.temporal_block = nn.Sequential(
            temporal_filter(config.depth_first_layer, config.t_filters, config.temp_layers, self.temp_len, activation = config.activation),
            get_tidnet_pooling(config.pooling.pooling_layer_config),
            nn.Dropout2d(config.prob_dropout),
        )

        # - - - - - - - - - - - - - - - - - - - - - - - - -
        # Spatial block (densely connected filter, collapses the EEG-channel axis)

        self.spatial_block = dense_spatial_filter(
            config.C, config.s_growth, config.spat_layers,
            n_features_in = config.t_filters, bottleneck = config.bottleneck, prob_dropout = config.prob_dropout, activation = config.activation,
        )

        # - - - - - - - - - - - - - - - - - - - - - - - - -
        # Adaptive summary of the time axis

        # The default summary length is the pooled one, so by default this pooling is a no-op that only
        # pins the feature size: an explicit summary makes the feature vector independent of the input length.
        summary = config.T // config.pooling.pooling_factor if config.summary == -1 else config.summary
        self.extract_features = nn.AdaptiveAvgPool1d(int(summary))

        self.n_features = (config.t_filters + config.s_growth * config.spat_layers) * int(summary)

        self.flatten_output = config.flatten_output

        # - - - - - - - - - - - - - - - - - - - - - - - - -
        # Print additional information after the creation of the network
        if config.print_var :
            print("TIDNet Created. Number of parameters:")
            print(f"\tNumber of trainable parameters (Temporal block) = {support_functions.count_trainable_parameters(self.temporal_block)}")
            print(f"\tNumber of trainable parameters (Spatial block)  = {support_functions.count_trainable_parameters(self.spatial_block)}\n")
            self.debug_shape(config.input_size)

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        # - - - - - - - - - - - - - - - - - - - - - - - - -
        # Temporal block : (B, depth, C, T) -> (B, t_filters, C, T_pool)
        x = self.temporal_block(x)

        # - - - - - - - - - - - - - - - - - - - - - - - - -
        # Spatial block : (B, t_filters, C, T_pool) -> (B, t_filters + s_growth * spat_layers, T_pool)
        x = self.spatial_block(x)

        # - - - - - - - - - - - - - - - - - - - - - - - - -
        # Adaptive summary of the time axis : (B, F, T_pool) -> (B, F, summary)
        x = self.extract_features(x)

        # (OPTIONAL) Flat the output
        if self.flatten_output : return x.flatten(1)
        else : return x

    def debug_shape(self, input_size) :
        """
        Method that create a fake input and pass it through the network, showing after each step the shape.
        """

        print("Shape tracking across TIDNet")

        x = torch.randn(input_size)
        print("\tInput shape :\t\t\t", x.shape)

        x = self.temporal_block(x)
        print("\tTemporal block :\t\t", x.shape)

        x = self.spatial_block(x)
        print("\tSpatial block :\t\t\t", x.shape)

        x = self.extract_features(x)
        print("\tAdaptive summary :\t\t", x.shape)

        print("\t(OPTIONAL) Flatten :\t\t", x.flatten(1).shape, "\n")


class TIDNet_Classifier(nn.Module) :
    """
    TIDNet feature extractor plus a feedforward classification head.

    The head flattens the feature sequences and maps them to the classes with a single linear layer, ending with a :class:`torch.nn.LogSoftmax` (when ``config.use_softmax`` is True, so the output are log-probabilities) or a :class:`torch.nn.Identity` (when False, so the output are raw logits).
    The choice must stay consistent with the training loss, see :func:`~jade.training.support_training.cross_check_config`; note that the original TIDNet always ends with a log-softmax, so ``use_softmax = True`` is the faithful setting.

    The auxiliary subject and run heads of the original are **not** reproduced: this classifier returns a plain tensor, like every other ``jade`` model.

    Parameters
    ----------
    config : TIDNet_config or dict
        The network configuration, an instance of :class:`~jade.config.model.TIDNet_cfg.TIDNet_config` (a dict is accepted too and promoted automatically).
        The classifier reads ``input_size`` (to size the feedforward layer with a dummy forward pass), ``n_classes`` and ``use_softmax`` on top of the fields used by :class:`TIDNet`.
    """

    def __init__(self, config) :
        super().__init__()

        # Accept a dict (or an already-built config) and promote it to a validated TIDNet_config.
        config = get_model_config(config)

        self.tidnet = TIDNet(config)

        # The head needs flat (batch, features) input, so the wrapped extractor must flatten regardless of
        # the config flatten_output (which only concerns the standalone TIDNet feature extractor).
        self.tidnet.flatten_output = True

        input_neurons   = self.compute_number_of_neurons(config.input_size)
        self.classifier = nn.Sequential(
            nn.Linear(input_neurons, config.n_classes),
            nn.LogSoftmax(dim = 1) if config.use_softmax else nn.Identity(),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        x = self.tidnet(x)

        x = self.classifier(x)

        return x

    def compute_number_of_neurons(self, input_size) :
        """
        Compute the total number of neurons for the feedforward layer of the head.
        """

        x = torch.rand(input_size)
        x = self.tidnet(x)
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
