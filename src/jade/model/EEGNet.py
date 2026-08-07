"""
Implementation of EEGNet model using PyTorch

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
#%% Network declaration

class EEGNet(nn.Module):
    """
    Implementation of EEGNet in PyTorch.
    
    Note that compared to the original EEGNet the last layer (feedforward + classification) it is not present.
    The network offer the possibility to flatten the output (and obtain a vector of features of shape batch x n. features) or to mantain it as a tensor

    The name of the variable respect the nomeclature of the original paper where it is possible.

    Parameters
    ----------
    config : EEGNet_config or dict
        The network configuration, an instance of :class:`~jade.config.model.EEGNet_cfg.EEGNet_config` (a dict is accepted too and promoted automatically through :func:`~jade.config.model.get_model_config`).
        Every field is documented on that config class; the fields validation (and defaulting) happens there, so this constructor just reads the already-validated attributes.
    """

    def __init__(self, config):
        super().__init__()

        # Accept a dict (or an already-built config) and promote it to a validated EEGNet_config.
        config = get_model_config(config)

        use_bias = config.use_bias
        D = config.D
        activation = support_functions.get_activation(config.activation)
        dropout = support_functions.get_dropout(config.prob_dropout, config.use_dropout_2d)

        # Pooling layers for the two pooling stages (see the [pooling] section of EEGNet_config).
        # 'default' reproduces the classic nn.AvgPool2d; the 1D pooling operators are applied channel-wise.
        pooling_layer_1 = support_functions.get_multichannel_pooling(config.pooling_layer_1_config)
        pooling_layer_2 = support_functions.get_multichannel_pooling(config.pooling_layer_2_config)

        # NOT USED
        self.C = config.C
        self.T = config.T

        # Used only if the EEGNet is used with time-frequency data
        # In this case the depth of the input is not 1
        depth_first_layer = config.depth_first_layer

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Convolutional section

        # Block 1 - Temporal filters
        self.temporal_filter = nn.Sequential(
            nn.Conv2d(depth_first_layer, config.filter_1, kernel_size = config.c_kernel_1,
                      padding = 'same', bias = use_bias),
            nn.BatchNorm2d(config.filter_1),
        )

        # Block 1 - Spatial filters
        self.spatial_filter = nn.Sequential(
            nn.Conv2d(config.filter_1, config.filter_1 * D, kernel_size = config.c_kernel_2, groups = config.filter_1, bias = use_bias),
            nn.BatchNorm2d(config.filter_1 * D),
            activation,
            pooling_layer_1,
            dropout
        )

        # Block 2 - Features Extraction
        self.separable_convolution = nn.Sequential(
            nn.Conv2d(config.filter_1 * D, config.filter_1 * D, kernel_size = config.c_kernel_3, groups = config.filter_1 * D, padding = 'same', bias = use_bias),
            nn.Conv2d(config.filter_1 * D, config.filter_2, kernel_size = (1, 1), groups = 1, bias = use_bias),
            nn.BatchNorm2d(config.filter_2),
            activation,
            pooling_layer_2,
            dropout
        )

        self.flatten_output = config.flatten_output

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Print additional information after the creation of the network
        if config.print_var:
            print("EEGNet Created. Number of parameters:")
            print(f"\tNumber of trainable parameters (Block 1 - Temporal) = {support_functions.count_trainable_parameters(self.temporal_filter)}")
            print(f"\tNumber of trainable parameters (Block 1 - Spatial)  = {support_functions.count_trainable_parameters(self.spatial_filter)}")
            print(f"\tNumber of trainable parameters (Block 2)            = {support_functions.count_trainable_parameters(self.separable_convolution)}\n")
            self.debug_shape(config.input_size)

    def forward(self, x : torch.tensor) -> torch.tensor :
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Block 1 (temporal + spatial filters)
        x = self.temporal_filter(x)
        x = self.spatial_filter(x)

        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Block 2 (separable convolution)
        x = self.separable_convolution(x)

        # (OPTIONAL) Flat the output
        if self.flatten_output: return x.flatten(1)
        else: return x

    def debug_shape(self, input_size):
        """
        Method that create a fake input and pass it through the network, showing after each step the shape
        """

        print("Shape tracking across EEGNet")

        x = torch.randn(input_size)
        print("\tInput shape :\t\t\t", x.shape)

        x = self.temporal_filter(x)
        print("\tTemporal filter (conv):\t\t", x.shape)

        x = self.spatial_filter[0](x)
        print("\tSpatial fitler (conv) :\t\t", x.shape)

        x = self.spatial_filter[3](x)
        print("\tSpatial filter (pool) :\t\t", x.shape)

        x = self.separable_convolution[0](x)
        print("\tSeparable convolution (conv 1): ", x.shape)

        x = self.separable_convolution[1](x)
        print("\tSeparable convolution (conv 2): ", x.shape)

        x = self.separable_convolution[4](x)
        print("\tSeparable convolution (pool) :  ", x.shape)

        print("\t(OPTIONAL) Flatten :\t\t", x.flatten(1).shape, "\n")


class EEGNet_Classifier(nn.Module):
    """
    EEGNet feature extractor plus a feedforward classification head.

    The head is a :class:`torch.nn.Linear` followed by a :class:`torch.nn.LogSoftmax` (when ``config.use_softmax`` is True, so the output are log-probabilities) or a :class:`torch.nn.Identity` (when False, so the output are raw logits).
    The choice must stay consistent with the training loss, see :func:`~jade.training.support_training.cross_check_config`.

    Parameters
    ----------
    config : EEGNet_config or dict
        The network configuration, an instance of :class:`~jade.config.model.EEGNet_cfg.EEGNet_config` (a dict is accepted too and promoted automatically).
        The classifier reads ``input_size`` (to size the feedforward layer with a dummy forward pass), ``n_classes`` and ``use_softmax`` on top of the fields used by :class:`~jade.model.EEGNet.EEGNet`.
    """

    def __init__(self, config):
        super().__init__()

        # Accept a dict (or an already-built config) and promote it to a validated EEGNet_config.
        config = get_model_config(config)

        self.eegnet = EEGNet(config)

        # The feedforward head needs flat (batch, features) input, so the wrapped extractor must flatten
        # regardless of the config flatten_output (which only concerns the standalone EEGNet feature extractor).
        self.eegnet.flatten_output = True

        input_neurons = self.compute_number_of_neurons(config.input_size)
        self.classifier = nn.Sequential(
            nn.Linear(input_neurons, config.n_classes),
            nn.LogSoftmax(dim = 1) if config.use_softmax else nn.Identity()
        )

                
    def forward(self, x):
        x = self.eegnet(x)

        x = self.classifier(x)

        return x

    def compute_number_of_neurons(self, input_size):
        """
        Compute the total number of neurons for the feedforward layer
        """

        x = torch.rand(input_size)
        x = self.eegnet(x)
        input_neurons = len(x.flatten())

        return input_neurons
    
    def classify(self, x, return_as_index = True):
        """
        Directly classify an input by returning the label (return_as_index = True) or the probability distribution on the labels (return_as_index = False)
        """

        label = self.forward(x)

        if return_as_index:
            # Note: do NOT squeeze here. A batch of size 1 has shape (1, n_classes) and torch.squeeze would
            # collapse the batch dimension to (n_classes,), making the following argmax(dim = 1) fail.
            # Keeping the (batch, n_classes) shape works for any batch size (exp is monotonic, so it does not
            # change the argmax result for either the LogSoftmax or the Identity/logits case).
            predict_prob = torch.exp(label).detach()
            label = torch.argmax(predict_prob, dim = 1)

        return label

