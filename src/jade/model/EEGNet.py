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
    config : dict
        Dictionary with the configuration of the network. The required keys are:
        - use_bias : bool
            If True, the convolutional layers will use bias, otherwise they will not.
        - D : int
            Number of spatial filters to learn within each temporal convolution. The total number of spatial filters will be D * filter_1.
        - activation : str
            String with the name of the activation function to use. Supported activation functions are: relu, elu, selu and gelu.
        - prob_dropout : float
            Probability of dropout. Must be between 0 and 1.
        - use_dropout_2d : bool
            If True, it will use nn.Dropout2d, otherwise it will use nn.Dropout.
        - filter_1 : int
            Number of filters for the first convolutional layer (temporal filters).
        - c_kernel_1 : tuple
            Kernel size for the first convolutional layer (temporal filters).
        - c_kernel_2 : tuple
            Kernel size for the second convolutional layer (spatial filters).
        - p_kernel_1 : tuple or None
            Kernel size for the first pooling layer. If None, it will not use pooling.
        - filter_2 : int
            Number of filters for the second convolutional layer (separable convolution).
        - c_kernel_3 : tuple
            Kernel size for the first convolutional layer of the separable convolution.
        - p_kernel_2 : tuple or None
            Kernel size for the second pooling layer. If None, it will not use pooling.
        - flatten_output : bool (optional, default = False)
            If True, the output of the network will be flattened, otherwise it will mantain its
    """
    
    def __init__(self, config : dict):
        super().__init__()
        
        use_bias = config['use_bias']
        D = config['D']
        activation = support_functions.get_activation(config['activation'])
        dropout = support_functions.get_dropout(config['prob_dropout'], config['use_dropout_2d'])
        
        # NOT USED
        self.C = config['C']
        self.T = config['T']
        
        # Used only if the EEGNet is used with time-frequency data
        # In this case the depth of the input is not 1
        depth_first_layer = config['depth_first_layer'] if 'depth_first_layer' in config else 1

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Convolutional section
        
        # Block 1 - Temporal filters
        self.temporal_filter = nn.Sequential(
            nn.Conv2d(depth_first_layer, config['filter_1'], kernel_size = config['c_kernel_1'],
                      padding = 'same', bias = use_bias),
            nn.BatchNorm2d(config['filter_1']),
        )
        
        # Block 1 - Spatial filters
        self.spatial_filter = nn.Sequential(
            nn.Conv2d(config['filter_1'], config['filter_1'] * D, kernel_size = config['c_kernel_2'], groups = config['filter_1'], bias = use_bias),
            nn.BatchNorm2d(config['filter_1'] * D),
            activation,
            nn.AvgPool2d(config['p_kernel_1']) if config['p_kernel_1'] is not None else nn.Identity(),
            dropout
        )

        # Block 2 - Features Extraction
        self.separable_convolution = nn.Sequential(
            nn.Conv2d(config['filter_1'] * D, config['filter_1'] * D, kernel_size = config['c_kernel_3'], groups = config['filter_1'] * D, padding = 'same', bias = use_bias),
            nn.Conv2d(config['filter_1'] * D, config['filter_2'], kernel_size = (1, 1), groups = 1, bias = use_bias),
            nn.BatchNorm2d(config['filter_2']),
            activation,
            nn.AvgPool2d(config['p_kernel_2']) if config['p_kernel_2'] is not None else nn.Identity(),
            dropout
        )
        
        if 'flatten_output' not in config: config['flatten_output'] = False
        self.flatten_output = config['flatten_output']

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Print additional information after the creation of the network
        if 'print_var' not in config: config['print_var'] = False
        if config['print_var']:
            print("EEGNet Created. Number of parameters:")
            print("\tNumber of trainable parameters (Block 1 - Temporal) = {}".format(support_functions.count_trainable_parameters(self.temporal_filter)))
            print("\tNumber of trainable parameters (Block 1 - Spatial)  = {}".format(support_functions.count_trainable_parameters(self.spatial_filter)))
            print("\tNumber of trainable parameters (Block 2)            = {}\n".format(support_functions.count_trainable_parameters(self.separable_convolution)))
            if 'input_size' in config: self.debug_shape(config['input_size'])

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

    def __init__(self, config : dict):
        super().__init__()
        """
        EEGNet + classifier
        """
        
        self.eegnet = EEGNet(config)
        
        input_neurons = self.compute_number_of_neurons(config['input_size'])
        self.classifier = nn.Sequential(
            nn.Linear(input_neurons, config['n_classes']),
            nn.LogSoftmax(dim = 1) if config['use_softmax'] else nn.Identity()
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

