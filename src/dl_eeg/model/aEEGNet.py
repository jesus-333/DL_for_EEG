"""
Variant of EEGNet enhanced with attention mechanism.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

import torch

from dl_eeg.model import EEGNet
from dl_eeg.model import support_functions

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class aEEGNet(torch.nn.Module) :
    """
    Implementation of aEEGNet in PyTorch.

    
    Attributes
    ----------

    """

    def __init__(self, config : dict) :
        super().__init__()

        a = 2


    def forward(self, x) :
        return x


class channels_attention_head(torch.nn.module) :
    """
    Variant of the classic attention head created to work with multichannel EEG data.
    It received as input a multichannel EEG and it output a weighted sum of the input channels. The weights are computed through the attention mechanism.
    """

    def __init__(self, config : dict) :
        super().__init__()
        
        # Create the linear layers for the query, key and value
        self.q_map, self.k_map, self.v_map = support_functions.create_head(config)

        self.normalize_qk = config['normalize_qk'] if 'normalize_qk' in config else True

    def forward(self, x) :
        """
        Forward pass of the attention head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape B x C x T

        Returns
        -------
        """

        # Compute the attention weights
        attention_weights = self.compute_attention_weights(x)

        # Compute the value array
        v = self.v_map(x)

        # Compute the final output
        output = torch.bmm(attention_weights, v)
        
        return output

    def compute_attention_weights(self, x : torch.Tensor) -> torch.Tensor :
        """
        Compute the attention weights given the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape B x C x T

        Returns
        -------
        attention_weights : torch.Tensor
            Attention weights of shape B x C
        """

        q = self.q_map(x)
        k = self.k_map(x)

        # Compute the attention weights
        attention_weights = torch.bmm(q, k.transpose(1, 2))

        # Normalize the attention weights
        if self.normalize_qk :
            attention_weights = attention_weights / (q.shape[-1] ** 0.5)
        
        # Apply the softmax function to the attention weights
        attention_weights = torch.softmax(attention_weights, dim = -1)

        return attention_weights
