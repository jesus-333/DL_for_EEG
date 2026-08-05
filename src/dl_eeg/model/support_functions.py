"""
Support function for model creation

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

import torch
from torch import nn

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Support during creation

def get_activation(activation_name : str) -> nn.Module:
    """
    Receive a string and return the relative activation function in PyTorch.
    Currently supported activation functions are: relu, elu, selu and gelu. The function is not case sensitive.


    Parameters
    ----------
    activation_name : str
        String with the name of the activation function. Supported activation functions are: relu, elu, selu and gelu. The function is not case sensitive.


    Returns
    -------
    activation_function : nn.Module
        Activation function corresponding to the input string.
    """

    if activation_name.lower() == 'relu':
        return nn.ReLU()
    elif activation_name.lower() == 'elu':
        return nn.ELU()
    elif activation_name.lower() == 'selu':
        return nn.SELU()
    elif activation_name.lower() == 'gelu':
        return nn.GELU()
    else:
        error_message = 'The activation must have one of the following string: relu, elu, selu, gelu'
        raise ValueError(error_message)

def get_dropout(prob_dropout: float, use_droput_2d : bool) -> nn.Module :
    """
    Get a dropout module given the probability and the type of dropout (2d or not).
    If use_dropout_2d is True, it will return a nn.Dropout2d, otherwise it will return a nn.Dropout.


    Parameters
    ----------
    prob_dropout : float
        Probability of dropout. Must be between 0 and 1.
    use_droput_2d : bool
        If True, it will return a nn.Dropout2d, otherwise it will return a nn.Dropout.


    Returns
    -------
    dropout_module : nn.Module
        Dropout module corresponding to the input parameters.
    """

    if use_droput_2d :
        return nn.Dropout2d(prob_dropout)
    else :
        return nn.Dropout(prob_dropout)


def count_trainable_parameters(torch_module : nn.Module) -> int:
    """
    Count the number of trainable parameters in a given torch module.

    Parameters
    ----------
    torch_module : nn.Module
        The torch module for which to count the number of trainable parameters.

    Returns
    -------
    n_parameters : int
        The number of trainable parameters in the given torch module.
    """
    n_parameters = sum(p.numel() for p in torch_module.parameters() if p.requires_grad)
    return n_parameters

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Function for aEEGNet

def create_head(config : dict) -> tuple[nn.Module, nn.Module, nn.Module] :
    """
    Create the head (Feedforward layers) for the attention module.

    Parameters
    ----------
    config : dict
        Configuration dictionary. It must contain the following keys:
        - input_length : int
            Length of the input array.
        - qk_output_length : int
            Length of the output of the head for the query and key arrays. It is used only if use_head_for_q or use_head_for_k is True.
        - use_head_for_v : bool
            If True, return a Linear layer for the value array mapping, otherwise return an Identity layer.
        - v_output_length : int
            Length of the output of the head for the value array. It is used only if use_head_for_v is True.

    Returns
    -------
    q_map : nn.Module
        Map for the query array. It is a Linear layer that maps from input_length to qk_output_length.
    k_map : nn.Module
        Map for the key array. It is a Linear layer that maps from input_length to qk_output_length.
    v_map : nn.Module
        Map for the value array. It is a Linear layer that maps from input_length to v_output_length if use_head_for_v is True, otherwise it is an Identity layer.
    """
    
    # Create the head for the query and key arrays
    q_map = nn.Linear(config['input_length'], config['qk_output_length'])
    k_map = nn.Linear(config['input_length'], config['qk_output_length'])
    
    # (OPTIONAL) Create the head for the value array
    if config['use_head_for_v'] :
        v_map = nn.Linear(config['input_length'], config['v_output_length'])
    else :
        v_map = nn.Identity()

    return q_map, k_map, v_map

def compute_attention_values(q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, normalize_qk : bool) -> torch.Tensor :
    """
    Compute the attention values given the query, key and value arrays.

    Parameters
    ----------
    q : torch.Tensor
        Query array of shape B x C x Dq
    k : torch.Tensor
        Key array of shape B x Dk x C
    v : torch.Tensor
        Value array of shape B x C x Dv
    normalize_qk : bool
        If True, the score is divided by sqrt(Dq) before applying the softmax.

    Returns
    -------
    z : torch.Tensor
        Attention values of shape B x C x Dv
    """
    # Transpose k arrays
    k = torch.permute(k, (0, 2, 1))
    
    # Compute the score to use in the linear combination of values
    score = torch.bmm(q, k)
    if normalize_qk : score /= torch.sqrt(torch.tensor(q.shape[2]))
    score = nn.functional.softmax(score, dim = 2)

    z = torch.bmm(score, v)

    return z
