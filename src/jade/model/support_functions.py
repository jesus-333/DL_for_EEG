"""
Support function for model creation

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

from __future__ import annotations

from typing import Union

import torch
from torch import nn

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Support during creation

def get_activation(activation_name : str) -> nn.Module:
    """
    Receive a string and return the relative activation function in PyTorch.
    Currently supported activation functions are: relu, leaky_relu, elu, selu and gelu. The function is not case sensitive.


    Parameters
    ----------
    activation_name : str
        String with the name of the activation function. Supported activation functions are: relu, leaky_relu, elu, selu and gelu. The function is not case sensitive.


    Returns
    -------
    activation_function : nn.Module
        Activation function corresponding to the input string.
    """

    if activation_name.lower() == 'relu':
        return nn.ReLU()
    elif activation_name.lower() == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation_name.lower() == 'elu':
        return nn.ELU()
    elif activation_name.lower() == 'selu':
        return nn.SELU()
    elif activation_name.lower() == 'gelu':
        return nn.GELU()
    else:
        error_message = 'The activation must have one of the following string: relu, leaky_relu, elu, selu, gelu'
        raise ValueError(error_message)

class square(nn.Module) :
    """
    Element-wise squaring, used as a nonlinearity by the Schirrmeister 2017 convolutional networks.

    In :class:`~jade.model.ShallowFBCSPNet.ShallowFBCSPNet` it is applied before the pooling stage: together with the :class:`safe_log` that follows the pooling it turns the mean of the squared samples into a log-variance feature, the quantity the FBCSP pipeline the model is inspired by classifies.
    """

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        return x * x


class safe_log(nn.Module) :
    """
    Element-wise logarithm clamped away from zero, used as a nonlinearity by the Schirrmeister 2017 convolutional networks.

    In :class:`~jade.model.ShallowFBCSPNet.ShallowFBCSPNet` it is applied after the pooling stage. Its input is an average of squared values, so it is non negative and the logarithm would diverge on an exact zero: the clamping keeps the gradient finite.

    Parameters
    ----------
    eps : float, default 1e-6
        Lower bound the input is clamped to before the logarithm.
    """

    def __init__(self, eps : float = 1e-6) :
        super().__init__()

        self.eps = eps

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        return torch.log(torch.clamp(x, min = self.eps))


def get_nonlinearity(nonlin_name : str) -> nn.Module :
    """
    Receive a string and return the relative nonlinearity in PyTorch.

    Superset of :func:`get_activation`: on top of the standard activations it also knows the two nonlinearities of the Schirrmeister 2017 convolutional networks (``square`` and ``safe_log``, see :class:`square` and :class:`safe_log`) and the no-op ``identity``.
    Any other name is delegated to :func:`get_activation`, which raizes ``ValueError`` if it does not know it either. The function is not case sensitive.


    Parameters
    ----------
    nonlin_name : str
        String with the name of the nonlinearity. On top of the activations of :func:`get_activation` it accepts: square, safe_log and identity. The function is not case sensitive.


    Returns
    -------
    nonlin_module : nn.Module
        Nonlinearity corresponding to the input string.
    """

    nonlin_name = nonlin_name.lower()

    if nonlin_name == 'square' :
        return square()
    elif nonlin_name == 'safe_log' :
        return safe_log()
    elif nonlin_name == 'identity' :
        return nn.Identity()
    else :
        # Every other name is a standard activation.
        return get_activation(nonlin_name)

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

def get_multichannel_pooling(config : Union["pooling_config", dict]) -> nn.Module :
    """
    Build a channel-wise pooling layer for multichannel EEG tensors of shape ``[B x D x C x T]``.

    Analogue of :func:`get_activation` and :func:`get_dropout`: it maps a small pooling config to a ready-to-use ``nn.Module``.
    It is the bridge between the EEGNet ``pooling`` config section and the 1D pooling family in :mod:`jade.model.pooling_1d`.

    The ``pooling_type`` selects the behaviour:

    - ``'default'`` reproduces the classic EEGNet pooling, ``nn.AvgPool2d((1, decimation_factor))`` (pooling along the time axis only).
    - one of the 1D pooling names (``'avg'``, ``'max'``, ``'blur'``, ``'sinc'``, ``'dwt'`` and their aliases) returns the matching ``{name}_multichannel`` wrapper, applied channel-wise.
    - ``'identity'`` / ``'none'``, or a non-positive ``decimation_factor``, returns ``nn.Identity`` (the pooling stage is disabled).

    Any other value raises ``ValueError``.

    Parameters
    ----------
    config : pooling_config or dict
        Per-stage pooling config, either a :class:`~jade.config.model.pooling_1d_cfg.pooling_config` dataclass or a plain dict.
        A dataclass is used **as an object** (its already-validated attributes are read directly, and it is handed straight to the channel-wise pooling layer, which accepts it as-is): there is no dict round-trip.
        The dict form exists for the EEGNet-only pooling types that have no ``pooling_config`` dataclass (``'default'`` / ``'identity'`` / ``'none'``, and a disabled stage), which is how :attr:`~jade.config.model.EEGNet_cfg.eegnet_pooling_config.pooling_layer_1_config` passes them; a raw dict for a 1D operator is accepted too.
        Whichever form, it carries ``pooling_type`` and ``decimation_factor`` (dict alias ``pooling_factor``), plus any parameters the chosen 1D pooling layer accepts (see :mod:`jade.model.pooling_1d`).

    Returns
    -------
    pooling_module : nn.Module
        The pooling layer to drop into the EEGNet ``nn.Sequential`` blocks.
    """

    # Import here (not at module load) to keep the dependency direction one-way and lazy.
    from jade.model.pooling_1d import MULTICHANNEL_POOLING_REGISTRY
    from jade.config.model.pooling_1d_cfg import pooling_config

    # Read the discriminator and the decimation factor straight off the config. A pooling_config
    # dataclass exposes them as (already-validated) attributes; the EEGNet-only 'default' /
    # 'identity' / 'none' and disabled stages have no dataclass, so they arrive as a plain dict.
    if isinstance(config, pooling_config) :
        pooling_type = config.pooling_type
        factor = config.decimation_factor
    elif isinstance(config, dict) :
        pooling_type = config.get('pooling_type', 'default')
        factor = config.get('decimation_factor', config.get('pooling_factor', None))
    else :
        raise TypeError(f"get_multichannel_pooling expects a pooling_config or a dict, got {type(config)}")

    pooling_type = str(pooling_type).lower()

    # Disabled stage: explicit identity, or a non-positive / no-op decimation factor.
    # A negative factor is the documented "skip this pooling layer" signal (the EEGNet config warns about it).
    if pooling_type in ('identity', 'none') or (factor is not None and factor < 1) :
        return nn.Identity()

    # Classic EEGNet behaviour: 2D average pooling along the time axis only.
    if pooling_type == 'default' :
        return nn.AvgPool2d((1, int(factor)))

    if pooling_type not in MULTICHANNEL_POOLING_REGISTRY :
        valid = sorted(set(MULTICHANNEL_POOLING_REGISTRY)) + ['default', 'identity']
        raise ValueError(f"pooling_type must be one of {valid}, got {pooling_type!r}")

    # 1D pooling operator applied channel-wise. The channel-wise wrappers accept a pooling_config
    # object directly (see jade.model.pooling_1d), so a dataclass is passed through untouched.
    if isinstance(config, pooling_config) :
        return MULTICHANNEL_POOLING_REGISTRY[pooling_type](config)

    # Raw dict path: never mutate the caller's dict, and honor the 'gaussian'/'binomial' aliases,
    # which select blur pooling *and* its kernel family (mirrors get_pooling_layer).
    cfg = dict(config)
    if pooling_type == 'gaussian' :
        cfg.setdefault('kernel', 'gaussian')
    elif pooling_type == 'binomial' :
        cfg.setdefault('kernel', 'binomial')

    return MULTICHANNEL_POOLING_REGISTRY[pooling_type](cfg)


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
# Generic modules

class lora_linear(nn.Module) :
    """
    Low-rank replacement of a :class:`torch.nn.Linear` layer, built on the idea behind LoRA (Hu et al., 2021).

    The ``out_features x in_features`` weight matrix is factorized as the product of two matrices of rank ``rank``, i.e. the input is first projected down to ``rank`` features and then back up to ``out_features``.
    With ``rank << min(in_features, out_features)`` the number of weights drops from ``in_features * out_features`` to ``rank * (in_features + out_features)``.
    Note that, contrary to the original LoRA, there is no frozen full-rank layer: this module *is* the layer.
    The output is scaled by ``alpha / rank``, and there is no bias.

    It is used by the vector latent mappings of :class:`~jade.model.hvEEGNet.hvEEGNet`, where the input (a flattened feature map) can have hundreds of thousands of elements.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    rank : int
        Rank of the factorization.
    alpha : float, optional
        Scaling numerator. A non-positive value (the default, -1) means ``alpha = rank``, i.e. no scaling.
    use_spectral_norm : bool, optional
        If True apply :func:`torch.nn.utils.parametrizations.spectral_norm` to the two factors, by default False.
    """

    def __init__(self, in_features : int, out_features : int, rank : int, alpha : float = -1, use_spectral_norm : bool = False) :
        super().__init__()

        down_projection = nn.Linear(in_features, rank, bias = False)
        up_projection   = nn.Linear(rank, out_features, bias = False)

        if use_spectral_norm :
            down_projection = nn.utils.parametrizations.spectral_norm(down_projection)
            up_projection   = nn.utils.parametrizations.spectral_norm(up_projection)

        # The names of the original implementation (lora_a = down projection, lora_b = up projection).
        self.lora_a = down_projection
        self.lora_b = up_projection

        self.rank  = rank
        self.alpha = alpha if alpha > 0 else rank

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        return (self.alpha / self.rank) * self.lora_b(self.lora_a(x))

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
