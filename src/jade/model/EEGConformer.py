"""
Implementation of the EEG Conformer model using PyTorch.

The model is the one presented in ``EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization`` (Song et al., IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2023, https://ieeexplore.ieee.org/document/9991178).
It couples a convolutional patch embedding (a shallow EEGNet-like front end that plays the role of the ViT patch splitting) with a transformer encoder and, for the classifier, a flatten + MLP head.

This implementation follows the structure of the authors' reference code, but is rewritten to fit the ``jade`` conventions:
- it is driven by a validated :class:`~jade.config.model.EEGConformer_cfg.EEGConformer_config` dataclass (like :class:`~jade.model.EEGNet.EEGNet`), so all the defaulting/validation lives in the config;
- the pooling stage of the patch embedding supports the custom anti-aliased operators of :mod:`jade.model.pooling_1d` (through :func:`get_conformer_pooling`), so the pooling ablation can run on the Conformer too;
- it depends only on ``torch`` (the original ``einops`` rearrangements are expressed with plain tensor operations).

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
#%% Pooling factory (patch embedding)

def get_conformer_pooling(config) -> nn.Module :
    """
    Build the pooling layer of the EEG Conformer patch embedding.

    This is the bridge between the ``[pooling]`` section of :class:`~jade.config.model.EEGConformer_cfg.EEGConformer_config` and the 1D pooling family in :mod:`jade.model.pooling_1d`.
    In the patch embedding the pooling follows the temporal + spatial convolutions and "acts as slicing to obtain the patches along the time dimension", so its operator (and decimation factor) directly control the number of transformer tokens.

    The ``pooling_type`` selects the behaviour:

    - ``'default'`` reproduces the original EEG Conformer pooling, an **overlapping** ``nn.AvgPool2d`` with kernel ``(1, pooling_kernel)`` and stride ``(1, pooling_stride)`` (pooling along the time axis only).
    - one of the 1D pooling names (``'avg'``, ``'max'``, ``'blur'``, ``'sinc'``, ``'dwt'`` and their aliases) returns the matching ``{name}_multichannel`` wrapper of :mod:`jade.model.pooling_1d`, applied channel-wise with decimation factor ``pooling_stride`` (through :func:`~jade.model.support_functions.get_multichannel_pooling`).
    - ``'identity'`` / ``'none'`` returns ``nn.Identity`` (the pooling stage is disabled).

    Parameters
    ----------
    config : conformer_pooling_config or dict
        The pooling section, either a :class:`~jade.config.model.EEGConformer_cfg.conformer_pooling_config` dataclass or a plain dict (promoted to one).

    Returns
    -------
    pooling_module : nn.Module
        The pooling layer to drop into the patch embedding.
    """

    # Accept a dict (or an already-built config) and promote it to a validated conformer_pooling_config.
    # Lazy import to keep the dependency direction one-way (model -> config) and avoid any import cycle.
    from jade.config.model import conformer_pooling_config
    if isinstance(config, dict) :
        config = conformer_pooling_config.from_dict(config)

    pooling_type = str(config.pooling_type).lower()

    # Disabled stage: explicit identity / none.
    if pooling_type in ("identity", "none") :
        return nn.Identity()

    # Original EEG Conformer behaviour: overlapping 2D average pooling along the time axis only.
    if pooling_type == "default" :
        return nn.AvgPool2d(kernel_size = (1, config.pooling_kernel), stride = (1, config.pooling_stride))

    # 1D anti-aliased operator applied channel-wise (the config exposes it as a ready pooling_config).
    return support_functions.get_multichannel_pooling(config.pooling_layer_config)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Building blocks

class patch_embedding(nn.Module) :
    """
    Convolutional patch embedding of the EEG Conformer.

    It is a shallow EEGNet-like front end that turns the raw EEG tensor into a sequence of tokens for the transformer encoder.
    A temporal convolution and a spatial convolution (which collapses the EEG-channel axis) are followed by batch normalization, an activation, a pooling stage that slices the time axis into patches, and dropout;
    a final ``(1, 1)`` convolution projects the feature maps to the token embedding size.

    Parameters
    ----------
    config : EEGConformer_config
        The (already validated) model configuration.
    """

    def __init__(self, config) :
        super().__init__()

        activation    = support_functions.get_activation(config.activation)
        pooling_layer = get_conformer_pooling(config.pooling)

        # Temporal + spatial convolutions, then normalization / activation / pooling / dropout.
        # The spatial convolution kernel (C, 1) collapses the EEG-channel axis to 1 (one spatial filter over all channels).
        self.shallow_net = nn.Sequential(
            nn.Conv2d(config.depth_first_layer, config.filter_temporal, config.c_kernel_temporal, (1, 1)),
            nn.Conv2d(config.filter_temporal, config.filter_temporal, (config.C, 1), (1, 1)),
            nn.BatchNorm2d(config.filter_temporal),
            activation,
            pooling_layer,
            nn.Dropout(config.prob_dropout),
        )

        # The 1x1 convolution projects the feature maps to the token embedding size (a linear map that,
        # as noted in the original, can enhance the fitting ability slightly compared to a plain transpose).
        self.projection = nn.Conv2d(config.filter_temporal, config.emb_size, (1, 1), stride = (1, 1))

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        # Convolutional front end: (B, depth, C, T) -> (B, filter_temporal, 1, T_pool).
        x = self.shallow_net(x)

        # Project to the embedding size: (B, filter_temporal, 1, T_pool) -> (B, emb_size, 1, T_pool).
        x = self.projection(x)

        # Flatten the two spatial dims (height is 1) and move the embedding to the last axis:
        # (B, emb_size, 1, T_pool) -> (B, emb_size, T_pool) -> (B, T_pool, emb_size).
        # This is the plain-tensor equivalent of einops' Rearrange('b e h w -> b (h w) e').
        x = x.flatten(2).transpose(1, 2)

        return x


class multi_head_attention(nn.Module) :
    """
    Multi-head self-attention block.

    Parameters
    ----------
    emb_size : int
        Token embedding size.
    num_heads : int
        Number of attention heads (``emb_size`` must be divisible by it).
    dropout : float
        Dropout probability applied to the attention weights.
    """

    def __init__(self, emb_size : int, num_heads : int, dropout : float) :
        super().__init__()

        self.emb_size  = emb_size
        self.num_heads = num_heads

        # Separate linear maps for queries, keys and values (each emb_size -> emb_size).
        self.queries = nn.Linear(emb_size, emb_size)
        self.keys    = nn.Linear(emb_size, emb_size)
        self.values  = nn.Linear(emb_size, emb_size)

        self.attention_dropout = nn.Dropout(dropout)
        self.projection        = nn.Linear(emb_size, emb_size)

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        batch_size, n_tokens, _ = x.shape
        head_dim = self.emb_size // self.num_heads

        # Project and split the embedding across the heads: (B, N, emb_size) -> (B, num_heads, N, head_dim).
        queries = self.queries(x).view(batch_size, n_tokens, self.num_heads, head_dim).permute(0, 2, 1, 3)
        keys    = self.keys(x).view(batch_size, n_tokens, self.num_heads, head_dim).permute(0, 2, 1, 3)
        values  = self.values(x).view(batch_size, n_tokens, self.num_heads, head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention.
        # Note that (faithfully to the reference implementation) the scaling uses sqrt(emb_size), not the more
        # common sqrt(head_dim); it is kept as-is to reproduce the original model exactly.
        energy    = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        scaling   = self.emb_size ** 0.5
        attention = torch.softmax(energy / scaling, dim = -1)
        attention = self.attention_dropout(attention)

        # Weighted sum of the values, then merge the heads back: (B, num_heads, N, head_dim) -> (B, N, emb_size).
        out = torch.einsum("bhql, bhld -> bhqd", attention, values)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, n_tokens, self.emb_size)
        out = self.projection(out)

        return out


class residual_add(nn.Module) :
    """
    Residual (skip) connection around a sub-module.

    Parameters
    ----------
    fn : nn.Module
        The sub-module wrapped by the residual connection.
    """

    def __init__(self, fn : nn.Module) :
        super().__init__()

        self.fn = fn

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        # Out-of-place addition (x + f(x)) keeps the autograd graph clean.
        return x + self.fn(x)


class feed_forward_block(nn.Sequential) :
    """
    Position-wise feedforward block of a transformer encoder block.

    Parameters
    ----------
    emb_size : int
        Token embedding size.
    expansion : int
        Expansion factor of the hidden layer (its size is ``expansion * emb_size``).
    dropout : float
        Dropout probability applied between the two linear layers.
    """

    def __init__(self, emb_size : int, expansion : int, dropout : float) :
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size),
        )


class transformer_encoder_block(nn.Sequential) :
    """
    Single transformer encoder block (pre-norm multi-head attention + feedforward, each with a residual connection).

    Parameters
    ----------
    emb_size : int
        Token embedding size.
    num_heads : int
        Number of attention heads.
    transformer_dropout : float
        Dropout probability of the attention weights and of the two post-sublayer dropouts.
    forward_expansion : int
        Expansion factor of the feedforward block.
    forward_dropout : float
        Dropout probability inside the feedforward block.
    """

    def __init__(self, emb_size : int, num_heads : int, transformer_dropout : float, forward_expansion : int, forward_dropout : float) :
        super().__init__(
            # Attention sub-layer (LayerNorm -> multi-head attention -> dropout), wrapped in a residual connection.
            residual_add(nn.Sequential(
                nn.LayerNorm(emb_size),
                multi_head_attention(emb_size, num_heads, transformer_dropout),
                nn.Dropout(transformer_dropout),
            )),
            # Feedforward sub-layer (LayerNorm -> feedforward -> dropout), wrapped in a residual connection.
            residual_add(nn.Sequential(
                nn.LayerNorm(emb_size),
                feed_forward_block(emb_size, forward_expansion, forward_dropout),
                nn.Dropout(transformer_dropout),
            )),
        )


class transformer_encoder(nn.Sequential) :
    """
    Stack of ``config.depth`` identical :class:`transformer_encoder_block` modules.

    Parameters
    ----------
    config : EEGConformer_config
        The (already validated) model configuration.
    """

    def __init__(self, config) :
        super().__init__(*[
            transformer_encoder_block(
                config.emb_size,
                config.num_heads,
                config.transformer_dropout,
                config.forward_expansion,
                config.forward_dropout,
            )
            for _ in range(config.depth)
        ])


class classification_head(nn.Module) :
    """
    Flatten + MLP classification head of the EEG Conformer.

    The whole token sequence is flattened (the ``EEGConformer`` extractor already flattens it) and fed to a small MLP.
    The head ends with a :class:`torch.nn.LogSoftmax` (when ``config.use_softmax`` is True, so the output are log-probabilities) or a :class:`torch.nn.Identity` (when False, so the output are raw logits), matching the loss coupling of :func:`~jade.training.support_training.cross_check_config`.

    Parameters
    ----------
    config : EEGConformer_config
        The (already validated) model configuration.
    input_neurons : int
        Number of input features (the flattened token sequence, ``n_tokens * emb_size``), sized by :class:`EEGConformer_Classifier`.
    """

    def __init__(self, config, input_neurons : int) :
        super().__init__()

        # A fresh activation instance per position (a single nn.Module cannot be shared across a Sequential).
        activation_1 = support_functions.get_activation(config.activation)
        activation_2 = support_functions.get_activation(config.activation)

        self.head = nn.Sequential(
            nn.Linear(input_neurons, config.hidden_size_1),
            activation_1,
            nn.Dropout(config.head_dropout_1),
            nn.Linear(config.hidden_size_1, config.hidden_size_2),
            activation_2,
            nn.Dropout(config.head_dropout_2),
            nn.Linear(config.hidden_size_2, config.n_classes),
            nn.LogSoftmax(dim = 1) if config.use_softmax else nn.Identity(),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        return self.head(x)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Network declaration

class EEGConformer(nn.Module) :
    """
    EEG Conformer feature extractor (convolutional patch embedding + transformer encoder).

    Analogous to :class:`~jade.model.EEGNet.EEGNet`, this is the feature-extractor part of the model: the final flatten + MLP classification head of the paper is **not** part of it (see :class:`EEGConformer_Classifier`).
    It turns a raw EEG tensor into a sequence of transformer tokens, which can optionally be flattened into a feature vector.

    Parameters
    ----------
    config : EEGConformer_config or dict
        The network configuration, an instance of :class:`~jade.config.model.EEGConformer_cfg.EEGConformer_config` (a dict is accepted too and promoted automatically through :func:`~jade.config.model.get_model_config`).
        Every field is documented on that config class; the fields validation (and defaulting) happens there, so this constructor just reads the already-validated attributes.
    """

    def __init__(self, config) :
        super().__init__()

        # Accept a dict (or an already-built config) and promote it to a validated EEGConformer_config.
        config = get_model_config(config)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Convolutional patch embedding + transformer encoder

        self.patch_embedding    = patch_embedding(config)
        self.transformer_encoder = transformer_encoder(config)

        self.flatten_output = config.flatten_output

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Print additional information after the creation of the network
        if config.print_var :
            print("EEGConformer Created. Number of parameters:")
            print(f"\tNumber of trainable parameters (Patch embedding)     = {support_functions.count_trainable_parameters(self.patch_embedding)}")
            print(f"\tNumber of trainable parameters (Transformer encoder) = {support_functions.count_trainable_parameters(self.transformer_encoder)}\n")
            self.debug_shape(config.input_size)

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Patch embedding (conv front end) -> token sequence (B, n_tokens, emb_size)
        x = self.patch_embedding(x)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Transformer encoder (keeps the (B, n_tokens, emb_size) shape)
        x = self.transformer_encoder(x)

        # (OPTIONAL) Flat the output
        if self.flatten_output : return x.flatten(1)
        else : return x

    def debug_shape(self, input_size) :
        """
        Method that create a fake input and pass it through the network, showing after each step the shape.
        """

        print("Shape tracking across EEGConformer")

        x = torch.randn(input_size)
        print("\tInput shape :\t\t\t", x.shape)

        x = self.patch_embedding.shallow_net(x)
        print("\tPatch embedding (conv + pool) :\t", x.shape)

        x = self.patch_embedding.projection(x)
        print("\tProjection (1x1 conv) :\t\t", x.shape)

        x = x.flatten(2).transpose(1, 2)
        print("\tToken sequence (b, n, e) :\t", x.shape)

        x = self.transformer_encoder(x)
        print("\tTransformer encoder :\t\t", x.shape)

        print("\t(OPTIONAL) Flatten :\t\t", x.flatten(1).shape, "\n")


class EEGConformer_Classifier(nn.Module) :
    """
    EEG Conformer feature extractor plus the flatten + MLP classification head of the paper.

    The head flattens the whole token sequence and passes it through a small MLP ending with a :class:`torch.nn.LogSoftmax` (when ``config.use_softmax`` is True, so the output are log-probabilities) or a :class:`torch.nn.Identity` (when False, so the output are raw logits).
    The choice must stay consistent with the training loss, see :func:`~jade.training.support_training.cross_check_config`.

    Parameters
    ----------
    config : EEGConformer_config or dict
        The network configuration, an instance of :class:`~jade.config.model.EEGConformer_cfg.EEGConformer_config` (a dict is accepted too and promoted automatically).
        The classifier reads ``input_size`` (to size the first layer of the MLP head with a dummy forward pass), the head sizes, ``n_classes`` and ``use_softmax`` on top of the fields used by :class:`EEGConformer`.
    """

    def __init__(self, config) :
        super().__init__()

        # Accept a dict (or an already-built config) and promote it to a validated EEGConformer_config.
        config = get_model_config(config)

        self.eegconformer = EEGConformer(config)

        # The MLP head needs flat (batch, features) input, so the wrapped extractor must flatten
        # regardless of the config flatten_output (which only concerns the standalone EEGConformer feature extractor).
        self.eegconformer.flatten_output = True

        input_neurons   = self.compute_number_of_neurons(config.input_size)
        self.classifier = classification_head(config, input_neurons)

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        x = self.eegconformer(x)

        x = self.classifier(x)

        return x

    def compute_number_of_neurons(self, input_size) :
        """
        Compute the total number of neurons for the first layer of the MLP head.
        """

        x = torch.rand(input_size)
        x = self.eegconformer(x)
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
