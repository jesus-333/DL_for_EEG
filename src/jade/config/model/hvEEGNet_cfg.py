"""
hvEEGNet model configuration.

:class:`~jade.config.model.hvEEGNet_cfg.hvEEGNet_config` describes every hyper-parameter needed to build a :class:`~jade.model.hvEEGNet.hvEEGNet`, i.e. the hierarchical variational autoencoder whose encoder is an EEGNet, whose decoder mirrors it, and which has a latent space after each of the three blocks of the encoder.

As for :class:`~jade.config.model.vEEGNet_cfg.vEEGNet_config`, the encoder hyper-parameters are declared here explicitly (and not by nesting an :class:`~jade.config.model.EEGNet_cfg.EEGNet_config`), so a config file stays flat, it carries a single ``name`` and no classifier-only field is reachable.
The bridge to the encoder is :meth:`hvEEGNet_config.to_encoder_config`.

The default values are the ones of the original hvEEGNet (Cisotto et al., 2024) for the BCI IV-2a dataset (22 channels, 1000 time samples at 250 Hz), so ``hvEEGNet_config()`` with no argument is the model of the paper:

* temporal kernel ``(1, 128)`` and separable kernel ``(1, 32)``;
* no pooling after the spatial filter (the first stage is an identity) and a pooling of factor 10 after the separable convolution;
* :class:`torch.nn.Dropout2d` with probability 0.5.

The config is torch-free (like the rest of :mod:`jade.config`); the actual layers are built by :class:`~jade.model.hvEEGNet.hvEEGNet`.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from dataclasses import dataclass, field
from typing import ClassVar, Tuple

# Internal imports
from jade.config.model.generic import model_config
from jade.config.model.EEGNet_cfg import EEGNet_config, eegnet_pooling_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module constants

# Names of the models this config can describe (mirrors the aliases of jade.model.MODEL_REGISTRY).
ALLOWED_NAMES = ("hveegnet", "hierarchical_veegnet")

# Activation functions supported by jade.model.support_functions.get_activation.
ACTIVATIONS = ("relu", "elu", "selu", "gelu")

# How each latent space is mapped from (and back to) the feature map it is attached to.
# - 'convolutional': 1x1 convolutions. The latent variable has the shape of the feature map. This is the mapping of the hvEEGNet paper.
# - 'linear'       : the feature map is flattened and feedforward layers produce vectors of hidden_space_dimension_list[l] elements.
# - 'lora'         : as 'linear', but every feedforward layer is a low-rank lora_linear (see jade.model.support_functions).
PARAMETERS_MAP_TYPES = ("convolutional", "linear", "lora")

# Interpolation modes accepted by torch.nn.Upsample for a 4D tensor.
# Note that 'linear' is deliberately absent: it is valid only for a 3D tensor.
UPSAMPLE_MODES = ("nearest", "bilinear", "bicubic", "area")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Helper functions

def _is_int(value) -> bool :
    """
    Return True only for a genuine integer.

    ``bool`` is a subclass of ``int`` in Python, so it is excluded explicitly (a flag is never a valid size).
    """

    return isinstance(value, int) and not isinstance(value, bool)


def _check_positive_int(value, field_name : str, minimum : int = 1) -> None :
    """
    Raize ``ValueError`` if ``value`` is not an integer greater than or equal to ``minimum``.
    """

    if not _is_int(value) or value < minimum :
        raise ValueError(f"{field_name} must be an integer >= {minimum}, got {value!r}")


def _check_bool(value, field_name : str) -> None :
    """
    Raize ``ValueError`` if ``value`` is not a genuine ``bool``.
    """

    if not isinstance(value, bool) :
        raise ValueError(f"{field_name} must be a bool, got {value!r}")


def _coerce_kernel(value, field_name : str) :
    """
    Validate (and normalize) a 2D kernel size.

    A TOML/JSON round-trip turns a tuple into a list, so the value is coerced back to a tuple.

    Returns
    -------
    tuple
        The normalized ``(height, width)`` tuple.
    """

    if value is None :
        raise ValueError(f"{field_name} cannot be None")

    # Normalize a list coming from a file back into a tuple.
    value = tuple(value)
    if len(value) != 2 :
        raise ValueError(f"{field_name} must be a pair (height, width), got {value!r}")
    for element in value :
        if not _is_int(element) or element <= 0 :
            raise ValueError(f"{field_name} must contain positive integers, got {value!r}")

    return value


def _default_hveegnet_pooling() -> eegnet_pooling_config :
    """
    Pooling stages of the original hvEEGNet: no pooling after the spatial filter, an average pooling of factor 10 after the separable convolution.

    The first stage is disabled through the ``'identity'`` type (and not through a negative factor) so that building the default config does not emit the "stage disabled" warning of :class:`~jade.config.model.EEGNet_cfg.eegnet_pooling_config`.
    """

    return eegnet_pooling_config(
        pooling_type_1   = "identity",
        pooling_type_2   = "default",
        pooling_factor_1 = 1,
        pooling_factor_2 = 10,
    )

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# hvEEGNet config class

@dataclass
class hvEEGNet_config(model_config) :
    """
    Configuration for :class:`~jade.model.hvEEGNet.hvEEGNet`.

    The encoder fields mirror the ones of :class:`~jade.config.model.EEGNet_cfg.EEGNet_config` (the encoder *is* an EEGNet), with the same names; see :meth:`to_encoder_config`.
    With the default ``parameters_map_type = 'convolutional'`` (the hvEEGNet of the paper) each of the three latent spaces has the shape of the output of the corresponding encoder block; the vector mappings (``'linear'``, ``'lora'``) are kept from the original implementation for experiments.

    Parameters
    ----------
    name : str, default 'hveegnet'
        Which model to build. One of ``ALLOWED_NAMES``.
    use_bias : bool, default False
        If True the convolutional layers (of both the encoder and the decoder) use a bias term.
    D : int, default 2
        Number of spatial filters learned within each temporal convolution. The total number of spatial filters is ``D * filter_1``.
    activation : str, default 'elu'
        Activation function. One of ``ACTIVATIONS`` (case-insensitive). It is also applied before the ``1x1`` convolutions that produce the parameters of the latent distributions, as in the original implementation.
    prob_dropout : float, default 0.5
        Dropout probability, in ``[0, 1]``.
    use_dropout_2d : bool, default True
        If True use :class:`torch.nn.Dropout2d`, otherwise :class:`torch.nn.Dropout`.
    C : int, default 22
        Number of EEG channels of the input.
    T : int, default 1000
        Number of time samples of the input.
    depth_first_layer : int, default 1
        Depth of the input tensor (``1`` for raw EEG, ``> 1`` for time-frequency input).
    filter_1 : int, default 8
        Number of temporal filters (first convolution of the encoder).
    c_kernel_1 : tuple of int, default (1, 128)
        Kernel size of the temporal convolution.
    c_kernel_2 : tuple of int, default (22, 1)
        Kernel size of the spatial (depthwise) convolution. Usually ``(C, 1)``. Its first element must not exceed ``C``.
    filter_2 : int, default 16
        Number of filters of the separable convolution (pointwise output).
    c_kernel_3 : tuple of int, default (1, 32)
        Kernel size of the depthwise convolution of the separable block.
    parameters_map_type : str, default 'convolutional'
        How each latent space is mapped from (and back to) its feature map. One of ``PARAMETERS_MAP_TYPES``.
    hidden_space_dimension_list : tuple of int, default (32, 128, 512)
        Dimension of the three latent spaces, deepest first (``z_0``, ``z_1``, ``z_2``). Used only with the vector mappings (``'linear'``, ``'lora'``).
    lora_rank : int, default 16
        Rank of the :class:`~jade.model.support_functions.lora_linear` layers. Used only with ``parameters_map_type = 'lora'``.
    use_spectral_norm : bool, default False
        If True apply the spectral normalization to the feedforward layers of the vector mappings. Ignored with ``'convolutional'``.
    use_activation_in_sampling : bool, default True
        If True the activation is applied before the maps that produce the parameters of the latent distributions (and, with the vector mappings, after the projection of the sample back to a feature map), as in the original implementation.
    use_h_in_decoder : bool, default False
        If True the deepest decoder level combines the sample of ``z_0`` with ``h``, a trainable tensor with the shape of the encoder output (Fig. 2 of the NVAE paper).
        If False the combination is done with a tensor of zeros, as in the trained hvEEGNet of the paper. Either way an explicit ``h`` can be passed to :meth:`~jade.model.hvEEGNet.hvEEGNet.forward` and :meth:`~jade.model.hvEEGNet.hvEEGNet.generate`.
    upsample_mode : str, default 'nearest'
        Interpolation used by the decoder to invert the pooling stages. One of ``UPSAMPLE_MODES``.
    input_size : tuple of int, default (1, 1, 22, 1000)
        Shape ``(batch, depth, C, T)`` of the dummy input used to probe the geometry of the encoder at build time. It must agree with ``depth_first_layer``, ``C`` and ``T``.
    print_var : bool, default False
        If True print the parameter count and the shape tracking at build time.
    pooling : eegnet_pooling_config, default no pooling at stage 1 and average pooling of factor 10 at stage 2
        The two pooling stages of the encoder (the ``[pooling]`` section), shared with :class:`~jade.model.EEGNet.EEGNet`.
        As in vEEGNet, the decoder does not read the pooling factors: it resizes the tensor back to the shape each stage received, so every pooling operator works.

    Notes
    -----
    The size of the latent spaces grows quickly with the input: with the defaults the shallowest one (after the temporal filter) has ``filter_1 * C * T = 176000`` elements per trial.
    This is by design (the shallow latent spaces carry the fine details of the signal), but keep it in mind for a long input or many channels.
    """

    # The model is trained with a generative objective, so it must be paired with a generative loss (the ELBO).
    # It is a ClassVar, so dataclasses ignores it and it never appears in a serialized config.
    task : ClassVar[str] = "generative"

    name : str = "hveegnet"

    # Encoder (EEGNet). These fields mirror the corresponding ones of EEGNet_config, see to_encoder_config.
    use_bias          : bool = False
    D                 : int = 2
    activation        : str = "elu"
    prob_dropout      : float = 0.5
    use_dropout_2d    : bool = True
    C                 : int = 22
    T                 : int = 1000
    depth_first_layer : int = 1
    filter_1          : int = 8
    c_kernel_1        : Tuple[int, int] = (1, 128)
    c_kernel_2        : Tuple[int, int] = (22, 1)
    filter_2          : int = 16
    c_kernel_3        : Tuple[int, int] = (1, 32)

    # Latent spaces
    parameters_map_type         : str = "convolutional"
    hidden_space_dimension_list : Tuple[int, int, int] = (32, 128, 512)
    lora_rank                   : int = 16
    use_spectral_norm           : bool = False
    use_activation_in_sampling  : bool = True

    # Decoder
    use_h_in_decoder  : bool = False
    upsample_mode     : str = "nearest"

    # Misc
    input_size        : Tuple[int, int, int, int] = (1, 1, 22, 1000)
    print_var         : bool = False

    # Pooling (the [pooling] section). Declared last so it serializes after the scalar keys (a nested
    # config becomes a TOML sub-table, which must follow the parent's scalar keys).
    pooling           : eegnet_pooling_config = field(default_factory = _default_hveegnet_pooling)


    def __post_init__(self) -> None :
        """
        Validate (and normalize) every hvEEGNet parameter.
        """

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Discriminator and string options

        # Lower-cased, so "hvEEGNet" (the spelling of the class and of the paper) is accepted too.
        self.name = str(self.name).lower()
        if self.name not in ALLOWED_NAMES :
            raise ValueError(f"name must be one of {ALLOWED_NAMES}, got {self.name!r}")

        self.activation = str(self.activation).lower()
        if self.activation not in ACTIVATIONS :
            raise ValueError(f"activation must be one of {ACTIVATIONS}, got {self.activation!r}")

        self.parameters_map_type = str(self.parameters_map_type).lower()
        if self.parameters_map_type not in PARAMETERS_MAP_TYPES :
            raise ValueError(f"parameters_map_type must be one of {PARAMETERS_MAP_TYPES}, got {self.parameters_map_type!r}")

        self.upsample_mode = str(self.upsample_mode).lower()
        if self.upsample_mode not in UPSAMPLE_MODES :
            raise ValueError(f"upsample_mode must be one of {UPSAMPLE_MODES}, got {self.upsample_mode!r}")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Boolean flags

        _check_bool(self.use_bias, "use_bias")
        _check_bool(self.use_dropout_2d, "use_dropout_2d")
        _check_bool(self.print_var, "print_var")
        _check_bool(self.use_spectral_norm, "use_spectral_norm")
        _check_bool(self.use_activation_in_sampling, "use_activation_in_sampling")
        _check_bool(self.use_h_in_decoder, "use_h_in_decoder")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Positive integer sizes

        _check_positive_int(self.D, "D")
        _check_positive_int(self.filter_1, "filter_1")
        _check_positive_int(self.filter_2, "filter_2")
        _check_positive_int(self.C, "C")
        _check_positive_int(self.T, "T")
        _check_positive_int(self.depth_first_layer, "depth_first_layer")
        _check_positive_int(self.lora_rank, "lora_rank")

        # One dimension per latent space. Normalize a list coming from a file back into a tuple.
        self.hidden_space_dimension_list = tuple(self.hidden_space_dimension_list)
        if len(self.hidden_space_dimension_list) != 3 :
            raise ValueError(f"hidden_space_dimension_list must contain 3 dimensions (z_0, z_1, z_2), got {self.hidden_space_dimension_list!r}")
        for hidden_space_dimension in self.hidden_space_dimension_list : _check_positive_int(hidden_space_dimension, "hidden_space_dimension_list elements")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Dropout probability

        if not isinstance(self.prob_dropout, (int, float)) or isinstance(self.prob_dropout, bool) or not 0.0 <= self.prob_dropout <= 1.0 :
            raise ValueError(f"prob_dropout must be a float in [0, 1], got {self.prob_dropout!r}")
        self.prob_dropout = float(self.prob_dropout)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Convolution kernels

        self.c_kernel_1 = _coerce_kernel(self.c_kernel_1, "c_kernel_1")
        self.c_kernel_2 = _coerce_kernel(self.c_kernel_2, "c_kernel_2")
        self.c_kernel_3 = _coerce_kernel(self.c_kernel_3, "c_kernel_3")

        # The spatial convolution of the encoder is not padded, so a kernel taller than the number of channels is invalid.
        if self.c_kernel_2[0] > self.C :
            raise ValueError(f"c_kernel_2 spans {self.c_kernel_2[0]} channels but the input has only C = {self.C}")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Pooling section

        # Promote a dict to eegnet_pooling_config (direct-construction path; from_dict already promotes it).
        if isinstance(self.pooling, dict) :
            self.pooling = eegnet_pooling_config.from_dict(self.pooling)
        elif not isinstance(self.pooling, eegnet_pooling_config) :
            raise ValueError(f"pooling must be an eegnet_pooling_config (or a dict), got {type(self.pooling)}")

        # The two pooling stages decimate the time axis, so a signal that is too short leaves nothing to decode.
        # A disabled stage (negative factor or identity type) does not decimate anything.
        time_samples_left = self.T
        for pooling_type, pooling_factor in ((self.pooling.pooling_type_1, self.pooling.pooling_factor_1), (self.pooling.pooling_type_2, self.pooling.pooling_factor_2)) :
            if pooling_factor > 0 and pooling_type not in ("identity", "none") : time_samples_left = time_samples_left // pooling_factor
        if time_samples_left < 1 :
            raise ValueError(f"T = {self.T} is too short for the pooling factors ({self.pooling.pooling_factor_1}, {self.pooling.pooling_factor_2}): no time sample would be left after the encoder")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Input size (batch, depth, channels, time)

        # Normalize a list coming from a file back into a tuple.
        self.input_size = tuple(self.input_size)
        if len(self.input_size) != 4 :
            raise ValueError(f"input_size must be a 4-tuple (batch, depth, channels, time), got {self.input_size!r}")
        for element in self.input_size :
            if not _is_int(element) or element <= 0 :
                raise ValueError(f"input_size must contain positive integers, got {self.input_size!r}")

        # input_size drives the whole decoder geometry (see jade.model.hvEEGNet.hvEEGNet), so it cannot disagree with the other fields.
        expected_input_size = (self.depth_first_layer, self.C, self.T)
        if self.input_size[1:] != expected_input_size :
            raise ValueError(f"input_size must be (batch, depth_first_layer, C, T) = (batch, {expected_input_size[0]}, {expected_input_size[1]}, {expected_input_size[2]}), got {self.input_size!r}")


    def to_encoder_config(self) -> EEGNet_config :
        """
        Build the configuration of the EEGNet used as encoder.

        Two fields are forced: ``name = 'eegnet'`` (the plain feature extractor, not its classifier wrapper) and ``flatten_output = False`` (the latent spaces are convolutional, so they need the feature maps).

        Returns
        -------
        EEGNet_config
            A freshly built config. Note that the ``pooling`` section is shared by reference with this config.
        """

        return EEGNet_config(
            name              = "eegnet",
            use_bias          = self.use_bias,
            D                 = self.D,
            activation        = self.activation,
            prob_dropout      = self.prob_dropout,
            use_dropout_2d    = self.use_dropout_2d,
            C                 = self.C,
            T                 = self.T,
            depth_first_layer = self.depth_first_layer,
            filter_1          = self.filter_1,
            c_kernel_1        = self.c_kernel_1,
            c_kernel_2        = self.c_kernel_2,
            filter_2          = self.filter_2,
            c_kernel_3        = self.c_kernel_3,
            flatten_output    = False,
            input_size        = self.input_size,
            print_var         = False,
            pooling           = self.pooling,
        )
