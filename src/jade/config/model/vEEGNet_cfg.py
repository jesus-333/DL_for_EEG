"""
vEEGNet model configuration.

:class:`~jade.config.model.vEEGNet_cfg.vEEGNet_config` describes every hyper-parameter needed to build a :class:`~jade.model.vEEGNet.vEEGNet`, i.e. the variational autoencoder whose encoder is an EEGNet and whose decoder mirrors it.

The encoder hyper-parameters are declared here explicitly, and not by nesting an :class:`~jade.config.model.EEGNet_cfg.EEGNet_config`, for the same reason every other model config of this subpackage re-declares the fields it shares with EEGNet: a config file stays flat (a single ``[pooling]`` sub-table), it carries a single ``name``, and no classifier-only field (``n_classes``, ``use_softmax``) is reachable on a model that has no classifier.
The bridge in the other direction is :meth:`vEEGNet_config.to_encoder_config`, which builds the EEGNet config of the encoder, so the encoder is created in exactly one place.

Every field has a sensible (BCI IV-2a style) default, so ``vEEGNet_config()`` with no argument is a valid config.
The default values purpose is to give a working skeleton that is easy to tweak, not a tuned model.
The config is torch-free (like the rest of :mod:`jade.config`); the actual layers are built by :class:`~jade.model.vEEGNet.vEEGNet`.

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
ALLOWED_NAMES = ("veegnet", "vae_eegnet")

# Activation functions supported by jade.model.support_functions.get_activation.
ACTIVATIONS = ("relu", "elu", "selu", "gelu")

# How the encoder maps its output to the parameters of the latent distribution.
# - 'linear'       : the output is flattened and two feedforward layers produce the mean and the log variance. The latent variable is a vector. This is the mapping of the original vEEGNet paper.
# - 'convolutional': two 1x1 convolutions produce the mean and the log variance. The latent variable keeps the shape of the encoder output.
PARAMETERS_MAP_TYPES = ("linear", "convolutional")

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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# vEEGNet config class

@dataclass
class vEEGNet_config(model_config) :
    """
    Configuration for :class:`~jade.model.vEEGNet.vEEGNet`.

    The encoder fields mirror the ones of :class:`~jade.config.model.EEGNet_cfg.EEGNet_config` (the encoder *is* an EEGNet), with the same names and the same defaults; see :meth:`to_encoder_config`.
    Note that there is no ``n_classes`` and no ``use_softmax`` field: the classification branch of the original vEEGNet is not implemented, so this model is purely generative.

    Parameters
    ----------
    name : str, default 'veegnet'
        Which model to build. One of ``ALLOWED_NAMES``.
    use_bias : bool, default False
        If True the convolutional layers (of both the encoder and the decoder) use a bias term.
    D : int, default 2
        Number of spatial filters learned within each temporal convolution. The total number of spatial filters is ``D * filter_1``.
    activation : str, default 'elu'
        Activation function. One of ``ACTIVATIONS`` (case-insensitive).
    prob_dropout : float, default 0.5
        Dropout probability, in ``[0, 1]``.
    use_dropout_2d : bool, default False
        If True use :class:`torch.nn.Dropout2d`, otherwise :class:`torch.nn.Dropout`.
    C : int, default 22
        Number of EEG channels of the input.
    T : int, default 1000
        Number of time samples of the input.
    depth_first_layer : int, default 1
        Depth of the input tensor (``1`` for raw EEG, ``> 1`` for time-frequency input).
    filter_1 : int, default 8
        Number of temporal filters (first convolution of the encoder).
    c_kernel_1 : tuple of int, default (1, 64)
        Kernel size of the temporal convolution.
    c_kernel_2 : tuple of int, default (22, 1)
        Kernel size of the spatial (depthwise) convolution. Usually ``(C, 1)``. Its first element must not exceed ``C``.
    filter_2 : int, default 16
        Number of filters of the separable convolution (pointwise output).
    c_kernel_3 : tuple of int, default (1, 16)
        Kernel size of the depthwise convolution of the separable block.
    parameters_map_type : str, default 'linear'
        How the encoder output is mapped to the mean and the log variance of the approximate posterior. One of ``PARAMETERS_MAP_TYPES``.
        With ``'linear'`` (the mapping of the original paper) the encoder output is flattened and two feedforward layers produce two vectors of ``hidden_space_dimension`` elements.
        With ``'convolutional'`` two ``1x1`` convolutions produce two tensors with the same shape as the encoder output, so the latent space keeps the time axis.
    hidden_space_dimension : int, default 64
        Dimension ``d`` of the latent space. It is the value used in the original paper.
        It is used only when ``parameters_map_type = 'linear'``: with the convolutional mapping the size of the latent space is set by the shape of the encoder output.
    upsample_mode : str, default 'nearest'
        Interpolation used by the decoder to invert the pooling stages. One of ``UPSAMPLE_MODES``.
    input_size : tuple of int, default (1, 1, 22, 1000)
        Shape ``(batch, depth, C, T)`` of the dummy input used to probe the geometry of the encoder at build time.
        Contrary to the classifiers, where it only sizes the feedforward head, here it drives the whole decoder, so it must agree with ``depth_first_layer``, ``C`` and ``T``.
    print_var : bool, default False
        If True print the parameter count and the shape tracking at build time.
    pooling : eegnet_pooling_config, default eegnet_pooling_config()
        The two pooling stages of the encoder (the ``[pooling]`` section), shared with :class:`~jade.model.EEGNet.EEGNet`.
        Note that the decoder does not read the pooling factors: it inverts each stage by resizing the tensor back to the shape that stage received (see :class:`~jade.model.vEEGNet.eegnet_decoder`), which works for every pooling operator.

    Notes
    -----
    The 1D pooling operators are *not* invertible (they low-pass, discard a sub-band or are non-linear), so the decoder restores the geometry and the transposed convolutions learn to repair the content.
    This matters for the anti-aliasing ablation: each arm changes how much information reaches the latent space, so the reconstruction errors of two arms compare different bottlenecks, not only different filters.
    """

    # The model is trained with a generative objective, so it must be paired with a generative loss (the ELBO).
    # It is a ClassVar, so dataclasses ignores it: it is a property of the model, not a user-editable parameter, and it never appears in a serialized config.
    task : ClassVar[str] = "generative"

    name : str = "veegnet"

    # Encoder (EEGNet). These fields mirror the corresponding ones of EEGNet_config, see to_encoder_config.
    use_bias               : bool = False
    D                      : int = 2
    activation             : str = "elu"
    prob_dropout           : float = 0.5
    use_dropout_2d         : bool = False
    C                      : int = 22
    T                      : int = 1000
    depth_first_layer      : int = 1
    filter_1               : int = 8
    c_kernel_1             : Tuple[int, int] = (1, 64)
    c_kernel_2             : Tuple[int, int] = (22, 1)
    filter_2               : int = 16
    c_kernel_3             : Tuple[int, int] = (1, 16)

    # Latent space
    parameters_map_type    : str = "linear"
    hidden_space_dimension : int = 64

    # Decoder
    upsample_mode          : str = "nearest"

    # Misc
    input_size             : Tuple[int, int, int, int] = (1, 1, 22, 1000)
    print_var              : bool = False

    # Pooling (the [pooling] section). Declared last so it serializes after the scalar keys (a nested
    # config becomes a TOML sub-table, which must follow the parent's scalar keys).
    pooling                : eegnet_pooling_config = field(default_factory = eegnet_pooling_config)


    def __post_init__(self) -> None :
        """
        Validate (and normalize) every vEEGNet parameter.
        """

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Discriminator, activation and the two string options

        # The model name selects the class in jade.model.get_model, so it must be one of the known models.
        # Note that the name is lower-cased, so "vEEGNet" (the spelling of the class and of the paper) is accepted too.
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

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Positive integer sizes

        _check_positive_int(self.D, "D")
        _check_positive_int(self.filter_1, "filter_1")
        _check_positive_int(self.filter_2, "filter_2")
        _check_positive_int(self.C, "C")
        _check_positive_int(self.T, "T")
        _check_positive_int(self.depth_first_layer, "depth_first_layer")
        _check_positive_int(self.hidden_space_dimension, "hidden_space_dimension")

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
        # The check is done here (and not left to torch) because the decoder inverts that convolution to restore the C channels, so the constraint belongs to the model geometry.
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
        # A negative factor disables the stage, so it does not decimate anything.
        time_samples_left = self.T
        for pooling_factor in (self.pooling.pooling_factor_1, self.pooling.pooling_factor_2) :
            if pooling_factor > 0 : time_samples_left = time_samples_left // pooling_factor
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

        # Contrary to the classifiers, input_size drives the whole decoder geometry (see jade.model.vEEGNet.vEEGNet), so it cannot disagree with the other fields.
        expected_input_size = (self.depth_first_layer, self.C, self.T)
        if self.input_size[1:] != expected_input_size :
            raise ValueError(f"input_size must be (batch, depth_first_layer, C, T) = (batch, {expected_input_size[0]}, {expected_input_size[1]}, {expected_input_size[2]}), got {self.input_size!r}")


    def to_encoder_config(self) -> EEGNet_config :
        """
        Build the configuration of the EEGNet used as encoder.

        Two fields are forced, whatever the user asked for:

        * ``name = 'eegnet'``, so the plain feature extractor is built and not its classifier wrapper;
        * ``flatten_output = False``, because the sampling layer (and, with the convolutional mapping, the decoder) needs the feature map and not a flat vector.

        The classifier-only fields of :class:`~jade.config.model.EEGNet_cfg.EEGNet_config` (``n_classes``, ``use_softmax``) are left at their defaults: a plain :class:`~jade.model.EEGNet.EEGNet` never reads them.

        Returns
        -------
        EEGNet_config
            A freshly built config. Note that the ``pooling`` section is shared by reference with this config, so the encoder and this object always describe the same pooling stages.
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
