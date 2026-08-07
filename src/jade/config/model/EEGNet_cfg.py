"""
EEGNet model configuration.

:class:`~jade.config.model.EEGNet_cfg.EEGNet_config` describes every hyper-parameter needed to build an :class:`~jade.model.EEGNet.EEGNet` and its classifier wrapper :class:`~jade.model.EEGNet.EEGNet_Classifier`.
The two share a single config because the classifier is just an EEGNet feature extractor plus a feedforward head, so the classifier keys are a superset of the plain-EEGNet ones.
Which of the two is actually built is decided by the ``name`` discriminator through :func:`~jade.model.get_model`.

Every field has a sensible (BCI IV-2a style) default, so ``EEGNet_config()`` with no argument is a valid config.
The default values purpose is to give a working skeleton that is easy to tweak, not a tuned model.
The config is torch-free (like the rest of :mod:`jade.config`); the actual layers are built by :class:`~jade.model.EEGNet.EEGNet`.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from dataclasses import dataclass
from typing import Optional, Tuple

# Internal imports
from jade.config.model.generic import model_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module constants

# Names of the models this config can describe (mirrors the aliases of jade.model.MODEL_REGISTRY).
ALLOWED_NAMES = ("eegnet", "eegnet_classifier", "eegnetclassifier")

# Activation functions supported by jade.model.support_functions.get_activation.
ACTIVATIONS = ("relu", "elu", "selu", "gelu")

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
        raise ValueError("{} must be an integer >= {}, got {}".format(field_name, minimum, repr(value)))


def _check_bool(value, field_name : str) -> None :
    """
    Raize ``ValueError`` if ``value`` is not a genuine ``bool``.
    """

    if not isinstance(value, bool) :
        raise ValueError("{} must be a bool, got {}".format(field_name, repr(value)))


def _coerce_kernel(value, field_name : str, allow_none : bool = False) :
    """
    Validate (and normalize) a 2D kernel/pool size.

    A TOML/JSON round-trip turns a tuple into a list, so the value is coerced back to a tuple.
    Pooling kernels can be ``None`` (meaning no pooling), which is only accepted when ``allow_none`` is True.

    Returns
    -------
    tuple or None
        The normalized ``(height, width)`` tuple (or ``None``).
    """

    # None handling (used by the optional pooling kernels).
    if value is None :
        if allow_none :
            return None
        raise ValueError("{} cannot be None".format(field_name))

    # Normalize a list coming from a file back into a tuple.
    value = tuple(value)
    if len(value) != 2 :
        raise ValueError("{} must be a pair (height, width), got {}".format(field_name, repr(value)))
    for element in value :
        if not _is_int(element) or element <= 0 :
            raise ValueError("{} must contain positive integers, got {}".format(field_name, repr(value)))

    return value

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# EEGNet config class

@dataclass
class EEGNet_config(model_config) :
    """
    Configuration for :class:`~jade.model.EEGNet.EEGNet` / :class:`~jade.model.EEGNet.EEGNet_Classifier`.

    The variable names respect the nomenclature of the original EEGNet paper (Lawhern et al.) where possible.

    Parameters
    ----------
    name : str, default 'eegnet_classifier'
        Which model to build. One of ``'eegnet'`` (feature extractor) or ``'eegnet_classifier'`` (feature extractor + feedforward head).
    use_bias : bool, default False
        If True the convolutional layers use a bias term.
    D : int, default 2
        Number of spatial filters learned within each temporal convolution. The total number of spatial filters is ``D * filter_1``.
    activation : str, default 'elu'
        Activation function. One of ``relu``, ``elu``, ``selu``, ``gelu`` (case-insensitive).
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
        Number of temporal filters (first convolution).
    c_kernel_1 : tuple of int, default (1, 64)
        Kernel size of the temporal convolution.
    c_kernel_2 : tuple of int, default (22, 1)
        Kernel size of the spatial (depthwise) convolution. Usually ``(C, 1)``.
    p_kernel_1 : tuple of int or None, default (1, 4)
        Kernel size of the first average pooling. ``None`` disables the first pooling.
    filter_2 : int, default 16
        Number of filters of the separable convolution (pointwise output).
    c_kernel_3 : tuple of int, default (1, 16)
        Kernel size of the depthwise convolution of the separable block.
    p_kernel_2 : tuple of int or None, default (1, 8)
        Kernel size of the second average pooling. ``None`` disables the second pooling.
    flatten_output : bool, default False
        If True the (plain) EEGNet flattens its output to ``(batch, n_features)``. Ignored by the classifier (which always flattens internally).
    input_size : tuple of int, default (1, 1, 22, 1000)
        Shape ``(batch, depth, C, T)`` of a dummy input used by :class:`~jade.model.EEGNet.EEGNet_Classifier` to size its feedforward layer.
    n_classes : int, default 4
        Number of output classes of the classifier head.
    use_softmax : bool, default True
        If True the classifier head ends with :class:`torch.nn.LogSoftmax` (log-probabilities, pair with an NLL loss);
        if False it ends with :class:`torch.nn.Identity` (raw logits, pair with a cross-entropy loss).
        See :func:`~jade.training.support_training.cross_check_config`.
    print_var : bool, default False
        If True print the parameter count (and, when ``input_size`` is given, the shape tracking) at build time.
    """

    name : str = "eegnet_classifier"

    # EEGNet feature extractor
    use_bias          : bool = False
    D                 : int = 2
    activation        : str = "elu"
    prob_dropout      : float = 0.5
    use_dropout_2d    : bool = False
    C                 : int = 22
    T                 : int = 1000
    depth_first_layer : int = 1
    filter_1          : int = 8
    c_kernel_1        : Tuple[int, int] = (1, 64)
    c_kernel_2        : Tuple[int, int] = (22, 1)
    p_kernel_1        : Optional[Tuple[int, int]] = (1, 4)
    filter_2          : int = 16
    c_kernel_3        : Tuple[int, int] = (1, 16)
    p_kernel_2        : Optional[Tuple[int, int]] = (1, 8)
    flatten_output    : bool = False

    # Classifier head
    input_size        : Tuple[int, int, int, int] = (1, 1, 22, 1000)
    n_classes         : int = 4
    use_softmax       : bool = True

    # Misc
    print_var         : bool = False


    def __post_init__(self) -> None :
        """
        Validate (and normalize) every EEGNet parameter.
        """

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Discriminator and activation

        # The model name selects the class in jade.model.get_model, so it must be one of the known models.
        self.name = str(self.name).lower()
        if self.name not in ALLOWED_NAMES :
            raise ValueError("name must be one of {}, got {}".format(ALLOWED_NAMES, repr(self.name)))

        self.activation = str(self.activation).lower()
        if self.activation not in ACTIVATIONS :
            raise ValueError("activation must be one of {}, got {}".format(ACTIVATIONS, repr(self.activation)))

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Boolean flags

        _check_bool(self.use_bias, "use_bias")
        _check_bool(self.use_dropout_2d, "use_dropout_2d")
        _check_bool(self.flatten_output, "flatten_output")
        _check_bool(self.use_softmax, "use_softmax")
        _check_bool(self.print_var, "print_var")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Positive integer sizes

        _check_positive_int(self.D, "D")
        _check_positive_int(self.filter_1, "filter_1")
        _check_positive_int(self.filter_2, "filter_2")
        _check_positive_int(self.C, "C")
        _check_positive_int(self.T, "T")
        _check_positive_int(self.depth_first_layer, "depth_first_layer")
        _check_positive_int(self.n_classes, "n_classes", minimum = 2)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Dropout probability

        if not isinstance(self.prob_dropout, (int, float)) or isinstance(self.prob_dropout, bool) or not 0.0 <= self.prob_dropout <= 1.0 :
            raise ValueError("prob_dropout must be a float in [0, 1], got {}".format(repr(self.prob_dropout)))
        self.prob_dropout = float(self.prob_dropout)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Convolution and pooling kernels

        self.c_kernel_1 = _coerce_kernel(self.c_kernel_1, "c_kernel_1")
        self.c_kernel_2 = _coerce_kernel(self.c_kernel_2, "c_kernel_2")
        self.c_kernel_3 = _coerce_kernel(self.c_kernel_3, "c_kernel_3")
        self.p_kernel_1 = _coerce_kernel(self.p_kernel_1, "p_kernel_1", allow_none = True)
        self.p_kernel_2 = _coerce_kernel(self.p_kernel_2, "p_kernel_2", allow_none = True)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Input size (batch, depth, channels, time)

        # Normalize a list coming from a file back into a tuple.
        self.input_size = tuple(self.input_size)
        if len(self.input_size) != 4 :
            raise ValueError("input_size must be a 4-tuple (batch, depth, channels, time), got {}".format(repr(self.input_size)))
        for element in self.input_size :
            if not _is_int(element) or element <= 0 :
                raise ValueError("input_size must contain positive integers, got {}".format(repr(self.input_size)))
