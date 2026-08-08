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

# Full module imports
import warnings

# Specific imports
from dataclasses import dataclass, field
from typing import Tuple

# Internal imports
from jade.config.generic import generic_config
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
        raise ValueError(f"{field_name} must be an integer >= {minimum}, got {value!r}")


def _check_bool(value, field_name : str) -> None :
    """
    Raize ``ValueError`` if ``value`` is not a genuine ``bool``.
    """

    if not isinstance(value, bool) :
        raise ValueError(f"{field_name} must be a bool, got {value!r}")


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
# EEGNet pooling config (the [pooling] section)

@dataclass
class eegnet_pooling_config(generic_config) :
    """
    Pooling configuration for the two pooling stages of EEGNet (the ``[pooling]`` section).

    EEGNet has two pooling stages: the first inside the spatial-filter block, the second inside the separable-convolution block.
    Each stage is configured independently: its own pooling type, its own decimation factor, and its own type-specific parameters.

    The two derived properties :attr:`pooling_layer_1_config` and :attr:`pooling_layer_2_config` assemble the per-stage dicts consumed by :func:`jade.model.support_functions.get_multichannel_pooling`.

    Parameters
    ----------
    pooling_type_1 : str, default 'default'
        Pooling operator of the first stage. One of:
        - ``'default'`` : the classic EEGNet ``nn.AvgPool2d`` (average pooling along the time axis only);
        - a 1D pooling name: ``'avg'``, ``'max'``, ``'blur'``, ``'sinc'``, ``'dwt'`` (and their aliases), applied channel-wise through the ``*_multichannel`` wrappers of :mod:`jade.model.pooling_1d`.
        Case-insensitive. Any other value raises ``ValueError``.
    pooling_type_2 : str, default 'default'
        Same as ``pooling_type_1`` but for the second stage.
    pooling_factor_1 : int, default 4
        Decimation factor ``M`` of the first stage: the number of time samples becomes ``T / M``.
        Must be a non-zero integer. A negative value disables the first pooling stage (a warning is emitted at instantiation).
    pooling_factor_2 : int, default 8
        Same as ``pooling_factor_1`` but for the second stage.
    params_1 : dict, default {}
        Extra, type-specific parameters for the first stage (e.g. ``{'wavelet' : 'sym4'}`` for ``'dwt'``, ``{'cutoff_scale' : 0.8}`` for ``'sinc'``).
        Empty means "use the pooling layer defaults". See :mod:`jade.model.pooling_1d` for the accepted keys of each operator.
    params_2 : dict, default {}
        Same as ``params_1`` but for the second stage.
    """

    pooling_type_1   : str  = "default"
    pooling_type_2   : str  = "default"
    pooling_factor_1 : int  = 4
    pooling_factor_2 : int  = 8
    params_1         : dict = field(default_factory = dict)
    params_2         : dict = field(default_factory = dict)


    @property
    def pooling_layer_1_config(self) -> dict :
        """
        Per-stage config dict for the first pooling stage.

        Consumed by :func:`jade.model.support_functions.get_multichannel_pooling`.
        """

        return {"pooling_type" : self.pooling_type_1, "decimation_factor" : self.pooling_factor_1, **self.params_1}

    @property
    def pooling_layer_2_config(self) -> dict :
        """
        Per-stage config dict for the second pooling stage.

        Consumed by :func:`jade.model.support_functions.get_multichannel_pooling`.
        """

        return {"pooling_type" : self.pooling_type_2, "decimation_factor" : self.pooling_factor_2, **self.params_2}


    def __post_init__(self) -> None :
        """
        Validate (and normalize) the pooling configuration of both stages.
        """

        # Lazy import to avoid an import cycle: this module is imported while jade.config.model is still initializing.
        from jade.config.model.pooling_1d_cfg import POOLING_CONFIG_REGISTRY, get_pooling_config

        # Valid pooling_type values: the 1D pooling names, plus 'default' (AvgPool2d) and the explicit no-op aliases.
        valid_types = set(POOLING_CONFIG_REGISTRY) | {"default", "identity", "none"}

        for suffix in ("1", "2") :
            type_attr   = f"pooling_type_{suffix}"
            factor_attr = f"pooling_factor_{suffix}"
            params_attr = f"params_{suffix}"

            # Pooling type (lower-cased and validated).
            pooling_type = str(getattr(self, type_attr)).lower()
            setattr(self, type_attr, pooling_type)
            if pooling_type not in valid_types :
                raise ValueError(f"{type_attr} must be one of {sorted(valid_types)}, got {pooling_type!r}")

            # Pooling factor: a non-zero integer. Negative disables the stage (documented behavior).
            factor = getattr(self, factor_attr)
            if not _is_int(factor) or factor == 0 :
                raise ValueError(f"{factor_attr} must be a non-zero integer, got {factor!r}")
            if factor < 0 :
                warnings.warn(
                    f"{factor_attr} = {factor} is negative: EEGNet pooling stage {suffix} is disabled "
                    "(replaced by nn.Identity).",
                    stacklevel = 2,
                )

            # Params: a dict of type-specific options.
            params = getattr(self, params_attr)
            if not isinstance(params, dict) :
                raise ValueError(f"{params_attr} must be a dict, got {params!r}")

            # Validate the type-specific params by round-tripping through the 1D pooling config family.
            # Only meaningful for an *active* (positive factor) non-'default' 1D pooling layer.
            if factor > 0 and pooling_type in POOLING_CONFIG_REGISTRY :
                merged = {"pooling_type" : pooling_type, "decimation_factor" : factor, **params}
                get_pooling_config(merged)

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
    filter_2 : int, default 16
        Number of filters of the separable convolution (pointwise output).
    c_kernel_3 : tuple of int, default (1, 16)
        Kernel size of the depthwise convolution of the separable block.
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
    pooling : eegnet_pooling_config, default eegnet_pooling_config()
        The two pooling stages (the ``[pooling]`` section). It replaces the old ``p_kernel_1`` / ``p_kernel_2`` fields.
        Each stage has its own ``pooling_type`` (``'default'`` = classic ``nn.AvgPool2d``, or a 1D pooling operator applied channel-wise), decimation factor and type-specific parameters.
        A dict is accepted and promoted to :class:`eegnet_pooling_config`.
        The derived ``pooling_layer_1_config`` / ``pooling_layer_2_config`` are what :class:`~jade.model.EEGNet.EEGNet` feeds to :func:`~jade.model.support_functions.get_multichannel_pooling`.
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
    filter_2          : int = 16
    c_kernel_3        : Tuple[int, int] = (1, 16)
    flatten_output    : bool = False

    # Classifier head
    input_size        : Tuple[int, int, int, int] = (1, 1, 22, 1000)
    n_classes         : int = 4
    use_softmax       : bool = True

    # Misc
    print_var         : bool = False

    # Pooling (the [pooling] section). Declared last so it serializes after the scalar keys (a nested
    # config becomes a TOML sub-table, which must follow the parent's scalar keys).
    pooling           : eegnet_pooling_config = field(default_factory = eegnet_pooling_config)


    def __post_init__(self) -> None :
        """
        Validate (and normalize) every EEGNet parameter.
        """

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Discriminator and activation

        # The model name selects the class in jade.model.get_model, so it must be one of the known models.
        self.name = str(self.name).lower()
        if self.name not in ALLOWED_NAMES :
            raise ValueError(f"name must be one of {ALLOWED_NAMES}, got {self.name!r}")

        self.activation = str(self.activation).lower()
        if self.activation not in ACTIVATIONS :
            raise ValueError(f"activation must be one of {ACTIVATIONS}, got {self.activation!r}")

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
            raise ValueError(f"prob_dropout must be a float in [0, 1], got {self.prob_dropout!r}")
        self.prob_dropout = float(self.prob_dropout)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Convolution kernels

        self.c_kernel_1 = _coerce_kernel(self.c_kernel_1, "c_kernel_1")
        self.c_kernel_2 = _coerce_kernel(self.c_kernel_2, "c_kernel_2")
        self.c_kernel_3 = _coerce_kernel(self.c_kernel_3, "c_kernel_3")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Pooling section

        # Promote a dict to eegnet_pooling_config (direct-construction path; from_dict already promotes it).
        if isinstance(self.pooling, dict) :
            self.pooling = eegnet_pooling_config.from_dict(self.pooling)
        elif not isinstance(self.pooling, eegnet_pooling_config) :
            raise ValueError(f"pooling must be an eegnet_pooling_config (or a dict), got {type(self.pooling)}")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Input size (batch, depth, channels, time)

        # Normalize a list coming from a file back into a tuple.
        self.input_size = tuple(self.input_size)
        if len(self.input_size) != 4 :
            raise ValueError(f"input_size must be a 4-tuple (batch, depth, channels, time), got {self.input_size!r}")
        for element in self.input_size :
            if not _is_int(element) or element <= 0 :
                raise ValueError(f"input_size must contain positive integers, got {self.input_size!r}")


    @property
    def pooling_layer_1_config(self) -> dict :
        """
        Per-stage config dict for the first pooling stage (delegates to :attr:`pooling`).

        This is what :class:`~jade.model.EEGNet.EEGNet` passes to :func:`~jade.model.support_functions.get_multichannel_pooling`.
        """

        return self.pooling.pooling_layer_1_config

    @property
    def pooling_layer_2_config(self) -> dict :
        """
        Per-stage config dict for the second pooling stage (delegates to :attr:`pooling`).

        This is what :class:`~jade.model.EEGNet.EEGNet` passes to :func:`~jade.model.support_functions.get_multichannel_pooling`.
        """

        return self.pooling.pooling_layer_2_config
