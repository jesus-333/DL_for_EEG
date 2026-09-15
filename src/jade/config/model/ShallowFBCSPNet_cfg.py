"""
ShallowFBCSPNet (Shallow ConvNet) model configuration.

:class:`~jade.config.model.ShallowFBCSPNet_cfg.ShallowFBCSPNet_config` describes every hyper-parameter needed to build a :class:`~jade.model.ShallowFBCSPNet.ShallowFBCSPNet` and its classifier wrapper :class:`~jade.model.ShallowFBCSPNet.ShallowFBCSPNet_Classifier`.
The two share a single config because the classifier is just a ShallowFBCSPNet feature extractor plus a feedforward classification head, so the classifier keys are a superset of the plain ones.
Which of the two is actually built is decided by the ``name`` discriminator through :func:`~jade.model.get_model`.

The model is the one presented as ``Shallow ConvNet`` in ``Deep learning with convolutional neural networks for EEG decoding and visualization`` (Schirrmeister et al., Human Brain Mapping, 2017, https://doi.org/10.1002/hbm.23730), better known through the ``ShallowFBCSPNet`` name of its reference implementation.
The variable names (and the default values) follow that reference implementation where possible, so a bare ``ShallowFBCSPNet_config()`` reproduces the original BCI IV-2a architecture (40 temporal filters of length 25, 40 spatial filters, a squaring nonlinearity, an overlapping average pooling of 75 samples with stride 15 and a logarithm).

The config is torch-free (like the rest of :mod:`jade.config`); the actual layers are built by :class:`~jade.model.ShallowFBCSPNet.ShallowFBCSPNet`.
The network has a single pooling stage (the ``[pooling]`` section), which can reproduce the original overlapping average pooling (``'default'``) or swap in one of the custom :mod:`jade.model.pooling_1d` operators applied channel-wise.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from dataclasses import dataclass, field
from typing import Tuple

# Internal imports
from jade.config.generic import generic_config
from jade.config.model.generic import model_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module constants

# Names of the models this config can describe (mirrors the aliases of jade.model.MODEL_REGISTRY).
ALLOWED_NAMES = ("shallowfbcspnet", "shallowfbcspnet_classifier", "shallowfbcspnetclassifier")

# Nonlinearities accepted around the pooling stage: the two paper-specific ones (square before the pooling,
# safe logarithm after it), the no-op, plus every activation of jade.model.support_functions.get_activation.
NONLINEARITIES = ("square", "safe_log", "identity", "relu", "leaky_relu", "elu", "selu", "gelu")

# Pooling modes of the 'default' (original) pooling stage.
POOL_MODES = ("mean", "max")

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


def _check_probability(value, field_name : str) -> float :
    """
    Raize ``ValueError`` if ``value`` is not a number in ``[0, 1]``, otherwise return it as a ``float``.

    ``bool`` is excluded explicitly (it subclasses ``int``, so a flag would otherwise slip through as 0/1).
    """

    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= value <= 1.0 :
        raise ValueError(f"{field_name} must be a float in [0, 1], got {value!r}")

    return float(value)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# ShallowFBCSPNet pooling config (the [pooling] section)

@dataclass
class shallowfbcspnet_pooling_config(generic_config) :
    """
    Pooling configuration for the single pooling stage of ShallowFBCSPNet (the ``[pooling]`` section).

    The pooling stage follows the squaring nonlinearity and, together with the logarithm that follows it, implements the log-variance feature of the FBCSP pipeline the model is inspired by.
    Its window is **overlapping** in the original model (75 samples every 15), so, exactly like the EEG Conformer patch embedding, the averaging window and the decimation factor are two separate keys.
    This stage can either reproduce the original overlapping pooling or swap in one of the anti-aliased 1D operators of :mod:`jade.model.pooling_1d`, applied channel-wise.

    The derived property :attr:`pooling_layer_config` assembles the per-stage config consumed by :class:`~jade.model.ShallowFBCSPNet.ShallowFBCSPNet` (through :func:`~jade.model.ShallowFBCSPNet.get_shallowfbcspnet_pooling`).

    Parameters
    ----------
    pooling_type : str, default 'default'
        Pooling operator of the stage. One of:
        - ``'default'`` : the original ShallowFBCSPNet pooling, an **overlapping** ``nn.AvgPool2d`` (or ``nn.MaxPool2d``, see ``pool_mode``) with kernel ``(1, pooling_kernel)`` and stride ``(1, pooling_stride)`` (pooling along the time axis only);
        - a 1D pooling name: ``'avg'``, ``'max'``, ``'blur'``, ``'sinc'``, ``'dwt'`` (and their aliases), applied channel-wise through the ``*_multichannel`` wrappers of :mod:`jade.model.pooling_1d` with decimation factor ``pooling_stride``;
        - ``'identity'`` / ``'none'`` : no pooling (the stage is replaced by ``nn.Identity``).
        Case-insensitive. Any other value raises ``ValueError``.
    pooling_kernel : int, default 75
        Length of the averaging window of the ``'default'`` overlapping pooling. Ignored by the 1D operators (which derive their kernel length from the decimation factor). Must be a positive integer.
    pooling_stride : int, default 15
        Stride of the ``'default'`` pooling and decimation factor ``M`` of the 1D operators (the number of time samples becomes ``T / M``). Must be a positive integer.
        Note that the ``'dwt'`` operator additionally requires this to be a power of two, so the original stride 15 is not usable with it (use 16, or another power of two).
    pool_mode : str, default 'mean'
        Which pooling the ``'default'`` type builds, ``'mean'`` (``nn.AvgPool2d``, the original) or ``'max'`` (``nn.MaxPool2d``). Ignored by the 1D operators (there the operator itself is the choice).
    params : dict, default {}
        Extra, type-specific parameters for a 1D operator (e.g. ``{'wavelet' : 'sym4'}`` for ``'dwt'``, ``{'cutoff_scale' : 0.8}`` for ``'sinc'``).
        Empty means "use the pooling layer defaults". See :mod:`jade.model.pooling_1d` for the accepted keys of each operator. Ignored by the ``'default'`` / ``'identity'`` types.
    """

    pooling_type   : str  = "default"
    pooling_kernel : int  = 75
    pooling_stride : int  = 15
    pool_mode      : str  = "mean"
    params         : dict = field(default_factory = dict)


    @property
    def pooling_layer_config(self) :
        """
        Per-stage pooling config consumed by :class:`~jade.model.ShallowFBCSPNet.ShallowFBCSPNet`.

        For an active 1D pooling operator it is a :class:`~jade.config.model.pooling_1d_cfg.pooling_config` dataclass (decimation factor ``pooling_stride`` plus the type-specific ``params``), keeping the whole flow dataclass-based;
        for the ShallowFBCSPNet-only ``'default'`` (overlapping pooling) and ``'identity'`` / ``'none'`` types, which have no config dataclass, it is a plain ``{pooling_type, pooling_kernel, pooling_stride, pool_mode, **params}`` dict.
        Consumed by :func:`jade.model.ShallowFBCSPNet.get_shallowfbcspnet_pooling`, which accepts either.

        Returns
        -------
        pooling_config or dict
            The per-stage pooling configuration (a dataclass for an active 1D operator, a dict otherwise).
        """

        # Lazy import to avoid an import cycle (see __post_init__).
        from jade.config.model.pooling_1d_cfg import POOLING_CONFIG_REGISTRY, get_pooling_config

        # A real, active 1D operator has a config dataclass -> return it as a dataclass.
        # 'default' / 'identity' / 'none' have none -> keep the plain dict.
        if self.pooling_type in POOLING_CONFIG_REGISTRY :
            merged = {"pooling_type" : self.pooling_type, "decimation_factor" : self.pooling_stride, **self.params}
            return get_pooling_config(merged)

        return {"pooling_type" : self.pooling_type, "pooling_kernel" : self.pooling_kernel, "pooling_stride" : self.pooling_stride, "pool_mode" : self.pool_mode, **self.params}


    def __post_init__(self) -> None :
        """
        Validate (and normalize) the pooling configuration.
        """

        # Lazy import to avoid an import cycle: this module is imported while jade.config.model is still initializing.
        from jade.config.model.pooling_1d_cfg import POOLING_CONFIG_REGISTRY, get_pooling_config

        # Valid pooling_type values: the 1D pooling names, plus 'default' (overlapping AvgPool2d/MaxPool2d) and the explicit no-op aliases.
        valid_types = set(POOLING_CONFIG_REGISTRY) | {"default", "identity", "none"}

        # Pooling type (lower-cased and validated).
        self.pooling_type = str(self.pooling_type).lower()
        if self.pooling_type not in valid_types :
            raise ValueError(f"pooling_type must be one of {sorted(valid_types)}, got {self.pooling_type!r}")

        # Pooling mode of the 'default' stage.
        self.pool_mode = str(self.pool_mode).lower()
        if self.pool_mode not in POOL_MODES :
            raise ValueError(f"pool_mode must be one of {POOL_MODES}, got {self.pool_mode!r}")

        # Pooling kernel and stride are both positive integers.
        _check_positive_int(self.pooling_kernel, "pooling_kernel")
        _check_positive_int(self.pooling_stride, "pooling_stride")

        # Params: a dict of type-specific options.
        if not isinstance(self.params, dict) :
            raise ValueError(f"params must be a dict, got {self.params!r}")

        # Validate the type-specific params by round-tripping through the 1D pooling config family.
        # Only meaningful for an active 1D pooling operator (this also enforces the 'dwt' power-of-two decimation factor).
        if self.pooling_type in POOLING_CONFIG_REGISTRY :
            merged = {"pooling_type" : self.pooling_type, "decimation_factor" : self.pooling_stride, **self.params}
            get_pooling_config(merged)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# ShallowFBCSPNet config class

@dataclass
class ShallowFBCSPNet_config(model_config) :
    """
    Configuration for :class:`~jade.model.ShallowFBCSPNet.ShallowFBCSPNet` / :class:`~jade.model.ShallowFBCSPNet.ShallowFBCSPNet_Classifier`.

    The model is a shallow convolutional network: a temporal convolution, a spatial convolution that collapses the EEG-channel axis, a squaring nonlinearity, a pooling stage and a logarithm.
    The variable names respect the nomenclature of the reference ShallowFBCSPNet implementation where possible.

    Parameters
    ----------
    name : str, default 'shallowfbcspnet_classifier'
        Which model to build. One of ``'shallowfbcspnet'`` (feature extractor) or ``'shallowfbcspnet_classifier'`` (feature extractor + feedforward head).

    Convolutional front end
    -----------------------
    C : int, default 22
        Number of EEG channels of the input. It sets the height of the spatial convolution kernel ``(C, 1)``, which collapses the channel axis to 1.
    T : int, default 1000
        Number of time samples of the input.
    depth_first_layer : int, default 1
        Depth of the input tensor (``1`` for raw EEG, ``> 1`` for time-frequency input).
    n_filters_time : int, default 40
        Number of filters of the temporal convolution.
    filter_time_length : int, default 25
        Length (in time samples) of the temporal convolution kernel, i.e. the kernel is ``(1, filter_time_length)``.
    n_filters_spat : int, default 40
        Number of filters of the spatial convolution. Used only when ``split_first_layer`` is True.
    split_first_layer : bool, default True
        If True (the original behaviour) the first layer is split in a temporal convolution ``(1, filter_time_length)`` followed by a spatial convolution ``(C, 1)``, with no nonlinearity in between.
        If False a single convolution ``(C, filter_time_length)`` does both at once, and ``n_filters_spat`` is ignored.
    batch_norm : bool, default True
        If True a :class:`torch.nn.BatchNorm2d` follows the convolutional front end (and the convolution that precedes it drops its bias term, which the normalization would make redundant).
    batch_norm_alpha : float, default 0.1
        Momentum of the batch normalization, in ``[0, 1]``. Ignored when ``batch_norm`` is False.
    conv_nonlin : str, default 'square'
        Nonlinearity applied after the convolutional front end (before the pooling). ``'square'`` is the original choice (together with ``pool_nonlin = 'safe_log'`` it gives the log-variance feature of FBCSP).
        One of ``square``, ``safe_log``, ``identity``, ``relu``, ``leaky_relu``, ``elu``, ``selu``, ``gelu`` (case-insensitive).
    pool_nonlin : str, default 'safe_log'
        Nonlinearity applied after the pooling stage. ``'safe_log'`` is the original choice, a logarithm clamped away from zero. Same accepted values as ``conv_nonlin``.
    prob_dropout : float, default 0.5
        Dropout probability applied after the pooling nonlinearity, in ``[0, 1]``.
    use_dropout_2d : bool, default False
        If True use :class:`torch.nn.Dropout2d` (a whole feature map is dropped at once) instead of :class:`torch.nn.Dropout`.
    pooling : shallowfbcspnet_pooling_config, default shallowfbcspnet_pooling_config()
        The single pooling stage (the ``[pooling]`` section).
        ``pooling_type = 'default'`` reproduces the original overlapping average pooling; a 1D pooling operator swaps it for a :mod:`jade.model.pooling_1d` layer applied channel-wise.
        A dict is accepted and promoted to :class:`shallowfbcspnet_pooling_config`.

    Classification head
    -------------------
    n_classes : int, default 4
        Number of output classes of the classifier head.
    use_softmax : bool, default False
        If False (the default) the classifier head ends with :class:`torch.nn.Identity` (raw logits, pair with a cross-entropy loss);
        if True it ends with :class:`torch.nn.LogSoftmax` (log-probabilities, pair with an NLL loss).
        The default is False because raw logits combined with the (default) cross-entropy loss are numerically more stable for training; this is the recommended default for any classifier.
        See :func:`~jade.training.support_training.cross_check_config`.

    Misc
    ----
    flatten_output : bool, default False
        If True the (plain) :class:`~jade.model.ShallowFBCSPNet.ShallowFBCSPNet` feature extractor flattens its output to ``(batch, n_features)``. Ignored by the classifier (which always flattens internally).
    input_size : tuple of int, default (1, 1, 22, 1000)
        Shape ``(batch, depth, C, T)`` of a dummy input used by :class:`~jade.model.ShallowFBCSPNet.ShallowFBCSPNet_Classifier` to size the feedforward layer of its head.
    print_var : bool, default False
        If True print the parameter count (and, using ``input_size``, the shape tracking) at build time.
    """

    name : str = "shallowfbcspnet_classifier"

    # Convolutional front end
    C                  : int = 22
    T                  : int = 1000
    depth_first_layer  : int = 1
    n_filters_time     : int = 40
    filter_time_length : int = 25
    n_filters_spat     : int = 40
    split_first_layer  : bool = True
    batch_norm         : bool = True
    batch_norm_alpha   : float = 0.1
    conv_nonlin        : str = "square"
    pool_nonlin        : str = "safe_log"
    prob_dropout       : float = 0.5
    use_dropout_2d     : bool = False

    # Classification head
    n_classes   : int = 4
    use_softmax : bool = False

    # Misc
    flatten_output : bool = False
    input_size     : Tuple[int, int, int, int] = (1, 1, 22, 1000)
    print_var      : bool = False

    # Pooling (the [pooling] section). Declared last so it serializes after the scalar keys (a nested
    # config becomes a TOML sub-table, which must follow the parent's scalar keys).
    pooling        : shallowfbcspnet_pooling_config = field(default_factory = shallowfbcspnet_pooling_config)


    def __post_init__(self) -> None :
        """
        Validate (and normalize) every ShallowFBCSPNet parameter.
        """

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Discriminator and nonlinearities

        # The model name selects the class in jade.model.get_model, so it must be one of the known models.
        self.name = str(self.name).lower()
        if self.name not in ALLOWED_NAMES :
            raise ValueError(f"name must be one of {ALLOWED_NAMES}, got {self.name!r}")

        self.conv_nonlin = str(self.conv_nonlin).lower()
        if self.conv_nonlin not in NONLINEARITIES :
            raise ValueError(f"conv_nonlin must be one of {NONLINEARITIES}, got {self.conv_nonlin!r}")

        self.pool_nonlin = str(self.pool_nonlin).lower()
        if self.pool_nonlin not in NONLINEARITIES :
            raise ValueError(f"pool_nonlin must be one of {NONLINEARITIES}, got {self.pool_nonlin!r}")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Boolean flags

        _check_bool(self.split_first_layer, "split_first_layer")
        _check_bool(self.batch_norm, "batch_norm")
        _check_bool(self.use_dropout_2d, "use_dropout_2d")
        _check_bool(self.use_softmax, "use_softmax")
        _check_bool(self.flatten_output, "flatten_output")
        _check_bool(self.print_var, "print_var")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Positive integer sizes

        _check_positive_int(self.C, "C")
        _check_positive_int(self.T, "T")
        _check_positive_int(self.depth_first_layer, "depth_first_layer")
        _check_positive_int(self.n_filters_time, "n_filters_time")
        _check_positive_int(self.filter_time_length, "filter_time_length")
        _check_positive_int(self.n_filters_spat, "n_filters_spat")
        _check_positive_int(self.n_classes, "n_classes", minimum = 2)

        # The temporal convolution is not padded, so it must fit inside the input window.
        if self.filter_time_length > self.T :
            raise ValueError(f"filter_time_length ({self.filter_time_length}) must not exceed T ({self.T})")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Probabilities

        self.prob_dropout     = _check_probability(self.prob_dropout, "prob_dropout")
        self.batch_norm_alpha = _check_probability(self.batch_norm_alpha, "batch_norm_alpha")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Pooling section

        # Promote a dict to shallowfbcspnet_pooling_config (direct-construction path; from_dict already promotes it).
        if isinstance(self.pooling, dict) :
            self.pooling = shallowfbcspnet_pooling_config.from_dict(self.pooling)
        elif not isinstance(self.pooling, shallowfbcspnet_pooling_config) :
            raise ValueError(f"pooling must be a shallowfbcspnet_pooling_config (or a dict), got {type(self.pooling)}")

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
    def pooling_layer_config(self) :
        """
        Per-stage pooling config of the single pooling stage (delegates to :attr:`pooling`).

        A :class:`~jade.config.model.pooling_1d_cfg.pooling_config` dataclass for an active 1D operator, or a plain dict for the ShallowFBCSPNet-only ``'default'`` (overlapping pooling) and ``'identity'`` / ``'none'`` types.
        This is what :class:`~jade.model.ShallowFBCSPNet.ShallowFBCSPNet` passes to :func:`~jade.model.ShallowFBCSPNet.get_shallowfbcspnet_pooling`.
        """

        return self.pooling.pooling_layer_config
