"""
TIDNet (Thinker Invariance DenseNet) model configuration.

:class:`~jade.config.model.TIDNet_cfg.TIDNet_config` describes every hyper-parameter needed to build a :class:`~jade.model.TIDNet.TIDNet` and its classifier wrapper :class:`~jade.model.TIDNet.TIDNet_Classifier`.
The two share a single config because the classifier is just a TIDNet feature extractor plus a feedforward classification head, so the classifier keys are a superset of the plain ones.
Which of the two is actually built is decided by the ``name`` discriminator through :func:`~jade.model.get_model`.

The model is the one presented in ``Thinker invariance: enabling deep neural networks for BCI across more people`` (Kostas & Rudzicz, Journal of Neural Engineering, 2020, https://doi.org/10.1088/1741-2552/abb7a7).
The variable names (and the default values) follow the authors' reference implementation where possible, so a bare ``TIDNet_config()`` reproduces the original architecture (a residual dilated temporal filter with 32 filters, followed by a densely connected spatial filter with growth 24 and an adaptive summary of the time axis).

The config is torch-free (like the rest of :mod:`jade.config`); the actual layers are built by :class:`~jade.model.TIDNet.TIDNet`.
The network has a single decimating pooling stage (the ``[pooling]`` section, which closes the temporal block), which can reproduce the original max pooling (``'default'``) or swap in one of the custom :mod:`jade.model.pooling_1d` operators applied channel-wise.
The adaptive average pooling that summarizes the time axis before the head is **not** part of that section: it is an adaptive resize, not a decimation, so it has no anti-aliasing counterpart in :mod:`jade.model.pooling_1d`.

Note that, unlike the original implementation, this version has no auxiliary subject / run classification heads: :class:`~jade.model.TIDNet.TIDNet_Classifier` returns a plain tensor like every other ``jade`` model, so it plugs into :mod:`jade.training` unchanged.
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
ALLOWED_NAMES = ("tidnet", "tidnet_classifier", "tidnetclassifier")

# Activation functions supported by jade.model.support_functions.get_activation.
ACTIVATIONS = ("relu", "leaky_relu", "elu", "selu", "gelu")

# Pooling modes of the 'default' (original) pooling stage.
POOL_MODES = ("max", "mean")

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
# TIDNet pooling config (the [pooling] section)

@dataclass
class tidnet_pooling_config(generic_config) :
    """
    Pooling configuration for the single decimating pooling stage of TIDNet (the ``[pooling]`` section).

    The stage closes the temporal block: it follows the residual dilated temporal filter and decimates the time axis before the densely connected spatial filter.
    In the original model it is a non overlapping ``nn.MaxPool2d`` of 15 samples, so a single decimation factor is enough (the ``'default'`` pooling and the 1D operators produce the same output length).

    The decimation factor also drives the default summary length of the adaptive pooling that closes the feature extractor (``summary = -1`` means ``T / M``), so changing it changes the size of the feature vector.

    The derived property :attr:`pooling_layer_config` assembles the per-stage config consumed by :class:`~jade.model.TIDNet.TIDNet` (through :func:`~jade.model.TIDNet.get_tidnet_pooling`).

    Parameters
    ----------
    pooling_type : str, default 'default'
        Pooling operator of the stage. One of:
        - ``'default'`` : the original TIDNet pooling, a non overlapping ``nn.MaxPool2d`` (or ``nn.AvgPool2d``, see ``pool_mode``) with kernel ``(1, pooling_factor)`` (pooling along the time axis only);
        - a 1D pooling name: ``'avg'``, ``'max'``, ``'blur'``, ``'sinc'``, ``'dwt'`` (and their aliases), applied channel-wise through the ``*_multichannel`` wrappers of :mod:`jade.model.pooling_1d`;
        - ``'identity'`` / ``'none'`` : no pooling (the stage is replaced by ``nn.Identity``).
        Case-insensitive. Any other value raises ``ValueError``.
    pooling_factor : int, default 15
        Decimation factor ``M`` of the stage: the number of time samples becomes ``T / M``. Must be a positive integer.
        Note that the ``'dwt'`` operator additionally requires this to be a power of two, so the original factor 15 is not usable with it (use 16, or another power of two).
    pool_mode : str, default 'max'
        Which pooling the ``'default'`` type builds, ``'max'`` (``nn.MaxPool2d``, the original) or ``'mean'`` (``nn.AvgPool2d``). Ignored by the 1D operators (there the operator itself is the choice).
    params : dict, default {}
        Extra, type-specific parameters for a 1D operator (e.g. ``{'wavelet' : 'sym4'}`` for ``'dwt'``, ``{'cutoff_scale' : 0.8}`` for ``'sinc'``).
        Empty means "use the pooling layer defaults". See :mod:`jade.model.pooling_1d` for the accepted keys of each operator. Ignored by the ``'default'`` / ``'identity'`` types.
    """

    pooling_type   : str  = "default"
    pooling_factor : int  = 15
    pool_mode      : str  = "max"
    params         : dict = field(default_factory = dict)


    @property
    def pooling_layer_config(self) :
        """
        Per-stage pooling config consumed by :class:`~jade.model.TIDNet.TIDNet`.

        For an active 1D pooling operator it is a :class:`~jade.config.model.pooling_1d_cfg.pooling_config` dataclass (decimation factor ``pooling_factor`` plus the type-specific ``params``), keeping the whole flow dataclass-based;
        for the TIDNet-only ``'default'`` (max / average pooling) and ``'identity'`` / ``'none'`` types, which have no config dataclass, it is a plain ``{pooling_type, decimation_factor, pool_mode, **params}`` dict.
        Consumed by :func:`jade.model.TIDNet.get_tidnet_pooling`, which accepts either.

        Returns
        -------
        pooling_config or dict
            The pooling configuration (a dataclass for an active 1D operator, a dict otherwise).
        """

        # Lazy import to avoid an import cycle (see __post_init__).
        from jade.config.model.pooling_1d_cfg import POOLING_CONFIG_REGISTRY, get_pooling_config

        # A real, active 1D operator has a config dataclass -> return it as a dataclass.
        # 'default' / 'identity' / 'none' have none -> keep the plain dict.
        if self.pooling_type in POOLING_CONFIG_REGISTRY :
            merged = {"pooling_type" : self.pooling_type, "decimation_factor" : self.pooling_factor, **self.params}
            return get_pooling_config(merged)

        return {"pooling_type" : self.pooling_type, "decimation_factor" : self.pooling_factor, "pool_mode" : self.pool_mode, **self.params}


    def __post_init__(self) -> None :
        """
        Validate (and normalize) the pooling configuration.
        """

        # Lazy import to avoid an import cycle: this module is imported while jade.config.model is still initializing.
        from jade.config.model.pooling_1d_cfg import POOLING_CONFIG_REGISTRY, get_pooling_config

        # Valid pooling_type values: the 1D pooling names, plus 'default' (MaxPool2d/AvgPool2d) and the explicit no-op aliases.
        valid_types = set(POOLING_CONFIG_REGISTRY) | {"default", "identity", "none"}

        # Pooling type (lower-cased and validated).
        self.pooling_type = str(self.pooling_type).lower()
        if self.pooling_type not in valid_types :
            raise ValueError(f"pooling_type must be one of {sorted(valid_types)}, got {self.pooling_type!r}")

        # Pooling mode of the 'default' stage.
        self.pool_mode = str(self.pool_mode).lower()
        if self.pool_mode not in POOL_MODES :
            raise ValueError(f"pool_mode must be one of {POOL_MODES}, got {self.pool_mode!r}")

        # The decimation factor also sets the default summary length, so (unlike EEGNet) a disabled stage
        # is asked for explicitly with pooling_type = 'identity', not with a negative factor.
        _check_positive_int(self.pooling_factor, "pooling_factor")

        # Params: a dict of type-specific options.
        if not isinstance(self.params, dict) :
            raise ValueError(f"params must be a dict, got {self.params!r}")

        # Validate the type-specific params by round-tripping through the 1D pooling config family.
        # Only meaningful for an active 1D pooling operator (this also enforces the 'dwt' power-of-two decimation factor).
        if self.pooling_type in POOLING_CONFIG_REGISTRY :
            merged = {"pooling_type" : self.pooling_type, "decimation_factor" : self.pooling_factor, **self.params}
            get_pooling_config(merged)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# TIDNet config class

@dataclass
class TIDNet_config(model_config) :
    """
    Configuration for :class:`~jade.model.TIDNet.TIDNet` / :class:`~jade.model.TIDNet.TIDNet_Classifier`.

    The model chains a residual dilated temporal filter, a decimating pooling stage, a densely connected spatial filter that collapses the EEG-channel axis, and an adaptive summary of the time axis.
    The variable names respect the nomenclature of the original TIDNet implementation where possible.

    Parameters
    ----------
    name : str, default 'tidnet_classifier'
        Which model to build. One of ``'tidnet'`` (feature extractor) or ``'tidnet_classifier'`` (feature extractor + feedforward head).

    Input geometry
    --------------
    C : int, default 22
        Number of EEG channels of the input. It sets the height of the channel-collapse convolution kernel ``(C, 1)`` of the spatial filter.
    T : int, default 1000
        Number of time samples of the input. It also sets the temporal kernel length, which is ``ceil(temp_span * T)``.
    depth_first_layer : int, default 1
        Depth of the input tensor (``1`` for raw EEG, ``> 1`` for time-frequency input).

    Temporal filter
    ---------------
    t_filters : int, default 32
        Number of filters of the residual dilated temporal filter.
    temp_layers : int, default 2
        Number of stacked dilated convolutions of the temporal filter. The dilation of the ``i``-th one is ``temp_layers - i``, so the first one is the most dilated.
    temp_span : float, default 0.05
        Length of the temporal kernel as a fraction of ``T``: the kernel covers ``ceil(temp_span * T)`` samples (rounded up to the next odd value, so the convolution preserves the time axis). Must be in ``(0, 1]``.

    Spatial filter
    --------------
    s_growth : int, default 24
        Growth rate of the densely connected spatial filter: every dense layer appends ``s_growth`` feature maps to its input.
    spat_layers : int, default 2
        Number of dense layers of the spatial filter. The feature extractor therefore outputs ``t_filters + s_growth * spat_layers`` feature maps.
    bottleneck : int, default 3
        Bottleneck factor of a dense layer: its internal ``1x1`` convolution expands to ``bottleneck * s_growth`` feature maps before the spatial convolution reduces them back to ``s_growth``.
    summary : int, default -1
        Length the time axis is resized to by the adaptive average pooling that closes the feature extractor.
        ``-1`` (the default) means "keep the pooled length", i.e. ``T / pooling_factor``. Any other value must be a positive integer.

    Shared options
    --------------
    activation : str, default 'leaky_relu'
        Activation function used in every block. One of ``relu``, ``leaky_relu``, ``elu``, ``selu``, ``gelu`` (case-insensitive). The original model uses ``leaky_relu``.
    prob_dropout : float, default 0.4
        Dropout probability used in the temporal and spatial blocks, in ``[0, 1]``.
    pooling : tidnet_pooling_config, default tidnet_pooling_config()
        The single decimating pooling stage that closes the temporal block (the ``[pooling]`` section).
        ``pooling_type = 'default'`` reproduces the original max pooling; a 1D pooling operator swaps it for a :mod:`jade.model.pooling_1d` layer applied channel-wise.
        A dict is accepted and promoted to :class:`tidnet_pooling_config`.

    Classification head
    -------------------
    n_classes : int, default 4
        Number of output classes of the classifier head.
    use_softmax : bool, default False
        If False (the default) the classifier head ends with :class:`torch.nn.Identity` (raw logits, pair with a cross-entropy loss);
        if True it ends with :class:`torch.nn.LogSoftmax` (log-probabilities, pair with an NLL loss).
        The default is False because raw logits combined with the (default) cross-entropy loss are numerically more stable for training; this is the recommended default for any classifier.
        Note that the original TIDNet always ends with a log-softmax, so ``use_softmax = True`` is the faithful setting.
        See :func:`~jade.training.support_training.cross_check_config`.

    Misc
    ----
    flatten_output : bool, default False
        If True the (plain) :class:`~jade.model.TIDNet.TIDNet` feature extractor flattens its output to ``(batch, n_features)``. Ignored by the classifier (which always flattens internally).
    input_size : tuple of int, default (1, 1, 22, 1000)
        Shape ``(batch, depth, C, T)`` of a dummy input used by :class:`~jade.model.TIDNet.TIDNet_Classifier` to size the feedforward layer of its head.
    print_var : bool, default False
        If True print the parameter count (and, using ``input_size``, the shape tracking) at build time.
    """

    name : str = "tidnet_classifier"

    # Input geometry
    C                 : int = 22
    T                 : int = 1000
    depth_first_layer : int = 1

    # Temporal filter
    t_filters   : int = 32
    temp_layers : int = 2
    temp_span   : float = 0.05

    # Spatial filter
    s_growth    : int = 24
    spat_layers : int = 2
    bottleneck  : int = 3
    summary     : int = -1

    # Shared options
    activation   : str = "leaky_relu"
    prob_dropout : float = 0.4

    # Classification head
    n_classes   : int = 4
    use_softmax : bool = False

    # Misc
    flatten_output : bool = False
    input_size     : Tuple[int, int, int, int] = (1, 1, 22, 1000)
    print_var      : bool = False

    # Pooling (the [pooling] section). Declared last so it serializes after the scalar keys (a nested
    # config becomes a TOML sub-table, which must follow the parent's scalar keys).
    pooling        : tidnet_pooling_config = field(default_factory = tidnet_pooling_config)


    def __post_init__(self) -> None :
        """
        Validate (and normalize) every TIDNet parameter.
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

        _check_bool(self.use_softmax, "use_softmax")
        _check_bool(self.flatten_output, "flatten_output")
        _check_bool(self.print_var, "print_var")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Positive integer sizes

        _check_positive_int(self.C, "C")
        _check_positive_int(self.T, "T")
        _check_positive_int(self.depth_first_layer, "depth_first_layer")
        _check_positive_int(self.t_filters, "t_filters")
        _check_positive_int(self.temp_layers, "temp_layers")
        _check_positive_int(self.s_growth, "s_growth")
        _check_positive_int(self.spat_layers, "spat_layers")
        _check_positive_int(self.bottleneck, "bottleneck")
        _check_positive_int(self.n_classes, "n_classes", minimum = 2)

        # summary is either the -1 sentinel ("keep the pooled length") or an explicit positive length.
        if self.summary != -1 :
            _check_positive_int(self.summary, "summary")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Probabilities

        self.prob_dropout = _check_probability(self.prob_dropout, "prob_dropout")

        # temp_span is a fraction of the input window, so it lives in (0, 1] (0 would give an empty kernel).
        self.temp_span = _check_probability(self.temp_span, "temp_span")
        if self.temp_span <= 0.0 :
            raise ValueError(f"temp_span must be a float in (0, 1], got {self.temp_span!r}")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Pooling section

        # Promote a dict to tidnet_pooling_config (direct-construction path; from_dict already promotes it).
        if isinstance(self.pooling, dict) :
            self.pooling = tidnet_pooling_config.from_dict(self.pooling)
        elif not isinstance(self.pooling, tidnet_pooling_config) :
            raise ValueError(f"pooling must be a tidnet_pooling_config (or a dict), got {type(self.pooling)}")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Input size (batch, depth, channels, time)

        # Normalize a list coming from a file back into a tuple.
        self.input_size = tuple(self.input_size)
        if len(self.input_size) != 4 :
            raise ValueError(f"input_size must be a 4-tuple (batch, depth, channels, time), got {self.input_size!r}")
        for element in self.input_size :
            if not _is_int(element) or element <= 0 :
                raise ValueError(f"input_size must contain positive integers, got {self.input_size!r}")

        # The pooling stage must leave at least one time sample (the temporal convolutions are padded, so
        # the pooling is the only step that shrinks the time axis).
        if self.input_size[3] < self.pooling.pooling_factor :
            raise ValueError(f"input_size time samples ({self.input_size[3]}) must be at least the pooling factor ({self.pooling.pooling_factor})")


    @property
    def pooling_layer_config(self) :
        """
        Pooling config of the decimating stage (delegates to :attr:`pooling`).

        A :class:`~jade.config.model.pooling_1d_cfg.pooling_config` dataclass for an active 1D operator, or a plain dict for the TIDNet-only ``'default'`` and ``'identity'`` / ``'none'`` types.
        This is what :class:`~jade.model.TIDNet.TIDNet` passes to :func:`~jade.model.TIDNet.get_tidnet_pooling`.
        """

        return self.pooling.pooling_layer_config
