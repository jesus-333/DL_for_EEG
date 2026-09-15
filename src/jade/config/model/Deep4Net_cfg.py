"""
Deep4Net (Deep ConvNet) model configuration.

:class:`~jade.config.model.Deep4Net_cfg.Deep4Net_config` describes every hyper-parameter needed to build a :class:`~jade.model.Deep4Net.Deep4Net` and its classifier wrapper :class:`~jade.model.Deep4Net.Deep4Net_Classifier`.
The two share a single config because the classifier is just a Deep4Net feature extractor plus a feedforward classification head, so the classifier keys are a superset of the plain ones.
Which of the two is actually built is decided by the ``name`` discriminator through :func:`~jade.model.get_model`.

The model is the one presented as ``Deep ConvNet`` in ``Deep learning with convolutional neural networks for EEG decoding and visualization`` (Schirrmeister et al., Human Brain Mapping, 2017, https://doi.org/10.1002/hbm.23730), better known through the ``Deep4Net`` name of its reference implementation.
The variable names (and the default values) follow that reference implementation where possible, so a bare ``Deep4Net_config()`` reproduces the original BCI IV-2a architecture (a split first convolution with 25 temporal and 25 spatial filters, then three convolution + pooling blocks with 50, 100 and 200 filters).

The config is torch-free (like the rest of :mod:`jade.config`); the actual layers are built by :class:`~jade.model.Deep4Net.Deep4Net`.
The network has **four** pooling stages, one per convolutional block, and each of them is configured independently in the ``[pooling]`` section: every stage can reproduce the original max pooling (``'default'``) or swap in one of the custom :mod:`jade.model.pooling_1d` operators applied channel-wise.
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
ALLOWED_NAMES = ("deep4net", "deep4net_classifier", "deep4netclassifier")

# Activation functions supported by jade.model.support_functions.get_activation.
ACTIVATIONS = ("relu", "leaky_relu", "elu", "selu", "gelu")

# Nonlinearities accepted after a pooling stage (the activations, plus the no-op used by the original model).
POOL_NONLINEARITIES = ("identity", "square", "safe_log") + ACTIVATIONS

# Pooling modes of a 'default' (original) pooling stage.
POOL_MODES = ("max", "mean")

# Suffixes of the four per-stage pooling keys (one stage per convolutional block).
POOLING_STAGES = ("1", "2", "3", "4")

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
# Deep4Net pooling config (the [pooling] section)

@dataclass
class deep4net_pooling_config(generic_config) :
    """
    Pooling configuration for the four pooling stages of Deep4Net (the ``[pooling]`` section).

    Deep4Net has one pooling stage per convolutional block: the first one closes the split temporal + spatial convolution, the other three close the three deeper convolution blocks.
    Each stage is configured independently (its own pooling type, decimation factor, pooling mode and type-specific parameters), exactly like the two stages of :class:`~jade.config.model.EEGNet_cfg.eegnet_pooling_config`, so the pooling ablation can also be run on a single stage at a time.

    Unlike ShallowFBCSPNet and the EEG Conformer, the original Deep4Net pooling is **non overlapping** (a window of 3 samples every 3), so a single decimation factor per stage is enough: the ``'default'`` pooling and the 1D operators produce the same output length.

    The four derived properties :attr:`pooling_layer_1_config` ... :attr:`pooling_layer_4_config` assemble the per-stage configs consumed by :class:`~jade.model.Deep4Net.Deep4Net` (through :func:`~jade.model.Deep4Net.get_deep4net_pooling`).

    Parameters
    ----------
    pooling_type_1 : str, default 'default'
        Pooling operator of the first stage. One of:
        - ``'default'`` : the original Deep4Net pooling, a non overlapping ``nn.MaxPool2d`` (or ``nn.AvgPool2d``, see ``pool_mode_1``) with kernel ``(1, pooling_factor_1)`` (pooling along the time axis only);
        - a 1D pooling name: ``'avg'``, ``'max'``, ``'blur'``, ``'sinc'``, ``'dwt'`` (and their aliases), applied channel-wise through the ``*_multichannel`` wrappers of :mod:`jade.model.pooling_1d`;
        - ``'identity'`` / ``'none'`` : no pooling (the stage is replaced by ``nn.Identity``).
        Case-insensitive. Any other value raises ``ValueError``.
    pooling_type_2, pooling_type_3, pooling_type_4 : str, default 'default'
        Same as ``pooling_type_1`` but for the second, third and fourth stage.
    pooling_factor_1 : int, default 3
        Decimation factor ``M`` of the first stage: the number of time samples becomes ``T / M``.
        Must be a non-zero integer. A negative value disables that pooling stage (a warning is emitted at instantiation).
        Note that the ``'dwt'`` operator additionally requires this to be a power of two, so the original factor 3 is not usable with it (use 2 or 4).
    pooling_factor_2, pooling_factor_3, pooling_factor_4 : int, default 3
        Same as ``pooling_factor_1`` but for the second, third and fourth stage.
    pool_mode_1 : str, default 'max'
        Which pooling the ``'default'`` type builds for the first stage, ``'max'`` (``nn.MaxPool2d``, the original) or ``'mean'`` (``nn.AvgPool2d``). Ignored by the 1D operators (there the operator itself is the choice).
    pool_mode_2, pool_mode_3, pool_mode_4 : str, default 'max'
        Same as ``pool_mode_1`` but for the second, third and fourth stage.
    params_1 : dict, default {}
        Extra, type-specific parameters for the first stage (e.g. ``{'wavelet' : 'sym4'}`` for ``'dwt'``, ``{'cutoff_scale' : 0.8}`` for ``'sinc'``).
        Empty means "use the pooling layer defaults". See :mod:`jade.model.pooling_1d` for the accepted keys of each operator. Ignored by the ``'default'`` / ``'identity'`` types.
    params_2, params_3, params_4 : dict, default {}
        Same as ``params_1`` but for the second, third and fourth stage.
    """

    pooling_type_1   : str  = "default"
    pooling_type_2   : str  = "default"
    pooling_type_3   : str  = "default"
    pooling_type_4   : str  = "default"
    pooling_factor_1 : int  = 3
    pooling_factor_2 : int  = 3
    pooling_factor_3 : int  = 3
    pooling_factor_4 : int  = 3
    pool_mode_1      : str  = "max"
    pool_mode_2      : str  = "max"
    pool_mode_3      : str  = "max"
    pool_mode_4      : str  = "max"
    params_1         : dict = field(default_factory = dict)
    params_2         : dict = field(default_factory = dict)
    params_3         : dict = field(default_factory = dict)
    params_4         : dict = field(default_factory = dict)


    @property
    def pooling_layer_1_config(self) :
        """
        Per-stage pooling config for the first pooling stage (see :meth:`_stage_pooling_config`).
        """

        return self._stage_pooling_config(self.pooling_type_1, self.pooling_factor_1, self.pool_mode_1, self.params_1)


    @property
    def pooling_layer_2_config(self) :
        """
        Per-stage pooling config for the second pooling stage (see :meth:`_stage_pooling_config`).
        """

        return self._stage_pooling_config(self.pooling_type_2, self.pooling_factor_2, self.pool_mode_2, self.params_2)


    @property
    def pooling_layer_3_config(self) :
        """
        Per-stage pooling config for the third pooling stage (see :meth:`_stage_pooling_config`).
        """

        return self._stage_pooling_config(self.pooling_type_3, self.pooling_factor_3, self.pool_mode_3, self.params_3)


    @property
    def pooling_layer_4_config(self) :
        """
        Per-stage pooling config for the fourth pooling stage (see :meth:`_stage_pooling_config`).
        """

        return self._stage_pooling_config(self.pooling_type_4, self.pooling_factor_4, self.pool_mode_4, self.params_4)


    def _stage_pooling_config(self, pooling_type : str, factor : int, pool_mode : str, params : dict) :
        """
        Assemble the pooling configuration of a single stage.

        Parameters
        ----------
        pooling_type : str
            Pooling operator of the stage.
        factor : int
            Decimation factor of the stage (negative disables it).
        pool_mode : str
            Pooling mode used by the ``'default'`` type.
        params : dict
            Type-specific parameters of the stage.

        Returns
        -------
        pooling_config or dict
            The per-stage pooling configuration (a dataclass for an active 1D operator, a dict otherwise).
        """

        # Lazy import to avoid an import cycle (see __post_init__).
        from jade.config.model.pooling_1d_cfg import POOLING_CONFIG_REGISTRY, get_pooling_config

        # A real, active 1D operator has a config dataclass -> return it as a dataclass.
        # 'default' / 'identity' / 'none' and disabled (factor < 1) stages have none -> keep the plain dict.
        if factor >= 1 and pooling_type in POOLING_CONFIG_REGISTRY :
            return get_pooling_config({"pooling_type" : pooling_type, "decimation_factor" : factor, **params})

        return {"pooling_type" : pooling_type, "decimation_factor" : factor, "pool_mode" : pool_mode, **params}


    def __post_init__(self) -> None :
        """
        Validate (and normalize) the pooling configuration of the four stages.
        """

        # Lazy import to avoid an import cycle: this module is imported while jade.config.model is still initializing.
        from jade.config.model.pooling_1d_cfg import POOLING_CONFIG_REGISTRY, get_pooling_config

        # Valid pooling_type values: the 1D pooling names, plus 'default' (MaxPool2d/AvgPool2d) and the explicit no-op aliases.
        valid_types = set(POOLING_CONFIG_REGISTRY) | {"default", "identity", "none"}

        for suffix in POOLING_STAGES :
            type_attr   = f"pooling_type_{suffix}"
            factor_attr = f"pooling_factor_{suffix}"
            mode_attr   = f"pool_mode_{suffix}"
            params_attr = f"params_{suffix}"

            # Pooling type (lower-cased and validated).
            pooling_type = str(getattr(self, type_attr)).lower()
            setattr(self, type_attr, pooling_type)
            if pooling_type not in valid_types :
                raise ValueError(f"{type_attr} must be one of {sorted(valid_types)}, got {pooling_type!r}")

            # Pooling mode of the 'default' stage.
            pool_mode = str(getattr(self, mode_attr)).lower()
            setattr(self, mode_attr, pool_mode)
            if pool_mode not in POOL_MODES :
                raise ValueError(f"{mode_attr} must be one of {POOL_MODES}, got {pool_mode!r}")

            # Pooling factor: a non-zero integer. Negative disables the stage (documented behavior).
            factor = getattr(self, factor_attr)
            if not _is_int(factor) or factor == 0 :
                raise ValueError(f"{factor_attr} must be a non-zero integer, got {factor!r}")
            if factor < 0 :
                warnings.warn(
                    f"{factor_attr} = {factor} is negative: Deep4Net pooling stage {suffix} is disabled "
                    "(replaced by nn.Identity).",
                    stacklevel = 2,
                )

            # Params: a dict of type-specific options.
            params = getattr(self, params_attr)
            if not isinstance(params, dict) :
                raise ValueError(f"{params_attr} must be a dict, got {params!r}")

            # Validate the type-specific params by round-tripping through the 1D pooling config family.
            # Only meaningful for an active 1D pooling operator (this also enforces the 'dwt' power-of-two decimation factor).
            if factor >= 1 and pooling_type in POOLING_CONFIG_REGISTRY :
                get_pooling_config({"pooling_type" : pooling_type, "decimation_factor" : factor, **params})

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Deep4Net config class

@dataclass
class Deep4Net_config(model_config) :
    """
    Configuration for :class:`~jade.model.Deep4Net.Deep4Net` / :class:`~jade.model.Deep4Net.Deep4Net_Classifier`.

    The model is a deep convolutional network: a split temporal + spatial convolution closed by a pooling stage, followed by three convolution + pooling blocks with a growing number of filters.
    The variable names respect the nomenclature of the reference Deep4Net implementation where possible.

    Parameters
    ----------
    name : str, default 'deep4net_classifier'
        Which model to build. One of ``'deep4net'`` (feature extractor) or ``'deep4net_classifier'`` (feature extractor + feedforward head).

    First block (split temporal + spatial convolution)
    --------------------------------------------------
    C : int, default 22
        Number of EEG channels of the input. It sets the height of the spatial convolution kernel ``(C, 1)``, which collapses the channel axis to 1.
    T : int, default 1000
        Number of time samples of the input.
    depth_first_layer : int, default 1
        Depth of the input tensor (``1`` for raw EEG, ``> 1`` for time-frequency input).
    n_filters_time : int, default 25
        Number of filters of the temporal convolution of the first block.
    filter_time_length : int, default 10
        Length (in time samples) of the temporal convolution kernel of the first block, i.e. the kernel is ``(1, filter_time_length)``.
    n_filters_spat : int, default 25
        Number of filters of the spatial convolution of the first block. Used only when ``split_first_layer`` is True.
    split_first_layer : bool, default True
        If True (the original behaviour) the first layer is split in a temporal convolution ``(1, filter_time_length)`` followed by a spatial convolution ``(C, 1)``, with no nonlinearity in between.
        If False a single convolution ``(C, filter_time_length)`` does both at once, and ``n_filters_spat`` is ignored.

    Deeper blocks
    -------------
    n_filters_2 : int, default 50
        Number of filters of the convolution of the second block.
    filter_length_2 : int, default 10
        Length (in time samples) of the convolution kernel of the second block.
    n_filters_3 : int, default 100
        Number of filters of the convolution of the third block.
    filter_length_3 : int, default 10
        Length (in time samples) of the convolution kernel of the third block.
    n_filters_4 : int, default 200
        Number of filters of the convolution of the fourth block.
    filter_length_4 : int, default 10
        Length (in time samples) of the convolution kernel of the fourth block.

    Shared block options
    --------------------
    activation : str, default 'elu'
        Activation function applied after every convolution. One of ``relu``, ``leaky_relu``, ``elu``, ``selu``, ``gelu`` (case-insensitive).
    pool_nonlin : str, default 'identity'
        Nonlinearity applied after every pooling stage. ``'identity'`` is the original choice (Deep4Net has no post-pooling nonlinearity).
        One of ``identity``, ``square``, ``safe_log``, or any activation name.
    batch_norm : bool, default True
        If True a :class:`torch.nn.BatchNorm2d` follows every convolution (and the convolution drops its bias term, which the normalization would make redundant).
    batch_norm_alpha : float, default 0.1
        Momentum of the batch normalization, in ``[0, 1]``. Ignored when ``batch_norm`` is False.
    prob_dropout : float, default 0.5
        Dropout probability applied at the beginning of the second, third and fourth block, in ``[0, 1]``.
    use_dropout_2d : bool, default False
        If True use :class:`torch.nn.Dropout2d` (a whole feature map is dropped at once) instead of :class:`torch.nn.Dropout`.
    pooling : deep4net_pooling_config, default deep4net_pooling_config()
        The four pooling stages, one per convolutional block (the ``[pooling]`` section).
        ``pooling_type_N = 'default'`` reproduces the original max pooling; a 1D pooling operator swaps it for a :mod:`jade.model.pooling_1d` layer applied channel-wise.
        A dict is accepted and promoted to :class:`deep4net_pooling_config`.

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
        If True the (plain) :class:`~jade.model.Deep4Net.Deep4Net` feature extractor flattens its output to ``(batch, n_features)``. Ignored by the classifier (which always flattens internally).
    input_size : tuple of int, default (1, 1, 22, 1000)
        Shape ``(batch, depth, C, T)`` of a dummy input used by :class:`~jade.model.Deep4Net.Deep4Net_Classifier` to size the feedforward layer of its head.
    print_var : bool, default False
        If True print the parameter count (and, using ``input_size``, the shape tracking) at build time.
    """

    name : str = "deep4net_classifier"

    # First block (split temporal + spatial convolution)
    C                  : int = 22
    T                  : int = 1000
    depth_first_layer  : int = 1
    n_filters_time     : int = 25
    filter_time_length : int = 10
    n_filters_spat     : int = 25
    split_first_layer  : bool = True

    # Deeper blocks
    n_filters_2     : int = 50
    filter_length_2 : int = 10
    n_filters_3     : int = 100
    filter_length_3 : int = 10
    n_filters_4     : int = 200
    filter_length_4 : int = 10

    # Shared block options
    activation       : str = "elu"
    pool_nonlin      : str = "identity"
    batch_norm       : bool = True
    batch_norm_alpha : float = 0.1
    prob_dropout     : float = 0.5
    use_dropout_2d   : bool = False

    # Classification head
    n_classes   : int = 4
    use_softmax : bool = False

    # Misc
    flatten_output : bool = False
    input_size     : Tuple[int, int, int, int] = (1, 1, 22, 1000)
    print_var      : bool = False

    # Pooling (the [pooling] section). Declared last so it serializes after the scalar keys (a nested
    # config becomes a TOML sub-table, which must follow the parent's scalar keys).
    pooling        : deep4net_pooling_config = field(default_factory = deep4net_pooling_config)


    def __post_init__(self) -> None :
        """
        Validate (and normalize) every Deep4Net parameter.
        """

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Discriminator and nonlinearities

        # The model name selects the class in jade.model.get_model, so it must be one of the known models.
        self.name = str(self.name).lower()
        if self.name not in ALLOWED_NAMES :
            raise ValueError(f"name must be one of {ALLOWED_NAMES}, got {self.name!r}")

        self.activation = str(self.activation).lower()
        if self.activation not in ACTIVATIONS :
            raise ValueError(f"activation must be one of {ACTIVATIONS}, got {self.activation!r}")

        self.pool_nonlin = str(self.pool_nonlin).lower()
        if self.pool_nonlin not in POOL_NONLINEARITIES :
            raise ValueError(f"pool_nonlin must be one of {POOL_NONLINEARITIES}, got {self.pool_nonlin!r}")

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
        _check_positive_int(self.n_filters_2, "n_filters_2")
        _check_positive_int(self.filter_length_2, "filter_length_2")
        _check_positive_int(self.n_filters_3, "n_filters_3")
        _check_positive_int(self.filter_length_3, "filter_length_3")
        _check_positive_int(self.n_filters_4, "n_filters_4")
        _check_positive_int(self.filter_length_4, "filter_length_4")
        _check_positive_int(self.n_classes, "n_classes", minimum = 2)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Probabilities

        self.prob_dropout     = _check_probability(self.prob_dropout, "prob_dropout")
        self.batch_norm_alpha = _check_probability(self.batch_norm_alpha, "batch_norm_alpha")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Pooling section

        # Promote a dict to deep4net_pooling_config (direct-construction path; from_dict already promotes it).
        if isinstance(self.pooling, dict) :
            self.pooling = deep4net_pooling_config.from_dict(self.pooling)
        elif not isinstance(self.pooling, deep4net_pooling_config) :
            raise ValueError(f"pooling must be a deep4net_pooling_config (or a dict), got {type(self.pooling)}")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Input size (batch, depth, channels, time)

        # Normalize a list coming from a file back into a tuple.
        self.input_size = tuple(self.input_size)
        if len(self.input_size) != 4 :
            raise ValueError(f"input_size must be a 4-tuple (batch, depth, channels, time), got {self.input_size!r}")
        for element in self.input_size :
            if not _is_int(element) or element <= 0 :
                raise ValueError(f"input_size must contain positive integers, got {self.input_size!r}")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Time axis depth

        # The four blocks shrink the time axis (unpadded convolutions and pooling), so a too short input
        # collapses it to zero somewhere in the middle of the network. Fail here with a clear message
        # instead of letting torch raise a cryptic shape error at the first forward pass.
        n_times = self.input_size[3]
        for filter_length, factor in zip(
                (self.filter_time_length, self.filter_length_2, self.filter_length_3, self.filter_length_4),
                (self.pooling.pooling_factor_1, self.pooling.pooling_factor_2, self.pooling.pooling_factor_3, self.pooling.pooling_factor_4),
        ) :
            n_times = n_times - filter_length + 1
            if factor >= 1 : n_times = n_times // factor
            if n_times < 1 :
                raise ValueError(f"the time axis collapses inside the network: input_size time samples ({self.input_size[3]}) are too few for the current kernel lengths and pooling factors")
