"""
EEG Conformer model configuration.

:class:`~jade.config.model.EEGConformer_cfg.EEGConformer_config` describes every hyper-parameter needed to build an :class:`~jade.model.EEGConformer.EEGConformer` and its classifier wrapper :class:`~jade.model.EEGConformer.EEGConformer_Classifier`.
The two share a single config because the classifier is just an EEG Conformer feature extractor plus a feedforward classification head, so the classifier keys are a superset of the plain ones.
Which of the two is actually built is decided by the ``name`` discriminator through :func:`~jade.model.get_model`.

The model is the one presented in ``EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization`` (Song et al., IEEE TNSRE 2023).
The variable names (and the default values) follow the authors' reference implementation where possible, so a bare ``EEGConformer_config()`` reproduces the original BCI IV-2a architecture (a convolutional patch embedding with 40 temporal filters, a 6-layer transformer encoder with 10 attention heads and embedding size 40, and the flatten + MLP classification head).
The default values purpose is to give a working skeleton that is easy to tweak, not a tuned model.

The config is torch-free (like the rest of :mod:`jade.config`); the actual layers are built by :class:`~jade.model.EEGConformer.EEGConformer`.
The convolutional patch embedding has a single pooling stage (the ``[pooling]`` section), which can reproduce the original overlapping average pooling (``'default'``) or swap in one of the custom :mod:`jade.model.pooling_1d` operators applied channel-wise.
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
ALLOWED_NAMES = ("eegconformer", "eegconformer_classifier", "eegconformerclassifier")

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


def _check_probability(value, field_name : str) -> float :
    """
    Raize ``ValueError`` if ``value`` is not a number in ``[0, 1]``, otherwise return it as a ``float``.

    ``bool`` is excluded explicitly (it subclasses ``int``, so a flag would otherwise slip through as 0/1).
    """

    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= value <= 1.0 :
        raise ValueError(f"{field_name} must be a float in [0, 1], got {value!r}")

    return float(value)


def _coerce_kernel(value, field_name : str) -> Tuple[int, int] :
    """
    Validate (and normalize) a 2D kernel size.

    A TOML/JSON round-trip turns a tuple into a list, so the value is coerced back to a tuple of two positive integers.

    Returns
    -------
    tuple
        The normalized ``(height, width)`` tuple.
    """

    # Normalize a list coming from a file back into a tuple.
    value = tuple(value)
    if len(value) != 2 :
        raise ValueError(f"{field_name} must be a pair (height, width), got {value!r}")
    for element in value :
        if not _is_int(element) or element <= 0 :
            raise ValueError(f"{field_name} must contain positive integers, got {value!r}")

    return value

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# EEG Conformer pooling config (the [pooling] section)

@dataclass
class conformer_pooling_config(generic_config) :
    """
    Pooling configuration for the single pooling stage of the EEG Conformer patch embedding (the ``[pooling]`` section).

    In the convolutional patch embedding a pooling layer follows the temporal + spatial convolutions and (as in the original paper) "acts as slicing to obtain the patches along the time dimension", exactly like the patch splitting of a Vision Transformer.
    This stage can either reproduce the original **overlapping** average pooling or swap in one of the anti-aliased 1D operators of :mod:`jade.model.pooling_1d`, applied channel-wise.

    The derived property :attr:`pooling_layer_config` assembles the per-stage config consumed by :class:`~jade.model.EEGConformer.EEGConformer` (through :func:`~jade.model.EEGConformer.get_conformer_pooling`).

    Parameters
    ----------
    pooling_type : str, default 'default'
        Pooling operator of the stage. One of:
        - ``'default'`` : the original EEG Conformer pooling, an **overlapping** ``nn.AvgPool2d`` with kernel ``(1, pooling_kernel)`` and stride ``(1, pooling_stride)`` (average pooling along the time axis only);
        - a 1D pooling name: ``'avg'``, ``'max'``, ``'blur'``, ``'sinc'``, ``'dwt'`` (and their aliases), applied channel-wise through the ``*_multichannel`` wrappers of :mod:`jade.model.pooling_1d` with decimation factor ``pooling_stride``;
        - ``'identity'`` / ``'none'`` : no pooling (the stage is replaced by ``nn.Identity``).
        Case-insensitive. Any other value raises ``ValueError``.
    pooling_kernel : int, default 75
        Length of the averaging window of the ``'default'`` overlapping average pooling. Ignored by the 1D operators (which derive their kernel length from the decimation factor). Must be a positive integer.
    pooling_stride : int, default 15
        Stride of the ``'default'`` pooling and decimation factor ``M`` of the 1D operators (the number of time samples becomes ``T / M``). Must be a positive integer.
        Note that the ``'dwt'`` operator additionally requires this to be a power of two (checked when the type-specific params are validated).
    params : dict, default {}
        Extra, type-specific parameters for a 1D operator (e.g. ``{'wavelet' : 'sym4'}`` for ``'dwt'``, ``{'cutoff_scale' : 0.8}`` for ``'sinc'``).
        Empty means "use the pooling layer defaults". See :mod:`jade.model.pooling_1d` for the accepted keys of each operator. Ignored by the ``'default'`` / ``'identity'`` types.
    """

    pooling_type   : str  = "default"
    pooling_kernel : int  = 75
    pooling_stride : int  = 15
    params         : dict = field(default_factory = dict)

    @property
    def pooling_layer_config(self) :
        """
        Per-stage pooling config consumed by :class:`~jade.model.EEGConformer.EEGConformer`.

        For an active 1D pooling operator it is a :class:`~jade.config.model.pooling_1d_cfg.pooling_config` dataclass (decimation factor ``pooling_stride`` plus the type-specific ``params``), keeping the whole flow dataclass-based;
        for the Conformer-only ``'default'`` (overlapping average pooling) and ``'identity'`` / ``'none'`` types, which have no config dataclass, it is a plain ``{pooling_type, pooling_kernel, pooling_stride, **params}`` dict.
        Consumed by :func:`jade.model.EEGConformer.get_conformer_pooling`, which accepts either.

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

        return {"pooling_type" : self.pooling_type, "pooling_kernel" : self.pooling_kernel, "pooling_stride" : self.pooling_stride, **self.params}


    def __post_init__(self) -> None :
        """
        Validate (and normalize) the pooling configuration.
        """

        # Lazy import to avoid an import cycle: this module is imported while jade.config.model is still initializing.
        from jade.config.model.pooling_1d_cfg import POOLING_CONFIG_REGISTRY, get_pooling_config

        # Valid pooling_type values: the 1D pooling names, plus 'default' (overlapping AvgPool2d) and the explicit no-op aliases.
        valid_types = set(POOLING_CONFIG_REGISTRY) | {"default", "identity", "none"}

        # Pooling type (lower-cased and validated).
        self.pooling_type = str(self.pooling_type).lower()
        if self.pooling_type not in valid_types :
            raise ValueError(f"pooling_type must be one of {sorted(valid_types)}, got {self.pooling_type!r}")

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
# EEG Conformer config class

@dataclass
class EEGConformer_config(model_config) :
    """
    Configuration for :class:`~jade.model.EEGConformer.EEGConformer` / :class:`~jade.model.EEGConformer.EEGConformer_Classifier`.

    The model couples a convolutional patch embedding (a shallow EEGNet-like front end) with a transformer encoder and, for the classifier, a flatten + MLP head.
    The variable names respect the nomenclature of the original EEG Conformer paper / reference implementation where possible.

    Parameters
    ----------
    name : str, default 'eegconformer_classifier'
        Which model to build. One of ``'eegconformer'`` (patch embedding + transformer encoder feature extractor) or ``'eegconformer_classifier'`` (feature extractor + flatten + MLP head).

    Patch embedding
    ---------------
    C : int, default 22
        Number of EEG channels of the input. It sets the height of the spatial convolution kernel ``(C, 1)``, which collapses the channel axis to 1.
    T : int, default 1000
        Number of time samples of the input.
    depth_first_layer : int, default 1
        Depth of the input tensor (``1`` for raw EEG, ``> 1`` for time-frequency input).
    filter_temporal : int, default 40
        Number of convolutional filters of both the temporal and the spatial convolution of the patch embedding.
    c_kernel_temporal : tuple of int, default (1, 25)
        Kernel size of the temporal convolution (convolution along the time axis only).
    emb_size : int, default 40
        Embedding size of the transformer tokens. The ``(1, 1)`` projection convolution maps ``filter_temporal`` to ``emb_size``. Must be divisible by ``num_heads``.
    activation : str, default 'elu'
        Activation function used in the patch embedding and the MLP head. One of ``relu``, ``elu``, ``selu``, ``gelu`` (case-insensitive).
    prob_dropout : float, default 0.5
        Dropout probability of the patch embedding (applied after pooling), in ``[0, 1]``.
    pooling : conformer_pooling_config, default conformer_pooling_config()
        The single pooling stage of the patch embedding (the ``[pooling]`` section).
        ``pooling_type = 'default'`` reproduces the original overlapping average pooling; a 1D pooling operator swaps it for a :mod:`jade.model.pooling_1d` layer applied channel-wise.
        A dict is accepted and promoted to :class:`conformer_pooling_config`.

    Transformer encoder
    -------------------
    depth : int, default 6
        Number of stacked transformer encoder blocks.
    num_heads : int, default 10
        Number of self-attention heads. ``emb_size`` must be divisible by it.
    transformer_dropout : float, default 0.5
        Dropout probability of the multi-head attention weights and of the two post-sublayer dropouts (the ``drop_p`` of the reference implementation), in ``[0, 1]``.
    forward_expansion : int, default 4
        Expansion factor of the position-wise feedforward block (its hidden size is ``forward_expansion * emb_size``).
    forward_dropout : float, default 0.5
        Dropout probability inside the feedforward block (the ``forward_drop_p`` of the reference implementation), in ``[0, 1]``.

    Classification head
    -------------------
    hidden_size_1 : int, default 256
        Size of the first hidden layer of the MLP classification head.
    hidden_size_2 : int, default 32
        Size of the second hidden layer of the MLP classification head.
    head_dropout_1 : float, default 0.5
        Dropout probability after the first hidden layer of the MLP head, in ``[0, 1]``.
    head_dropout_2 : float, default 0.3
        Dropout probability after the second hidden layer of the MLP head, in ``[0, 1]``.
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
        If True the (plain) :class:`~jade.model.EEGConformer.EEGConformer` feature extractor flattens its token sequence to ``(batch, n_tokens * emb_size)``. Ignored by the classifier (which always flattens internally).
    input_size : tuple of int, default (1, 1, 22, 1000)
        Shape ``(batch, depth, C, T)`` of a dummy input used by :class:`~jade.model.EEGConformer.EEGConformer_Classifier` to size the first layer of its MLP head.
    print_var : bool, default False
        If True print the parameter count (and, using ``input_size``, the shape tracking) at build time.
    """

    name : str = "eegconformer_classifier"

    # Patch embedding (convolutional front end)
    C                 : int = 22
    T                 : int = 1000
    depth_first_layer : int = 1
    filter_temporal   : int = 40
    c_kernel_temporal : Tuple[int, int] = (1, 25)
    emb_size          : int = 40
    activation        : str = "elu"
    prob_dropout      : float = 0.5

    # Transformer encoder
    depth               : int = 6
    num_heads           : int = 10
    transformer_dropout : float = 0.5
    forward_expansion   : int = 4
    forward_dropout     : float = 0.5

    # Classification head
    hidden_size_1  : int = 256
    hidden_size_2  : int = 32
    head_dropout_1 : float = 0.5
    head_dropout_2 : float = 0.3
    n_classes      : int = 4
    use_softmax    : bool = False

    # Misc
    flatten_output : bool = False
    input_size     : Tuple[int, int, int, int] = (1, 1, 22, 1000)
    print_var      : bool = False

    # Pooling (the [pooling] section). Declared last so it serializes after the scalar keys (a nested
    # config becomes a TOML sub-table, which must follow the parent's scalar keys).
    pooling        : conformer_pooling_config = field(default_factory = conformer_pooling_config)


    def __post_init__(self) -> None :
        """
        Validate (and normalize) every EEG Conformer parameter.
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
        _check_positive_int(self.filter_temporal, "filter_temporal")
        _check_positive_int(self.emb_size, "emb_size")
        _check_positive_int(self.depth, "depth")
        _check_positive_int(self.num_heads, "num_heads")
        _check_positive_int(self.forward_expansion, "forward_expansion")
        _check_positive_int(self.hidden_size_1, "hidden_size_1")
        _check_positive_int(self.hidden_size_2, "hidden_size_2")
        _check_positive_int(self.n_classes, "n_classes", minimum = 2)

        # The multi-head attention splits the embedding evenly across the heads, so emb_size must be divisible by num_heads.
        if self.emb_size % self.num_heads != 0 :
            raise ValueError(f"emb_size ({self.emb_size}) must be divisible by num_heads ({self.num_heads})")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Dropout probabilities

        self.prob_dropout        = _check_probability(self.prob_dropout, "prob_dropout")
        self.transformer_dropout = _check_probability(self.transformer_dropout, "transformer_dropout")
        self.forward_dropout     = _check_probability(self.forward_dropout, "forward_dropout")
        self.head_dropout_1      = _check_probability(self.head_dropout_1, "head_dropout_1")
        self.head_dropout_2      = _check_probability(self.head_dropout_2, "head_dropout_2")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Convolution kernel

        self.c_kernel_temporal = _coerce_kernel(self.c_kernel_temporal, "c_kernel_temporal")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Pooling section

        # Promote a dict to conformer_pooling_config (direct-construction path; from_dict already promotes it).
        if isinstance(self.pooling, dict) :
            self.pooling = conformer_pooling_config.from_dict(self.pooling)
        elif not isinstance(self.pooling, conformer_pooling_config) :
            raise ValueError(f"pooling must be a conformer_pooling_config (or a dict), got {type(self.pooling)}")

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
        Per-stage pooling config of the patch embedding (delegates to :attr:`pooling`).

        A :class:`~jade.config.model.pooling_1d_cfg.pooling_config` dataclass for an active 1D operator, or a plain dict for the Conformer-only ``'default'`` (overlapping average pooling) and ``'identity'`` / ``'none'`` types.
        This is what :class:`~jade.model.EEGConformer.EEGConformer` passes to :func:`~jade.model.EEGConformer.get_conformer_pooling`.
        """

        return self.pooling.pooling_layer_config
