"""
Configuration for the 1D anti-aliased pooling layers.

This single file holds the whole configuration family for :mod:`jade.model.pooling_1d`.
Pooling is meant to appear several times inside a model, and the various operators (average, blur, sinc, dwt, max) share most of their parameters, so a single config file with one dataclass per operator (rather than a subpackage) is the natural fit.

The family follows the same polymorphic pattern as the optimizer/loss configs: a base :class:`~jade.config.model.pooling_1d_cfg.pooling_config` carries a ``pooling_type`` discriminator and an overridden :meth:`from_dict`, an intermediate :class:`~jade.config.model.pooling_1d_cfg._fir_pooling_config` groups the options shared by the FIR-based operators, and each concrete config freezes its own ``pooling_type`` and validates its own parameters.
The ``pooling_type`` field (and the canonical field names) match exactly what :func:`~jade.model.pooling_1d.get_pooling_layer` reads, so ``config.to_dict()`` can be fed straight to the factory.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from dataclasses import dataclass
from typing import Optional, Union

# Internal imports
from jade.config.generic import generic_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module constants

# Kernel families accepted by blur_pool_1d.
BLUR_KERNELS = ("binomial", "gaussian", "boxcar")

# Padding modes and gain normalizations of the FIR base layer.
PADDING_MODES = ("reflect", "replicate", "constant", "circular")
GAINS = ("dc", "orthonormal", "none")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Helper functions

def _is_int(value) -> bool :
    """
    Return True only for a genuine integer (``bool`` is excluded, since it subclasses ``int``).
    """

    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool :
    """
    Return True for a genuine ``int``/``float`` (``bool`` excluded).
    """

    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_positive_int(value, field_name : str) -> None :
    """
    Raize ``ValueError`` if ``value`` is not a positive integer.
    """

    if not _is_int(value) or value <= 0 :
        raise ValueError("{} must be a positive integer, got {}".format(field_name, repr(value)))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Pooling config base class

@dataclass
class pooling_config(generic_config) :
    """
    Base class for every 1D pooling configuration.

    Parameters
    ----------
    pooling_type : str
        Discriminator with the name of the concrete pooling operator (e.g. ``'avg'``, ``'blur'``).
        Concrete subclasses freeze this value in their ``__post_init__``. It is the key read by :func:`~jade.model.pooling_1d.get_pooling_layer`.
    decimation_factor : int, default 2
        The factor ``M`` by which the time axis is reduced (kernel size and stride of the pooling).
    """

    pooling_type      : str = ""
    decimation_factor : int = 2


    @classmethod
    def from_dict(cls, data : dict) -> "pooling_config" :
        """
        Build a pooling config from a ``dict``.

        When called on the base class the concrete subclass is selected through the ``pooling_type`` field (polymorphic deserialization).
        When called on a concrete subclass the standard :meth:`~jade.config.generic.generic_config.from_dict` is used.

        Parameters
        ----------
        data : dict
            The dictionary to convert into a pooling config instance.

        Returns
        -------
        pooling_config
            An instance of the proper concrete subclass.
        """

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Polymorphic dispatch (only when called on the base class)

        if cls is pooling_config :
            pooling_type = str(data.get("pooling_type", "")).lower()
            if pooling_type not in POOLING_CONFIG_REGISTRY :
                raise ValueError("unknown pooling_type {}, known types: {}".format(repr(pooling_type), tuple(POOLING_CONFIG_REGISTRY)))
            return POOLING_CONFIG_REGISTRY[pooling_type].from_dict(data)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Standard behavior (called on a concrete subclass)

        return super().from_dict(data)


    def __post_init__(self) -> None :
        """
        Validate the decimation factor (shared by every pooling operator).
        """

        _check_positive_int(self.decimation_factor, "decimation_factor")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# FIR-based pooling (shared options)

@dataclass
class _fir_pooling_config(pooling_config) :
    """
    Options shared by every FIR-filter-based pooling operator (avg, blur, sinc, dwt).

    Parameters
    ----------
    padding_mode : str, default 'reflect'
        How the signal is extended before filtering. One of ``reflect``, ``replicate``, ``constant``, ``circular``.
    phase : {'center', 'first'} or int, default 'center'
        Which sample of each block of ``M`` is retained after filtering. ``'center'`` reproduces :class:`torch.nn.AvgPool1d`.
        An integer phase must lie in ``[0, decimation_factor)``.
    gain : {'dc', 'orthonormal', 'none'}, default 'dc'
        Kernel normalization (``'dc'`` -> unit DC gain, comparable with an average).
    warn_if_worse_than_boxcar : bool, default True
        Emit a warning at build time if the chosen filter passes more energy above the post-decimation Nyquist than a plain boxcar would.
    """

    padding_mode              : str = "reflect"
    phase                     : Union[str, int] = "center"
    gain                      : str = "dc"
    warn_if_worse_than_boxcar : bool = True


    def __post_init__(self) -> None :
        """
        Validate the shared FIR options (after the base decimation-factor check).
        """

        super().__post_init__()

        # Padding mode and gain are simple categorical checks.
        if self.padding_mode not in PADDING_MODES :
            raise ValueError("padding_mode must be one of {}, got {}".format(PADDING_MODES, repr(self.padding_mode)))
        if self.gain not in GAINS :
            raise ValueError("gain must be one of {}, got {}".format(GAINS, repr(self.gain)))

        # Phase is either a keyword or an integer sample offset in [0, M).
        if isinstance(self.phase, str) :
            if self.phase not in ("center", "first") :
                raise ValueError("string phase must be 'center' or 'first', got {}".format(repr(self.phase)))
        elif _is_int(self.phase) :
            if not 0 <= self.phase < self.decimation_factor :
                raise ValueError("integer phase must lie in [0, {}), got {}".format(self.decimation_factor, self.phase))
        else :
            raise ValueError("phase must be 'center', 'first' or an int, got {}".format(repr(self.phase)))

        if not isinstance(self.warn_if_worse_than_boxcar, bool) :
            raise ValueError("warn_if_worse_than_boxcar must be a bool, got {}".format(repr(self.warn_if_worse_than_boxcar)))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Average pooling (boxcar baseline)

@dataclass
class avg_pooling_config(_fir_pooling_config) :
    """
    Configuration for :class:`~jade.model.pooling_1d.avg_pool.avg_pool_1d` (boxcar / moving-average baseline).

    It adds no parameter beyond the shared FIR options.
    """

    pooling_type : str = "avg"


    def __post_init__(self) -> None :
        """
        Freeze the pooling type and validate the shared options.
        """

        self.pooling_type = "avg"
        super().__post_init__()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Blur pooling (binomial / gaussian / boxcar)

@dataclass
class blur_pooling_config(_fir_pooling_config) :
    """
    Configuration for :class:`~jade.model.pooling_1d.blur_pool.blur_pool_1d` (fixed heuristic kernel).

    Parameters
    ----------
    kernel : {'binomial', 'gaussian', 'boxcar'}, default 'binomial'
        The fixed kernel family.
    kernel_size : int, optional
        Length ``L`` of the blur kernel. ``None`` lets the layer use its ``4 * M + 1`` default (which scales with the decimation factor).
    sigma : float, optional
        Standard deviation of the Gaussian kernel. ``None`` lets the layer default to ``kernel_size / 6``. Only used when ``kernel == 'gaussian'``.
    """

    pooling_type : str = "blur"

    kernel      : str = "binomial"
    kernel_size : Optional[int] = None
    sigma       : Optional[float] = None


    def __post_init__(self) -> None :
        """
        Freeze the pooling type and validate the blur-specific parameters.
        """

        self.pooling_type = "blur"
        super().__post_init__()

        # Kernel family.
        self.kernel = str(self.kernel).lower()
        if self.kernel not in BLUR_KERNELS :
            raise ValueError("kernel must be one of {}, got {}".format(BLUR_KERNELS, repr(self.kernel)))

        # Optional overrides: validated only when provided (None -> layer default).
        if self.kernel_size is not None :
            _check_positive_int(self.kernel_size, "kernel_size")
        if self.sigma is not None :
            if not _is_number(self.sigma) or self.sigma <= 0 :
                raise ValueError("sigma must be a positive number, got {}".format(repr(self.sigma)))
            self.sigma = float(self.sigma)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Sinc pooling (Kaiser windowed-sinc)

@dataclass
class sinc_pooling_config(_fir_pooling_config) :
    """
    Configuration for :class:`~jade.model.pooling_1d.sinc_pool.sinc_pool_1d` (designed windowed-sinc low-pass).

    Parameters
    ----------
    cutoff_scale : float, default 0.9
        Cutoff as a fraction of the post-decimation Nyquist ``pi / M``. Must be in ``(0, 1]``.
    window : str, optional
        Any :func:`scipy.signal.firwin` window name. ``None`` lets the layer use a Kaiser window derived from ``beta``.
    num_taps : int, optional
        Filter length. ``None`` lets the layer use its ``8 * M + 1`` default (forced odd).
    beta : float, default 8.6
        Kaiser ``beta`` used when ``window`` is ``None`` and no stopband attenuation is given.
    stopband_attenuation_db : float, optional
        Target stopband attenuation (dB). When set it drives the Kaiser ``beta`` (and, with ``transition_width``, the tap count).
    transition_width : float, optional
        Transition width as a fraction of the cutoff. With ``stopband_attenuation_db`` it sizes ``num_taps``.
    max_taps : int, optional
        Optional cap on a derived ``num_taps``.
    """

    pooling_type : str = "sinc"

    cutoff_scale            : float = 0.9
    window                  : Optional[str] = None
    num_taps                : Optional[int] = None
    beta                    : float = 8.6
    stopband_attenuation_db : Optional[float] = None
    transition_width        : Optional[float] = None
    max_taps                : Optional[int] = None


    def __post_init__(self) -> None :
        """
        Freeze the pooling type and validate the sinc-specific parameters.
        """

        self.pooling_type = "sinc"
        super().__post_init__()

        # Cutoff is a fraction of the post-decimation Nyquist.
        if not _is_number(self.cutoff_scale) or not 0.0 < self.cutoff_scale <= 1.0 :
            raise ValueError("cutoff_scale must be a number in (0, 1], got {}".format(repr(self.cutoff_scale)))
        self.cutoff_scale = float(self.cutoff_scale)

        # Kaiser beta is always meaningful (used when no window/attenuation is given).
        if not _is_number(self.beta) or self.beta <= 0 :
            raise ValueError("beta must be a positive number, got {}".format(repr(self.beta)))
        self.beta = float(self.beta)

        # Window name (only validated as a non-empty string when provided).
        if self.window is not None and (not isinstance(self.window, str) or self.window == "") :
            raise ValueError("window must be a non-empty string or None, got {}".format(repr(self.window)))

        # Optional overrides: validated only when provided (None -> layer default).
        if self.num_taps is not None :
            _check_positive_int(self.num_taps, "num_taps")
        if self.max_taps is not None :
            _check_positive_int(self.max_taps, "max_taps")
        if self.stopband_attenuation_db is not None :
            if not _is_number(self.stopband_attenuation_db) or self.stopband_attenuation_db <= 0 :
                raise ValueError("stopband_attenuation_db must be a positive number, got {}".format(repr(self.stopband_attenuation_db)))
            self.stopband_attenuation_db = float(self.stopband_attenuation_db)
        if self.transition_width is not None :
            if not _is_number(self.transition_width) or self.transition_width <= 0 :
                raise ValueError("transition_width must be a positive number, got {}".format(repr(self.transition_width)))
            self.transition_width = float(self.transition_width)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# DWT pooling (orthogonal wavelet low-pass)

@dataclass
class dwt_pooling_config(_fir_pooling_config) :
    """
    Configuration for :class:`~jade.model.pooling_1d.dwt_pool.dwt_pool_1d` (orthogonal wavelet analysis low-pass).

    Parameters
    ----------
    wavelet : str, default 'db4'
        Orthogonal wavelet name (e.g. ``haar``, ``db1`` ... ``db4``, ``sym4``, or any PyWavelets orthogonal wavelet).

    Notes
    -----
    The decimation factor of a wavelet pooling must be a power of two (it sets the number of cascaded analysis levels ``log2(M)``).
    """

    pooling_type : str = "dwt"

    wavelet : str = "db4"


    def __post_init__(self) -> None :
        """
        Freeze the pooling type, validate the wavelet name and enforce the power-of-two decimation factor.
        """

        self.pooling_type = "dwt"
        super().__post_init__()

        # A wavelet pooling cascades log2(M) analysis levels, so M must be a power of two.
        M = self.decimation_factor
        if (M & (M - 1)) != 0 :
            raise ValueError("decimation_factor must be a power of two for dwt pooling, got {}".format(repr(M)))

        # Wavelet name.
        if not isinstance(self.wavelet, str) or self.wavelet == "" :
            raise ValueError("wavelet must be a non-empty string, got {}".format(repr(self.wavelet)))
        self.wavelet = self.wavelet.lower()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Max pooling (nonlinear reference)

@dataclass
class max_pooling_config(pooling_config) :
    """
    Configuration for :class:`~jade.model.pooling_1d.max_pool.max_pool_1d` (nonlinear, un-anti-aliased reference).

    The max operator has no FIR filter, so it does not inherit the shared FIR options (padding/phase/gain/...).

    Parameters
    ----------
    ceil_mode : bool, default False
        If True use ``ceil`` instead of ``floor`` to compute the output length (see :class:`torch.nn.MaxPool1d`).
    """

    pooling_type : str = "max"

    ceil_mode : bool = False


    def __post_init__(self) -> None :
        """
        Freeze the pooling type and validate the max-specific parameter.
        """

        self.pooling_type = "max"
        super().__post_init__()

        if not isinstance(self.ceil_mode, bool) :
            raise ValueError("ceil_mode must be a bool, got {}".format(repr(self.ceil_mode)))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Registry and factory

# Map every alias accepted by jade.model.pooling_1d.get_pooling_layer to the matching config class.
POOLING_CONFIG_REGISTRY = {
    "avg"      : avg_pooling_config,
    "average"  : avg_pooling_config,
    "boxcar"   : avg_pooling_config,
    "max"      : max_pooling_config,
    "blur"     : blur_pooling_config,
    "blurpool" : blur_pooling_config,
    "binomial" : blur_pooling_config,
    "gaussian" : blur_pooling_config,
    "sinc"     : sinc_pooling_config,
    "kaiser"   : sinc_pooling_config,
    "dwt"      : dwt_pooling_config,
    "wavelet"  : dwt_pooling_config,
}


def get_pooling_config(config) -> pooling_config :
    """
    Return a pooling config instance from a dict or an already-built config.

    Parameters
    ----------
    config : pooling_config or dict
        Either an already-built pooling config (returned untouched) or a dict with at least the key ``pooling_type`` (promoted to the proper subclass through the registry).

    Returns
    -------
    pooling_config
        The pooling configuration instance.
    """

    # Already a config: nothing to do.
    if isinstance(config, pooling_config) :
        return config

    # A dict: promote it through the polymorphic from_dict of the base class.
    if isinstance(config, dict) :
        return pooling_config.from_dict(config)

    raise TypeError("get_pooling_config expects a pooling_config or a dict, got {}".format(type(config)))
