"""
Configuration for the braindecode pooling-extension wrapper.

:class:`~jade.config.model.braindecode_pooling_extension_cfg.braindecode_wrapper_config` describes how to
build a :class:`~jade.model.braindecode_pooling_extension.braindecode_pooling_wrapper`: which braindecode
model to instantiate, the keyword arguments to hand it, and which :mod:`jade.model.pooling_1d` operator to
substitute for its pooling layers.

Unlike :class:`~jade.config.model.EEGNet_cfg.EEGNet_config`, the braindecode model hyper-parameters are **not**
given their own dataclass: braindecode already exposes a large, per-model set of arguments and validates them
itself, so they travel as a plain ``model_kwargs`` dict handed straight to the braindecode constructor.
What *is* described here (through the shared :class:`~jade.config.model.pooling_1d_cfg.pooling_config` family)
is the pooling operator to swap in - the point of the wrapper.

The config is torch-free and braindecode-free (like the rest of :mod:`jade.config`): braindecode is imported
only by the runtime module :mod:`jade.model.braindecode_pooling_extension`.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Internal imports
from jade.config.model.generic import model_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module constants

# Names this config can describe (mirrors the aliases of jade.model.MODEL_REGISTRY / MODEL_CONFIG_REGISTRY).
ALLOWED_NAMES = ("braindecode_wrapper", "braindecode", "braindecode_pooling")

# Where the decimation factor M of each swapped layer is read from on the original braindecode pooling layer.
POOLING_FACTOR_SOURCES = ("stride", "kernel")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Braindecode wrapper config class

@dataclass
class braindecode_wrapper_config(model_config) :
    """
    Configuration for :class:`~jade.model.braindecode_pooling_extension.braindecode_pooling_wrapper`.

    The defaults give a valid skeleton (a ``ShallowFBCSPNet`` whose average pooling is re-expressed with the
    ``avg`` operator of :mod:`jade.model.pooling_1d`); a real run overrides ``braindecode_model_name``,
    ``model_kwargs`` and ``pooling``. Note that ``model_kwargs`` is left empty by default: the braindecode
    model validates its own (per-model, mostly required) arguments, so an empty dict only becomes an error at
    build time.

    Parameters
    ----------
    name : str, default 'braindecode_wrapper'
        Discriminator selecting the wrapper in :func:`~jade.model.get_model`. One of ``'braindecode_wrapper'``,
        ``'braindecode'``, ``'braindecode_pooling'`` (case-insensitive; they are aliases of the same wrapper).
    braindecode_model_name : str, default 'ShallowFBCSPNet'
        Name of the braindecode model class to build (resolved against ``braindecode.models``), e.g.
        ``'ShallowFBCSPNet'``, ``'EEGNetv4'``, ``'Deep4Net'``.
    model_kwargs : dict, default {}
        Keyword arguments passed straight to the braindecode model constructor (e.g. ``n_chans``, ``n_outputs``,
        ``n_times``). braindecode validates them, so they are not checked here.
    pooling : dict, default {'pooling_type': 'avg'}
        The pooling operator to substitute for every pooling layer of the braindecode model, as a
        :class:`~jade.config.model.pooling_1d_cfg.pooling_config` dict (``pooling_type`` plus any operator
        parameters, e.g. ``{'pooling_type': 'sinc', 'cutoff_scale': 0.8}``).
        Its ``decimation_factor`` is **ignored**: every swapped layer keeps the factor ``M`` of the original
        braindecode pooling layer (only the pooling *type* changes). Reachable as a dataclass through
        :attr:`target_pooling_config`.
    keep_pool_nonlin : bool, default True
        Whether to keep the post-pool nonlinearity that some braindecode models wrap the pooling with
        (e.g. ShallowFBCSPNet's ``safe_log`` in the ``pool_nonlin`` / ``pool_nonlin_exp`` module).
        ``True`` (default) changes only the pooling type, leaving the ``square -> pool -> log`` pipeline intact;
        ``False`` replaces that post-pool nonlinearity with :class:`torch.nn.Identity` (an extra ablation arm).
    pooling_factor_source : str, default 'stride'
        Which attribute of the original pooling layer defines the decimation factor ``M`` of its replacement.
        ``'stride'`` (default) uses the layer stride - the true downsampling factor, correct for both
        non-overlapping and overlapping pooling; ``'kernel'`` uses the kernel size instead.
    example_input_shape : tuple of int, optional
        Optional explicit input shape (batch included) used to probe the model - the dummy forward that re-sizes
        the trailing classifier after an overlapping-pooling swap, and the build-time validation forward. When
        ``None`` the shape is inferred from the built model / ``model_kwargs`` (``n_chans`` / ``n_times`` ...),
        which matches the braindecode-native ``(B, n_chans, n_times)`` layout (braindecode's own ``ensuredims`` /
        ``dimshuffle`` front layers then adapt it to each model's internal 3D/4D layout).
    validate_forward : bool, default True
        If True, run one probe forward on the swapped model at build time (using ``example_input_shape`` or the
        inferred shape) and raise a clear error if it fails. This is the "check the expected data shape" step: it
        confirms the swap produced a runnable model, and turns the cryptic errors of the few unsupported cases
        (an overlapping pooling feeding a non-conv head, or a pooling filter longer than the signal) into an
        actionable message. Set to False to skip the check (e.g. for a model whose input shape cannot be inferred).
    """

    name : str = "braindecode_wrapper"

    braindecode_model_name : str  = "ShallowFBCSPNet"
    model_kwargs           : dict = field(default_factory = dict)
    pooling                : dict = field(default_factory = lambda : {"pooling_type" : "avg"})
    keep_pool_nonlin       : bool = True
    pooling_factor_source  : str  = "stride"
    example_input_shape    : Optional[Tuple[int, ...]] = None
    validate_forward       : bool = True


    @property
    def target_pooling_config(self) :
        """
        The pooling operator to swap in, as a :class:`~jade.config.model.pooling_1d_cfg.pooling_config` dataclass.

        Promotes the ``pooling`` dict through :func:`~jade.config.model.pooling_1d_cfg.get_pooling_config`.
        Its ``decimation_factor`` is a placeholder - the wrapper overrides it per layer with the factor of the
        braindecode pooling layer being replaced.

        Returns
        -------
        pooling_config
            The (validated) target pooling configuration.
        """

        # Lazy import to avoid an import cycle: this module is imported while jade.config.model is still initializing.
        from jade.config.model.pooling_1d_cfg import get_pooling_config

        return get_pooling_config(self.pooling)


    def __post_init__(self) -> None :
        """
        Validate (and normalize) the wrapper configuration.
        """

        # Lazy import to avoid an import cycle (see target_pooling_config).
        from jade.config.model.pooling_1d_cfg import get_pooling_config

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Discriminator

        # The model name selects the class in jade.model.get_model, so it must be one of the known aliases.
        self.name = str(self.name).lower()
        if self.name not in ALLOWED_NAMES :
            raise ValueError(f"name must be one of {ALLOWED_NAMES}, got {self.name!r}")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # braindecode model selection

        # The model name is resolved (and its kwargs validated) by braindecode at build time; here we only make
        # sure it is a usable, non-empty string.
        if not isinstance(self.braindecode_model_name, str) or self.braindecode_model_name.strip() == "" :
            raise ValueError(f"braindecode_model_name must be a non-empty string, got {self.braindecode_model_name!r}")

        if not isinstance(self.model_kwargs, dict) :
            raise ValueError(f"model_kwargs must be a dict, got {self.model_kwargs!r}")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Pooling operator to swap in

        if not isinstance(self.pooling, dict) :
            raise ValueError(f"pooling must be a dict, got {self.pooling!r}")

        # Validate the target pooling config early by round-tripping it through the 1D pooling config family
        # (this checks the pooling_type and any operator-specific parameters).
        get_pooling_config(self.pooling)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Flags and enumerations

        if not isinstance(self.keep_pool_nonlin, bool) :
            raise ValueError(f"keep_pool_nonlin must be a bool, got {self.keep_pool_nonlin!r}")

        if not isinstance(self.validate_forward, bool) :
            raise ValueError(f"validate_forward must be a bool, got {self.validate_forward!r}")

        self.pooling_factor_source = str(self.pooling_factor_source).lower()
        if self.pooling_factor_source not in POOLING_FACTOR_SOURCES :
            raise ValueError(f"pooling_factor_source must be one of {POOLING_FACTOR_SOURCES}, got {self.pooling_factor_source!r}")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Optional probe shape

        # A TOML/JSON round-trip turns a tuple into a list, so coerce it back and validate it holds positive ints.
        if self.example_input_shape is not None :
            shape = tuple(self.example_input_shape)
            for element in shape :
                if not (isinstance(element, int) and not isinstance(element, bool)) or element <= 0 :
                    raise ValueError(f"example_input_shape must contain positive integers, got {self.example_input_shape!r}")
            self.example_input_shape = shape
