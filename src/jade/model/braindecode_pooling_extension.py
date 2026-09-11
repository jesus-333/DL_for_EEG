"""
Wrap a braindecode model and swap its pooling layers for the jade pooling operators.

Motivation
----------
:mod:`jade.model.pooling_1d` implements a family of anti-aliased decimation ("pooling") operators
(``avg`` / ``max`` / ``blur`` / ``sinc`` / ``dwt``) meant to be compared as the *only* changing object of an
ablation. So far that ablation could only be run on the repo's own :class:`~jade.model.EEGNet.EEGNet`.
This module lets the same ablation run on the many models already implemented in
`braindecode <https://braindecode.org/>`_ (``ShallowFBCSPNet``, ``EEGNet``, ``Deep4Net``, ...): it builds the
braindecode model untouched, then **substitutes every pooling layer** with the chosen jade operator, *keeping the
decimation factor the same* - only the pooling *type* changes.

How the substitution works
--------------------------
The wrapper walks the module tree of the braindecode model and, for each pooling layer
(:class:`torch.nn.AvgPool1d` / :class:`~torch.nn.MaxPool1d` / :class:`~torch.nn.AvgPool2d` /
:class:`~torch.nn.MaxPool2d`), builds a replacement from the :class:`~jade.config.model.pooling_1d_cfg.pooling_config`
and splices it in.

* **Decimation factor ``M``** is read from the *original* layer (its stride by default - the true downsampling
  factor - or its kernel size), so the replacement reduces the time axis by exactly the same amount.
* **1D vs 2D and the pooled axis** are detected from the original kernel, not hard-coded per model: a 2D kernel
  ``(M, 1)`` pools axis 2 (e.g. ShallowFBCSPNet, internal layout ``B x F x T x 1``), a kernel ``(1, M)`` pools
  axis 3 (e.g. EEGNet, ``B x F x C x T``). A kernel that pools *both* spatial axes is rejected (these operators
  only decimate along time). This is what the spec's "check the expected input data shape / 1D vs 2D" asks for.
* The jade 1D operators consume ``(B, D, T)`` and output exactly ``T // M`` (see :mod:`jade.model.pooling_1d`).
  For a 2D pooling site the wrapped 1D layer is applied through :class:`_pool_along_axis_2d`, which moves the
  temporal axis last, flattens the other axes into the depth axis (the same trick as
  :class:`~jade.model.pooling_1d.support_pooling._multichannel_pool_1d`), pools, and restores the layout.

Overlapping pooling
-------------------
When the original pooling overlaps (stride != kernel, e.g. ShallowFBCSPNet's ``kernel 75, stride 15``) the jade
decimator's ``T // stride`` output length differs from braindecode's ``(T - kernel) // stride + 1``, which
invalidates the size of the trailing classification layer that braindecode computed at construction. In that case
the wrapper re-sizes that trailing layer with a single dummy forward (:func:`_resize_trailing_classifier`). Purely
non-overlapping models (EEGNet, Deep4Net) keep every downstream shape and never trigger this.

The braindecode model hyper-parameters are validated by braindecode itself (they travel as a raw ``model_kwargs``
dict); braindecode is imported lazily, so importing :mod:`jade.model` never requires it.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Full module imports
import inspect
import warnings

# Specific imports
from typing import List, Optional, Tuple

# Third party imports
import torch
from torch import nn

# Internal imports
from jade.config.model import get_model_config
from jade.config.model.braindecode_pooling_extension_cfg import braindecode_wrapper_config
from jade.model.pooling_1d import get_pooling_layer

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module constants

# Pooling layer types recognized (and replaced) inside a braindecode model.
_POOL_1D_TYPES = (nn.AvgPool1d, nn.MaxPool1d)
_POOL_2D_TYPES = (nn.AvgPool2d, nn.MaxPool2d)
_POOL_TYPES = _POOL_1D_TYPES + _POOL_2D_TYPES

# Attribute-name prefix of the post-pool nonlinearity some braindecode models wrap the pooling with
# (e.g. ShallowFBCSPNet's 'pool_nonlin_exp' = safe_log, Deep4Net's 'pool_nonlin' = identity).
_POOL_NONLIN_PREFIX = "pool_nonlin"

# Attribute names under which braindecode models expose their trailing classification block.
_FINAL_LAYER_ATTRS = ("final_layer", "conv_classifier", "classifier")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Small helpers

def _as_pair(value) -> Tuple[int, int] :
    """
    Normalize a pooling attribute (``kernel_size`` / ``stride``) to a ``(height, width)`` int pair.

    Accepts an ``int`` (both dimensions equal), a 1-tuple (same) or a 2-tuple/list.

    Parameters
    ----------
    value : int or tuple or list
        The raw attribute value.

    Returns
    -------
    tuple of int
        The ``(height, width)`` pair.
    """

    if isinstance(value, (tuple, list)) :
        values = [int(single_value) for single_value in value]
        if len(values) == 1 :
            return values[0], values[0]
        if len(values) == 2 :
            return values[0], values[1]
        raise ValueError(f"cannot interpret {value!r} as a 2D pooling attribute")
    return int(value), int(value)


def _set_submodule(root : nn.Module, qualified_name : str, new_module : nn.Module) -> None :
    """
    Replace the submodule reached by ``qualified_name`` with ``new_module`` (module surgery).

    ``qualified_name`` is a dotted path as produced by :meth:`torch.nn.Module.named_modules` (e.g.
    ``'block_3.3'``); the parent is looked up with :meth:`~torch.nn.Module.get_submodule` and the leaf is set with
    :func:`setattr` (which also handles the numeric attribute names used by :class:`torch.nn.Sequential`).

    Parameters
    ----------
    root : torch.nn.Module
        The model holding the submodule.
    qualified_name : str
        Dotted path to the submodule to replace.
    new_module : torch.nn.Module
        The replacement module.
    """

    parent_path, _, attr = qualified_name.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, attr, new_module)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# braindecode model resolution

def _get_braindecode_model_class(model_name : str) -> type :
    """
    Resolve a braindecode model class by name from :mod:`braindecode.models`.

    Only names that resolve to an :class:`torch.nn.Module` subclass are accepted (the ``braindecode.models``
    namespace also exposes lowercase sub-*modules* and the ``EEGModuleMixin`` base, which are not buildable models).

    Parameters
    ----------
    model_name : str
        The braindecode model class name (e.g. ``'ShallowFBCSPNet'``).

    Returns
    -------
    type
        The braindecode model class.

    Raises
    ------
    ValueError
        If ``model_name`` does not name a braindecode model class.
    """

    # Lazy import: braindecode is an optional dependency, so importing jade.model must not require it.
    import braindecode.models as braindecode_models

    def _is_model_class(obj) -> bool :
        return inspect.isclass(obj) and issubclass(obj, nn.Module) and obj.__name__ != "EEGModuleMixin"

    candidate = getattr(braindecode_models, model_name, None)
    if _is_model_class(candidate) :
        return candidate

    available = sorted({name for name in dir(braindecode_models) if _is_model_class(getattr(braindecode_models, name, None))})
    raise ValueError(f"unknown braindecode model {model_name!r}; available models: {available}")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Pooling replacement

def _resolve_factor_and_axis(original_pool : nn.Module, factor_source : str) :
    """
    Read the geometry of a braindecode pooling layer needed to build its replacement.

    Parameters
    ----------
    original_pool : torch.nn.Module
        The original braindecode pooling layer.
    factor_source : {'stride', 'kernel'}
        Which attribute defines the decimation factor ``M``.

    Returns
    -------
    ndim : int
        Dimensionality of the pooling (``1`` or ``2``).
    axis : int or None
        The tensor axis that is decimated (``2`` for a 1D pool, ``2`` or ``3`` for a 2D pool). ``None`` when the
        layer does not pool any axis (a ``(1, 1)`` kernel).
    factor : int
        The decimation factor ``M`` (from the stride or the kernel size, per ``factor_source``).
    overlapping : bool
        Whether the original pooling overlaps (stride != kernel on the pooled axis).
    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 1D pooling: the model tensor is (B, D, T); the pooled axis is the last one.

    if isinstance(original_pool, _POOL_1D_TYPES) :
        kernel = original_pool.kernel_size
        stride = original_pool.stride
        kernel = int(kernel[0]) if isinstance(kernel, (tuple, list)) else int(kernel)
        stride = kernel if stride is None else (int(stride[0]) if isinstance(stride, (tuple, list)) else int(stride))
        factor = stride if factor_source == "stride" else kernel
        return 1, 2, factor, stride != kernel

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 2D pooling: the decimated axis is the kernel dimension that is > 1 (purely temporal decimation).

    kernel_h, kernel_w = _as_pair(original_pool.kernel_size)
    stride = original_pool.stride
    stride_h, stride_w = (kernel_h, kernel_w) if stride is None else _as_pair(stride)

    pooled_h = kernel_h > 1
    pooled_w = kernel_w > 1

    if pooled_h and pooled_w :
        raise ValueError(
            f"2D pooling kernel {(kernel_h, kernel_w)} decimates both spatial axes; these operators only "
            "decimate along the temporal axis. This braindecode model is not supported by the wrapper."
        )

    # A (1, 1) kernel pools nothing.
    if not pooled_h and not pooled_w :
        return 2, None, 1, False

    if pooled_h :
        axis, kernel, stride_value = 2, kernel_h, stride_h
    else :
        axis, kernel, stride_value = 3, kernel_w, stride_w

    factor = stride_value if factor_source == "stride" else kernel
    return 2, axis, factor, stride_value != kernel


def _build_replacement(original_pool : nn.Module, pooling_config, factor_source : str) :
    """
    Build the jade pooling layer that replaces ``original_pool``, keeping its decimation factor.

    Parameters
    ----------
    original_pool : torch.nn.Module
        The original braindecode pooling layer.
    pooling_config : pooling_config
        The target pooling operator (its ``decimation_factor`` is overridden with the factor of ``original_pool``).
    factor_source : {'stride', 'kernel'}
        Which attribute of ``original_pool`` defines the decimation factor.

    Returns
    -------
    replacement : torch.nn.Module or None
        The replacement layer, or ``None`` when the layer does not decimate (factor ``<= 1``) and must be kept.
    overlapping : bool
        Whether the original pooling overlapped.
    """

    ndim, axis, factor, overlapping = _resolve_factor_and_axis(original_pool, factor_source)

    # A stride-1 (or (1, 1)) pooling is not a decimation: the jade operators cannot reproduce it, so leave the
    # original layer untouched rather than silently turning it into an identity.
    if factor <= 1 :
        return None, False

    # Clone the target config and override its decimation factor with the one of this pooling layer.
    # (never mutate the caller's config).
    config_dict = pooling_config.to_dict()
    config_dict["decimation_factor"] = int(factor)

    # A 1D pooling site takes the 1D operator directly (it works on the (B, D, T) tensor).
    if ndim == 1 :
        return get_pooling_layer(config_dict), overlapping

    # A 2D pooling site wraps the 1D operator in the axis-aware adapter.
    return _pool_along_axis_2d(get_pooling_layer(config_dict), axis), overlapping


def _replace_pooling_layers(model : nn.Module, pooling_config, factor_source : str) -> Tuple[List[dict], bool] :
    """
    Replace every pooling layer of ``model`` with the jade operator described by ``pooling_config``.

    Parameters
    ----------
    model : torch.nn.Module
        The braindecode model (modified in place).
    pooling_config : pooling_config
        The target pooling operator.
    factor_source : {'stride', 'kernel'}
        Which attribute of each original layer defines its decimation factor.

    Returns
    -------
    swaps : list of dict
        One entry ``{'name', 'type', 'factor', 'overlapping'}`` per replaced layer.
    any_overlapping : bool
        Whether any replaced layer overlapped (so the trailing classifier must be re-sized).
    """

    # Collect the targets first: the tree must not be mutated while it is being walked.
    targets = [(name, module) for name, module in model.named_modules() if isinstance(module, _POOL_TYPES)]

    swaps = []
    any_overlapping = False
    for name, original_pool in targets :
        replacement, overlapping = _build_replacement(original_pool, pooling_config, factor_source)

        if replacement is None :
            warnings.warn(
                f"pooling layer {name!r} ({type(original_pool).__name__}) does not decimate "
                "(stride-1 / (1, 1) kernel); it is left unchanged.",
                stacklevel = 3,
            )
            continue

        _set_submodule(model, name, replacement)
        _, _, factor, _ = _resolve_factor_and_axis(original_pool, factor_source)
        swaps.append({"name" : name, "type" : type(original_pool).__name__, "factor" : factor, "overlapping" : overlapping})
        any_overlapping = any_overlapping or overlapping

    if len(swaps) == 0 :
        warnings.warn("no pooling layer was replaced in the braindecode model.", stacklevel = 3)

    return swaps, any_overlapping


def _drop_pool_nonlin(model : nn.Module) -> List[str] :
    """
    Replace every post-pool nonlinearity (``pool_nonlin*`` module) with :class:`torch.nn.Identity`.

    Used when ``keep_pool_nonlin`` is ``False``. Modules already equal to :class:`torch.nn.Identity` (e.g.
    Deep4Net's) are skipped.

    Parameters
    ----------
    model : torch.nn.Module
        The braindecode model (modified in place).

    Returns
    -------
    list of str
        The qualified names of the modules that were replaced.
    """

    targets = [
        name
        for name, module in model.named_modules()
        if name.rpartition(".")[2].startswith(_POOL_NONLIN_PREFIX) and not isinstance(module, nn.Identity)
    ]

    for name in targets :
        _set_submodule(model, name, nn.Identity())

    return targets

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2D axis-aware pooling adapter

class _pool_along_axis_2d(nn.Module) :
    """
    Apply a 1D jade pooling layer along a chosen axis of a 4D tensor.

    braindecode models decimate the time axis of a 4D tensor with a 2D pooling layer whose kernel is ``(M, 1)`` or
    ``(1, M)``. The jade operators are 1D (they consume ``(B, D, T)``), so this adapter brings the temporal axis to
    the last position, flattens the remaining non-batch axes into a single depth axis, runs the 1D layer once, and
    restores the original layout::

        (B, a, b, c)  --movedim(axis, -1)-->  (B, ..., T)  --reshape-->  (B, D, T)
                      --pool-->  (B, D, T // M)  --reshape / movedim-->  (B, a, b, c') with axis decimated

    This is the same flatten-pool-restore trick as
    :class:`~jade.model.pooling_1d.support_pooling._multichannel_pool_1d`, generalized to pool axis 2 *or* 3.

    Parameters
    ----------
    pool_1d : torch.nn.Module
        The 1D pooling layer to apply. It must map ``(B, D, T) -> (B, D, T // M)``.
    axis : int
        The tensor axis (``2`` or ``3``) that ``pool_1d`` should decimate.
    """

    def __init__(self, pool_1d : nn.Module, axis : int) :
        super().__init__()
        self.pool_1d = pool_1d
        self.axis = int(axis)


    def forward(self, x : torch.Tensor) -> torch.Tensor :
        """
        Decimate the temporal axis of a 4D tensor with the wrapped 1D pooling layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, a, b, c)`` whose axis ``self.axis`` is temporal.

        Returns
        -------
        torch.Tensor
            The tensor with axis ``self.axis`` decimated by the wrapped layer's factor.
        """

        if x.dim() != 4 :
            raise ValueError(f"_pool_along_axis_2d expects a 4D tensor, got shape {tuple(x.shape)}")

        # Bring the temporal axis to the last position, flatten the rest into the depth axis, pool, and restore.
        x = x.movedim(self.axis, -1)
        batch, dim_a, dim_b, time = x.shape
        pooled = self.pool_1d(x.reshape(batch, dim_a * dim_b, time))
        pooled = pooled.reshape(batch, dim_a, dim_b, pooled.shape[-1])
        return pooled.movedim(-1, self.axis)


    def extra_repr(self) -> str :
        """
        One-line description used by ``print(model)``.
        """

        return f"axis={self.axis}"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Trailing-classifier re-sizing (overlapping pooling only)

class _ProbeDone(Exception) :
    """
    Sentinel raised by the probe hook to stop a dummy forward once the trailing layer's input is captured.
    """

    def __init__(self, shape : Tuple[int, ...]) :
        super().__init__()
        self.shape = shape


def _find_final_layer(model : nn.Module) :
    """
    Locate the trailing classification layer (a :class:`~torch.nn.Conv2d` / :class:`~torch.nn.Linear`) to re-size.

    It first looks inside the usual braindecode blocks (``final_layer`` / ``conv_classifier`` / ``classifier``)
    and, failing that, falls back to the last ``Conv2d`` / ``Linear`` of the whole model.

    Parameters
    ----------
    model : torch.nn.Module
        The braindecode model.

    Returns
    -------
    tuple
        ``(qualified_name, module)`` of the trailing layer, or ``(None, None)`` if none is found.
    """

    # Preferred: the first conv/linear inside a known trailing block.
    for attr in _FINAL_LAYER_ATTRS :
        block = getattr(model, attr, None)
        if isinstance(block, nn.Module) :
            for name, module in block.named_modules() :
                if isinstance(module, (nn.Conv2d, nn.Linear)) :
                    qualified = attr if name == "" else f"{attr}.{name}"
                    return qualified, module

    # Fallback: the last conv/linear anywhere in the model.
    last = (None, None)
    for name, module in model.named_modules() :
        if isinstance(module, (nn.Conv2d, nn.Linear)) :
            last = (name, module)
    return last


def _probe_input_shape(model : nn.Module, dummy_input : torch.Tensor, layer : nn.Module) -> Optional[Tuple[int, ...]] :
    """
    Run a dummy forward and capture the shape of the tensor entering ``layer`` (stopping there).

    Parameters
    ----------
    model : torch.nn.Module
        The (already pooling-swapped) braindecode model.
    dummy_input : torch.Tensor
        A batch-shaped probe input.
    layer : torch.nn.Module
        The layer whose input shape is wanted.

    Returns
    -------
    tuple of int or None
        The captured input shape, or ``None`` if the forward did not reach ``layer``.
    """

    def _pre_hook(module, inputs) :
        raise _ProbeDone(tuple(inputs[0].shape))

    handle = layer.register_forward_pre_hook(_pre_hook)
    was_training = model.training
    captured = None
    try :
        model.eval()
        with torch.no_grad() :
            model(dummy_input)
    except _ProbeDone as probe :
        captured = probe.shape
    finally :
        handle.remove()
        model.train(was_training)

    return captured


def _rebuild_final_layer(layer : nn.Module, input_shape : Tuple[int, ...]) -> Optional[nn.Module] :
    """
    Build a copy of the trailing layer re-sized to ``input_shape``.

    A braindecode trailing ``Conv2d`` collapses the temporal axis to length 1 with a kernel equal to the current
    temporal length; after a pooling swap changed that length, the kernel is rebuilt to match. A trailing
    ``Linear`` is rebuilt with ``in_features`` equal to the flattened input size.

    Parameters
    ----------
    layer : torch.nn.Module
        The original trailing layer.
    input_shape : tuple of int
        The shape of the tensor now entering the layer.

    Returns
    -------
    torch.nn.Module or None
        The re-sized layer, or ``None`` if the type is not handled (a warning is emitted by the caller).
    """

    if isinstance(layer, nn.Conv2d) :
        kernel_h, kernel_w = layer.kernel_size
        # Match each kernel dimension that used to span its axis (size > 1) to the new axis length.
        new_kernel = (
            input_shape[2] if kernel_h > 1 else kernel_h,
            input_shape[3] if kernel_w > 1 else kernel_w,
        )
        new_conv = nn.Conv2d(
            layer.in_channels, layer.out_channels, new_kernel,
            stride = layer.stride, padding = layer.padding, dilation = layer.dilation,
            groups = layer.groups, bias = layer.bias is not None,
        )
        return new_conv

    if isinstance(layer, nn.Linear) :
        new_in_features = 1
        for dimension in input_shape[1 :] :
            new_in_features *= int(dimension)
        return nn.Linear(new_in_features, layer.out_features, bias = layer.bias is not None)

    return None


def _resize_trailing_classifier(model : nn.Module, dummy_input : torch.Tensor) -> None :
    """
    Re-size the trailing classification layer after an overlapping-pooling swap changed the pooled length.

    Parameters
    ----------
    model : torch.nn.Module
        The pooling-swapped braindecode model (modified in place).
    dummy_input : torch.Tensor
        A batch-shaped probe input.
    """

    name, layer = _find_final_layer(model)
    if layer is None :
        warnings.warn(
            "an overlapping pooling layer was replaced but no trailing classification layer could be found to "
            "re-size; the model output may have the wrong shape.",
            stacklevel = 3,
        )
        return

    input_shape = _probe_input_shape(model, dummy_input, layer)
    if input_shape is None :
        warnings.warn(
            f"could not probe the input shape of the trailing layer {name!r}; the model output may have the wrong "
            "shape after the overlapping-pooling swap.",
            stacklevel = 3,
        )
        return

    new_layer = _rebuild_final_layer(layer, input_shape)
    if new_layer is None :
        warnings.warn(
            f"the trailing layer {name!r} ({type(layer).__name__}) could not be re-sized automatically after the "
            "overlapping-pooling swap; the model output may have the wrong shape.",
            stacklevel = 3,
        )
        return

    _set_submodule(model, name, new_layer)


def _infer_dummy_input(config : braindecode_wrapper_config, model : nn.Module, batch_size : int = 2) -> torch.Tensor :
    """
    Build a probe input for the trailing-layer re-sizing dummy forward.

    Uses ``config.example_input_shape`` when given (taken as the full shape, batch included), otherwise the number
    of channels and time samples exposed by the braindecode model (``n_chans`` / ``n_times``) or its ``model_kwargs``.

    Parameters
    ----------
    config : braindecode_wrapper_config
        The wrapper configuration.
    model : torch.nn.Module
        The built braindecode model.
    batch_size : int, default 2
        Batch size of the probe input (when the shape is inferred).

    Returns
    -------
    torch.Tensor
        A zero tensor of the probe shape.
    """

    if config.example_input_shape is not None :
        return torch.zeros(tuple(config.example_input_shape))

    def _first_available(*names) :
        for source in (model, config.model_kwargs) :
            for name in names :
                value = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
                if value is not None :
                    return value
        return None

    n_chans = _first_available("n_chans", "in_chans", "n_channels")
    n_times = _first_available("n_times", "input_window_samples", "n_times_input")

    if n_chans is None or n_times is None :
        raise ValueError(
            "cannot infer a probe input shape for the trailing-classifier re-sizing; set 'example_input_shape' in "
            "the braindecode_wrapper_config."
        )

    return torch.zeros(batch_size, int(n_chans), int(n_times))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Public wrapper

class braindecode_pooling_wrapper(nn.Module) :
    """
    A braindecode model whose pooling layers are replaced by a :mod:`jade.model.pooling_1d` operator.

    The wrapper builds the braindecode model named by the config, swaps every pooling layer for the chosen jade
    operator (keeping each layer's decimation factor), optionally drops the post-pool nonlinearity, and re-sizes the
    trailing classifier when an overlapping pooling was replaced. It then behaves like any other jade model: it is an
    :class:`torch.nn.Module` with :meth:`forward` and a :meth:`classify` helper, so :mod:`jade.training` can consume it
    unchanged.

    Parameters
    ----------
    config : braindecode_wrapper_config or dict
        The wrapper configuration (a dict with ``name = 'braindecode_wrapper'`` is promoted through
        :func:`~jade.config.model.get_model_config`).

    Attributes
    ----------
    model : torch.nn.Module
        The wrapped braindecode model (with the swapped pooling layers).
    swaps : list of dict
        Information about each replaced pooling layer.
    dropped_pool_nonlin : list of str
        The post-pool nonlinearity modules replaced by identity (empty when ``keep_pool_nonlin`` is ``True``).
    """

    def __init__(self, config) :
        super().__init__()

        # Accept a dict and promote it to the proper config (an already-built config is returned untouched).
        config = get_model_config(config)
        if not isinstance(config, braindecode_wrapper_config) :
            raise TypeError(f"braindecode_pooling_wrapper expects a braindecode_wrapper_config, got {type(config)}")
        self.config = config

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Build the braindecode model (braindecode validates its own kwargs).

        model_class = _get_braindecode_model_class(config.braindecode_model_name)
        model = model_class(**config.model_kwargs)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Swap the pooling layers.

        self.swaps, any_overlapping = _replace_pooling_layers(model, config.target_pooling_config, config.pooling_factor_source)

        # (OPTIONAL) Drop the post-pool nonlinearity.
        self.dropped_pool_nonlin = _drop_pool_nonlin(model) if not config.keep_pool_nonlin else []

        # (OPTIONAL) Re-size the trailing classifier if an overlapping pooling was replaced (its output length changed).
        if any_overlapping :
            _resize_trailing_classifier(model, _infer_dummy_input(config, model))

        self.model = model


    def forward(self, x : torch.Tensor) -> torch.Tensor :
        """
        Forward pass, delegated to the wrapped braindecode model.

        Parameters
        ----------
        x : torch.Tensor
            Input in the layout expected by the braindecode model (typically ``(B, n_chans, n_times)``).

        Returns
        -------
        torch.Tensor
            The braindecode model output (typically ``(B, n_outputs)``).
        """

        return self.model(x)


    def classify(self, x : torch.Tensor, return_as_index : bool = True) -> torch.Tensor :
        """
        Classify a batch, returning the predicted class indices by default.

        Provided for parity with :meth:`~jade.model.EEGNet.EEGNet_Classifier.classify` so that
        :mod:`jade.training.metrics` can use it directly.

        Parameters
        ----------
        x : torch.Tensor
            Input batch.
        return_as_index : bool, default True
            If True return the ``argmax`` class index; otherwise return the raw model output.

        Returns
        -------
        torch.Tensor
            The predicted class indices (or the raw output when ``return_as_index`` is False).
        """

        output = self.model(x)
        if return_as_index :
            return torch.argmax(output, dim = 1)
        return output
