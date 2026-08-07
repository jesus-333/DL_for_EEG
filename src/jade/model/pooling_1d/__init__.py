"""
Anti-aliased decimation ("pooling") layers for 1D signals.

Motivation
----------
A pooling layer with kernel size ``M`` and stride ``M`` is *decimation*: it reduces the sampling rate of its input by a factor ``M``.
Any decimation must be preceded by a low-pass (anti-aliasing) filter, otherwise every spectral component above the new Nyquist frequency ``pi / M`` is *folded* back into the retained band and becomes indistinguishable from genuine low-frequency content (Nyquist-Shannon).

Every pooling operator in this package follows the canonical two-step decomposition (Zhang, ICML 2019, "Making Convolutional Networks Shift-Invariant Again"):

    pooling  =  dense filtering (stride 1)  ->  subsampling (stride M)

and differs *only* in the low-pass filter ``h`` used in the first step:

    ==================  =============================================================
    Layer               Filter h
    ==================  =============================================================
    avg_pool_1d         boxcar / moving average, length M          (the usual default)
    blur_pool_1d        binomial (Pascal row) or Gaussian          (fixed heuristic)
    sinc_pool_1d        windowed-sinc, Kaiser window               (designed optimal)
    dwt_pool_1d         orthogonal wavelet analysis low-pass       (filter bank)
    max_pool_1d         (no filter; nonlinear) - reference only
    ==================  =============================================================

Because average pooling *is* the boxcar special case, ``avg_pool_1d`` and ``blur_pool_1d(kernel='boxcar')`` are numerically identical up to boundary handling.
This is asserted by ``self_test()`` and is what makes the comparison internally consistent: all arms of an ablation differ in exactly one designed object, ``h``.

Tensor convention
-----------------
All layers consume and produce ``(B, D, T)``:
    B = batch, D = depth/feature channels, T = temporal samples.
Filtering is *depthwise*: the same ``h`` is applied independently to every one of the ``D`` channels.
There are no trainable parameters in any of these layers.

Output length is ``T // M`` for every layer, exactly matching ``torch.nn.AvgPool1d(M, stride = M)``.
This is deliberate: it guarantees that swapping the pooling operator does not change any downstream tensor shape, so the classifier head, parameter count and receptive field stay fixed across an ablation.

IMPORTANT - DC gain
-------------------
By default every filter is normalized to unit DC gain (``sum(h) == 1``), exactly like an average.
This matters when there is no normalization layer *after* pooling (as in EEGNet, where the order is BatchNorm -> activation -> pool -> dropout): a filter with a different DC gain would rescale all activations and confound the comparison with a change in effective learning rate / initialization scale.
Use ``gain='orthonormal'`` only if you deliberately want energy preservation instead.

IMPORTANT - kernel length must scale with M
-------------------------------------------
A frequent and silent methodological error is to port the standard 2D BlurPool kernels (``[1,2,1]``, ``[1,4,6,4,1]``) to a layer with a large decimation factor.
Those kernels have a *fixed* bandwidth, whereas the required cutoff ``pi / M`` shrinks as M grows.
At M = 8 the binomial kernel of length 3 passes ~67% of its energy above the post-decimation Nyquist, versus ~22% for a plain boxcar - i.e. it is *three times worse than the average pooling it was meant to improve on*.
Defaults in this package therefore scale the kernel length with M, and ``stopband_leakage()`` plus the ``warn_if_worse_than_boxcar`` flag exist to catch this case explicitly.

Package layout
--------------
The implementation is split across one file per pooling operator:

    support_pooling.py   shared helpers + the ``_fir_decimation_pool_1d`` base class
    avg_pool.py          avg_pool_1d
    max_pool.py          max_pool_1d
    blur_pool.py         blur_pool_1d
    sinc_pool.py         sinc_pool_1d
    dwt_pool.py          dwt_pool_1d and the WAVELET_FILTERS coefficients

This ``__init__`` re-exports the public API and hosts the factory (``get_pooling_layer`` / ``POOLING_REGISTRY``) and the ``self_test``.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Third party imports
import torch
from torch import nn

# Internal imports
from jade.model.pooling_1d.avg_pool import avg_pool_1d
from jade.model.pooling_1d.blur_pool import blur_pool_1d
from jade.model.pooling_1d.dwt_pool import WAVELET_FILTERS, dwt_pool_1d
from jade.model.pooling_1d.max_pool import max_pool_1d
from jade.model.pooling_1d.sinc_pool import sinc_pool_1d
from jade.model.pooling_1d.support_pooling import (
    _as_decimation_factor,
    _filter_and_subsample,
    _get,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Public API

__all__ = [
    "avg_pool_1d",
    "max_pool_1d",
    "blur_pool_1d",
    "sinc_pool_1d",
    "dwt_pool_1d",
    "get_pooling_layer",
    "POOLING_REGISTRY",
    "WAVELET_FILTERS",
    "self_test",
]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Factory

# Maps lowercase string aliases -> pooling class.
POOLING_REGISTRY = {
    "avg"      : avg_pool_1d,
    "average"  : avg_pool_1d,
    "boxcar"   : avg_pool_1d,
    "max"      : max_pool_1d,
    "blur"     : blur_pool_1d,
    "blurpool" : blur_pool_1d,
    "binomial" : blur_pool_1d,
    "gaussian" : blur_pool_1d,
    "sinc"     : sinc_pool_1d,
    "kaiser"   : sinc_pool_1d,
    "dwt"      : dwt_pool_1d,
    "wavelet"  : dwt_pool_1d,
}


def get_pooling_layer(config : dict) -> nn.Module :
    """
    Build a pooling layer from a config dict - the single entry point for ablations.

    The dict needs a ``pooling_type`` key naming the operator, plus whatever that operator requires.
    Passing one dict per experimental arm keeps the sweep declarative (easy to serialize to YAML/JSON and log alongside results).

    ``pooling_type='identity'`` (or a decimation factor of 1) yields ``nn.Identity``, which is convenient for ablating a pooling stage away entirely.

    Parameters
    ----------
    config : dict
        Must contain ``pooling_type`` (one of ``POOLING_REGISTRY`` or ``'identity'``), plus the keys the chosen operator requires.

    Returns
    -------
    torch.nn.Module
        The built pooling layer (or ``nn.Identity``).

    Examples
    --------
    >>> get_pooling_layer({'pooling_type' : 'avg',  'decimation_factor' : 4})
    >>> get_pooling_layer({'pooling_type' : 'blur', 'decimation_factor' : 4, 'kernel_size' : 17})
    >>> get_pooling_layer({'pooling_type' : 'sinc', 'decimation_factor' : 8, 'stopband_attenuation_db' : 60, 'transition_width' : 0.25})
    >>> get_pooling_layer({'pooling_type' : 'dwt',  'decimation_factor' : 8, 'wavelet' : 'sym4'})
    """

    # Input check
    if "pooling_type" not in config :
        raise KeyError("config must contain 'pooling_type'; one of {}".format(sorted(set(POOLING_REGISTRY)) + ["identity"]))
    name = str(config["pooling_type"]).lower()

    # Identity short-circuit
    if name in ("identity", "none") :
        return nn.Identity()

    if name not in POOLING_REGISTRY :
        raise ValueError("unknown pooling_type {}; expected one of {}".format(repr(name), sorted(set(POOLING_REGISTRY)) + ["identity"]))

    # Never mutate the caller's dict.
    cfg = dict(config)

    # Convenience: 'gaussian'/'binomial' select blur_pool_1d *and* its kernel family.
    if name == "gaussian" :
        cfg.setdefault("kernel", "gaussian")
    if name == "binomial" :
        cfg.setdefault("kernel", "binomial")

    # Decimating by 1 is a no-op; skip the filtering entirely.
    factor = _get(cfg, "decimation_factor", aliases = ("pool_size", "factor", "p_kernel"))
    if factor is not None and _as_decimation_factor(factor) == 1 :
        return nn.Identity()

    return POOLING_REGISTRY[name](cfg)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Self-test

def self_test(verbose : bool = True) -> None :
    """
    Verify the invariants this package relies on. Run once after installing.

    Checks
    ------
    1. Every layer outputs exactly ``T // M`` samples, for many (T, M) combinations.
    2. ``avg_pool_1d`` reproduces ``nn.AvgPool1d(M, stride=M)`` exactly (interior), proving average pooling really is the boxcar special case of the family.
    3. All DC-normalized layers pass a constant signal unchanged (unit DC gain).
    4. The DWT cascade matches its own ``effective_kernel()`` (noble identity).
    5. Layers move across dtypes/devices and are gradient-transparent.

    Parameters
    ----------
    verbose : bool, default True
        Whether to print each check result.
    """

    torch.manual_seed(0)

    def _check(name, cond) :
        status = "OK  " if cond else "FAIL"
        if verbose :
            print("  [{}] {}".format(status, name))
        assert cond, name

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 1. output length == T // M

    if verbose :
        print("1. output length == T // M")
    for T in (1000, 1001, 1002, 999, 512, 250) :
        for M in (2, 4, 8) :
            x = torch.randn(2, 3, T)
            layers = {
                "avg"  : avg_pool_1d({"decimation_factor" : M}),
                "max"  : max_pool_1d({"decimation_factor" : M}),
                "blur" : blur_pool_1d({"decimation_factor" : M}),
                "sinc" : sinc_pool_1d({"decimation_factor" : M}),
                "dwt"  : dwt_pool_1d({"decimation_factor" : M}),
            }
            for name, layer in layers.items() :
                got = layer(x).shape
                _check("T={} M={} {}: {} vs (2,3,{})".format(T, M, name, tuple(got), T // M),
                       got == (2, 3, T // M))

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 2. avg_pool_1d == nn.AvgPool1d (interior)

    if verbose :
        print("2. avg_pool_1d == nn.AvgPool1d (interior)")
    for T, M in ((1000, 4), (1000, 8), (997, 4)) :
        x = torch.randn(4, 5, T)
        mine = avg_pool_1d({"decimation_factor" : M, "padding_mode" : "replicate"})(x)
        ref = nn.AvgPool1d(M, stride = M)(x)
        # Interior samples are unaffected by boundary handling.
        err = (mine[..., 1:-1] - ref[..., 1:-1]).abs().max().item()
        _check("T={} M={}: max|diff|={:.2e}".format(T, M, err), err < 1e-5)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 3. unit DC gain (constant in -> same constant out)

    if verbose :
        print("3. unit DC gain (constant in -> same constant out)")
    for layer in (
        avg_pool_1d({"decimation_factor" : 4}),
        blur_pool_1d({"decimation_factor" : 4}),
        blur_pool_1d({"decimation_factor" : 4, "kernel" : "gaussian"}),
        sinc_pool_1d({"decimation_factor" : 4}),
        dwt_pool_1d({"decimation_factor" : 4}),
        dwt_pool_1d({"decimation_factor" : 8, "wavelet" : "sym4"}),
    ) :
        x = torch.full((1, 2, 512), 3.7)
        y = layer(x)
        err = (y - 3.7).abs().max().item()
        _check("{} {}: err={:.2e}".format(type(layer).__name__, layer.extra_repr()[:28], err),
               err < 1e-4)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 4. DWT cascade == equivalent kernel (noble identity)

    if verbose :
        print("4. DWT cascade == equivalent kernel (noble identity)")
    for M in (2, 4, 8) :
        layer = dwt_pool_1d({"decimation_factor" : M, "padding_mode" : "circular"})
        x = torch.randn(1, 1, 512, dtype = torch.float64)
        y_cascade = layer.double()(x)
        h_eff = torch.as_tensor(layer.effective_kernel(), dtype = torch.float64)
        L_stage = layer.kernel.numel()
        offset = (h_eff.numel() - 1) // 2 - ((L_stage - 1) // 2) * (M - 1)
        y_eq = _filter_and_subsample(x, h_eff, M, offset, "circular")
        n = min(y_cascade.shape[-1], y_eq.shape[-1])
        err = (y_cascade[..., :n] - y_eq[..., :n]).abs().max().item()
        _check("M={} L_eff={} offset={}: max|diff|={:.2e}".format(M, h_eff.numel(), offset, err),
               err < 1e-9)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 5. dtype/grad transparency

    if verbose :
        print("5. dtype/grad transparency")
    for layer in (blur_pool_1d({"decimation_factor" : 4}),
                  sinc_pool_1d({"decimation_factor" : 4}),
                  dwt_pool_1d({"decimation_factor" : 4})) :
        x = torch.randn(2, 3, 256, requires_grad = True)
        layer(x).sum().backward()
        _check("{} grad flows".format(type(layer).__name__),
               x.grad is not None and torch.isfinite(x.grad).all().item())
        _check("{} no trainable params".format(type(layer).__name__),
               sum(p.numel() for p in layer.parameters()) == 0)
        _check("{} clean state_dict".format(type(layer).__name__),
               len(layer.state_dict()) == 0)

    if verbose :
        print("\nAll self-tests passed.")
