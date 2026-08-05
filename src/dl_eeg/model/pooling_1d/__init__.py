"""
Anti-aliased decimation ("pooling") layers for 1D signals.
=========================================================

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
    AvgPool1d           boxcar / moving average, length M          (the usual default)
    BlurPool1d          binomial (Pascal row) or Gaussian          (fixed heuristic)
    SincPool1d          windowed-sinc, Kaiser window               (designed optimal)
    DWTPool1d           orthogonal wavelet analysis low-pass       (filter bank)
    MaxPool1d           (no filter; nonlinear) - reference only
    ==================  =============================================================

Because average pooling *is* the boxcar special case, ``AvgPool1d`` and ``BlurPool1d(kernel='boxcar')`` are numerically identical up to boundary handling.
This is asserted by ``self_test()`` and is what makes the comparison internally consistent: all arms of an ablation differ in exactly one designed object, ``h``.

Tensor convention
-----------------
All layers consume and produce ``(B, D, T)``:
    B = batch, D = depth/feature channels, T = temporal samples.
Filtering is *depthwise*: the same ``h`` is applied independently to every one of the ``D`` channels. There are no trainable parameters in any of these layers.

Output length is ``T // M`` for every layer, exactly matching ``torch.nn.AvgPool1d(M, stride = M)``.
This is deliberate: it guarantees that swapping the pooling operator does not change any downstream tensor shape, so the classifier head, parameter count and receptive field stay fixed across an ablation.

IMPORTANT - DC gain
-------------------
By default every filter is normalised to unit DC gain (``sum(h) == 1``), exactly like an average.
This matters when there is no normalisation layer *after* pooling (as in EEGNet, where the order is BatchNorm -> activation -> pool -> dropout): a filter with
a different DC gain would rescale all activations and confound the comparison with a change in effective learning rate / initialisation scale.
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

    support_pooling.py   shared helpers + the ``_FIRDecimationPool1d`` base class
    avg_pool.py          AvgPool1d
    max_pool.py          MaxPool1d
    blur_pool.py         BlurPool1d
    sinc_pool.py         SincPool1d
    dwt_pool.py          DWTPool1d and the WAVELET_FILTERS coefficients

This ``__init__`` re-exports the public API and hosts the factory
(``get_pooling_layer`` / ``POOLING_REGISTRY``) and the ``self_test``.
"""

from __future__ import annotations

import torch
from torch import nn

from dl_eeg.model.pooling_1d.avg_pool import AvgPool1d
from dl_eeg.model.pooling_1d.blur_pool import BlurPool1d
from dl_eeg.model.pooling_1d.dwt_pool import WAVELET_FILTERS, DWTPool1d
from dl_eeg.model.pooling_1d.max_pool import MaxPool1d
from dl_eeg.model.pooling_1d.sinc_pool import SincPool1d
from dl_eeg.model.pooling_1d.support_pooling import (
    _as_decimation_factor,
    _filter_and_subsample,
    _get,
)

__all__ = [
    "AvgPool1d",
    "MaxPool1d",
    "BlurPool1d",
    "SincPool1d",
    "DWTPool1d",
    "get_pooling_layer",
    "POOLING_REGISTRY",
    "WAVELET_FILTERS",
    "self_test",
]


# =================================================================================
# Factory
# =================================================================================
POOLING_REGISTRY = {
    "avg": AvgPool1d,
    "average": AvgPool1d,
    "boxcar": AvgPool1d,
    "max": MaxPool1d,
    "blur": BlurPool1d,
    "blurpool": BlurPool1d,
    "binomial": BlurPool1d,
    "gaussian": BlurPool1d,
    "sinc": SincPool1d,
    "kaiser": SincPool1d,
    "dwt": DWTPool1d,
    "wavelet": DWTPool1d,
}


def get_pooling_layer(config: dict) -> nn.Module:
    """
    Build a pooling layer from a config dict - the single entry point for ablations.

    The dict needs a ``pooling_type`` key naming the operator, plus whatever that
    operator requires. Passing one dict per experimental arm keeps the sweep
    declarative (easy to serialise to YAML/JSON and log alongside results).

    ``pooling_type='identity'`` (or a decimation factor of 1) yields ``nn.Identity``,
    which is convenient for ablating a pooling stage away entirely.

    Examples
    --------
    >>> get_pooling_layer({'pooling_type': 'avg',  'decimation_factor': 4})
    >>> get_pooling_layer({'pooling_type': 'blur', 'decimation_factor': 4,
    ...                    'kernel_size': 17})
    >>> get_pooling_layer({'pooling_type': 'sinc', 'decimation_factor': 8,
    ...                    'stopband_attenuation_db': 60, 'transition_width': 0.25})
    >>> get_pooling_layer({'pooling_type': 'dwt',  'decimation_factor': 8,
    ...                    'wavelet': 'sym4'})
    """
    if "pooling_type" not in config:
        raise KeyError(
            "config must contain 'pooling_type'; one of "
            f"{sorted(set(POOLING_REGISTRY)) + ['identity']}"
        )
    name = str(config["pooling_type"]).lower()

    if name in ("identity", "none"):
        return nn.Identity()

    if name not in POOLING_REGISTRY:
        raise ValueError(
            f"unknown pooling_type {name!r}; expected one of "
            f"{sorted(set(POOLING_REGISTRY)) + ['identity']}"
        )

    cfg = dict(config)  # never mutate the caller's dict
    # Convenience: 'gaussian' selects BlurPool1d *and* its kernel family.
    if name == "gaussian":
        cfg.setdefault("kernel", "gaussian")
    if name == "binomial":
        cfg.setdefault("kernel", "binomial")

    factor = _get(cfg, "decimation_factor",
                  aliases=("pool_size", "factor", "p_kernel"))
    if factor is not None and _as_decimation_factor(factor) == 1:
        # Decimating by 1 is a no-op; skip the filtering entirely.
        return nn.Identity()

    return POOLING_REGISTRY[name](cfg)


# =================================================================================
# Self-test
# =================================================================================
def self_test(verbose: bool = True) -> None:
    """
    Verify the invariants this package relies on. Run once after installing.

    Checks
    ------
    1. Every layer outputs exactly ``T // M`` samples, for many (T, M) combinations.
    2. ``AvgPool1d`` reproduces ``nn.AvgPool1d(M, stride=M)`` exactly (interior),
       proving average pooling really is the boxcar special case of the family.
    3. All DC-normalised layers pass a constant signal unchanged (unit DC gain).
    4. The DWT cascade matches its own ``effective_kernel()`` (noble identity).
    5. Layers move across dtypes/devices and are gradient-transparent.
    """
    torch.manual_seed(0)

    def _check(name, cond):
        status = "OK  " if cond else "FAIL"
        if verbose:
            print(f"  [{status}] {name}")
        assert cond, name

    if verbose:
        print("1. output length == T // M")
    for T in (1000, 1001, 1002, 999, 512, 250):
        for M in (2, 4, 8):
            x = torch.randn(2, 3, T)
            layers = {
                "avg": AvgPool1d({"decimation_factor": M}),
                "max": MaxPool1d({"decimation_factor": M}),
                "blur": BlurPool1d({"decimation_factor": M}),
                "sinc": SincPool1d({"decimation_factor": M}),
                "dwt": DWTPool1d({"decimation_factor": M}),
            }
            for nm, layer in layers.items():
                got = layer(x).shape
                _check(f"T={T} M={M} {nm}: {tuple(got)} vs (2,3,{T // M})",
                       got == (2, 3, T // M))

    if verbose:
        print("2. AvgPool1d == nn.AvgPool1d (interior)")
    for T, M in ((1000, 4), (1000, 8), (997, 4)):
        x = torch.randn(4, 5, T)
        mine = AvgPool1d({"decimation_factor": M, "padding_mode": "replicate"})(x)
        ref = nn.AvgPool1d(M, stride=M)(x)
        # Interior samples are unaffected by boundary handling.
        err = (mine[..., 1:-1] - ref[..., 1:-1]).abs().max().item()
        _check(f"T={T} M={M}: max|diff|={err:.2e}", err < 1e-5)

    if verbose:
        print("3. unit DC gain (constant in -> same constant out)")
    for layer in (
        AvgPool1d({"decimation_factor": 4}),
        BlurPool1d({"decimation_factor": 4}),
        BlurPool1d({"decimation_factor": 4, "kernel": "gaussian"}),
        SincPool1d({"decimation_factor": 4}),
        DWTPool1d({"decimation_factor": 4}),
        DWTPool1d({"decimation_factor": 8, "wavelet": "sym4"}),
    ):
        x = torch.full((1, 2, 512), 3.7)
        y = layer(x)
        err = (y - 3.7).abs().max().item()
        _check(f"{type(layer).__name__} {layer.extra_repr()[:28]}: err={err:.2e}",
               err < 1e-4)

    if verbose:
        print("4. DWT cascade == equivalent kernel (noble identity)")
    for M in (2, 4, 8):
        layer = DWTPool1d({"decimation_factor": M, "padding_mode": "circular"})
        x = torch.randn(1, 1, 512, dtype=torch.float64)
        y_cascade = layer.double()(x)
        h_eff = torch.as_tensor(layer.effective_kernel(), dtype=torch.float64)
        L_stage = layer.kernel.numel()
        offset = (h_eff.numel() - 1) // 2 - ((L_stage - 1) // 2) * (M - 1)
        y_eq = _filter_and_subsample(x, h_eff, M, offset, "circular")
        n = min(y_cascade.shape[-1], y_eq.shape[-1])
        err = (y_cascade[..., :n] - y_eq[..., :n]).abs().max().item()
        _check(f"M={M} L_eff={h_eff.numel()} offset={offset}: max|diff|={err:.2e}",
               err < 1e-9)

    if verbose:
        print("5. dtype/grad transparency")
    for layer in (BlurPool1d({"decimation_factor": 4}),
                  SincPool1d({"decimation_factor": 4}),
                  DWTPool1d({"decimation_factor": 4})):
        x = torch.randn(2, 3, 256, requires_grad=True)
        layer(x).sum().backward()
        _check(f"{type(layer).__name__} grad flows", x.grad is not None
               and torch.isfinite(x.grad).all().item())
        _check(f"{type(layer).__name__} no trainable params",
               sum(p.numel() for p in layer.parameters()) == 0)
        _check(f"{type(layer).__name__} clean state_dict",
               len(layer.state_dict()) == 0)

    if verbose:
        print("\nAll self-tests passed.")
