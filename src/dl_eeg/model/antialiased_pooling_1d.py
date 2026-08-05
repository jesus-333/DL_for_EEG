"""
Anti-aliased decimation ("pooling") layers for 1D signals.
=========================================================

Motivation
----------
A pooling layer with kernel size ``M`` and stride ``M`` is *decimation*: it reduces the sampling rate of its input by a factor ``M``. 
Any decimation must be preceded by a low-pass (anti-aliasing) filter, otherwise every spectral component above the new Nyquist frequency ``pi / M`` is *folded* back into the retained band and becomes indistinguishable from genuine low-frequency content (Nyquist-Shannon).

Every pooling operator in this file follows the canonical two-step decomposition (Zhang, ICML 2019, "Making Convolutional Networks Shift-Invariant Again"):

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
Defaults in this file therefore scale the kernel length with M, and ``stopband_leakage()`` plus the ``warn_if_worse_than_boxcar`` flag exist to catch this case explicitly.

"""

from __future__ import annotations

import math
import warnings
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

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


# ---------------------------------------------------------------------------------
# Wavelet filter bank coefficients
# ---------------------------------------------------------------------------------
# These are the *reconstruction* low-pass (scaling) filters ``rec_lo`` in the standard
# orthonormal convention, i.e. they satisfy
#       sum(h)      = sqrt(2)                      (DC gain, unit-norm convention)
#       sum(h**2)   = 1                            (unit energy)
#       sum_n h[n] * h[n + 2k] = 0  for k != 0     (double-shift orthogonality)
# All five sets below were verified numerically against those three conditions to
# ~1e-12. The analysis (decomposition) filter used by the DWT is the time reverse,
# ``dec_lo = rec_lo[::-1]``; see ``DWTPool1d`` for how that interacts with the fact
# that ``F.conv1d`` computes cross-correlation rather than convolution.
WAVELET_FILTERS: Dict[str, Sequence[float]] = {
    # Haar / db1: the shortest orthogonal wavelet. Its low-pass is a 2-tap boxcar,
    # so 'haar' DWT pooling is a very weak anti-aliasing filter (included as a
    # deliberate lower bound / sanity arm, not as a serious candidate).
    "haar": [0.7071067811865476, 0.7071067811865476],
    "db1": [0.7071067811865476, 0.7071067811865476],
    "db2": [
        0.48296291314469025, 0.836516303737469,
        0.22414386804185735, -0.12940952255092145,
    ],
    "db3": [
        0.3326705529509569, 0.8068915093133388, 0.4598775021193313,
        -0.13501102001039084, -0.08544127388224149, 0.035226291882100656,
    ],
    # db4 (8 taps, 4 vanishing moments) is the usual default: a good compromise
    # between stopband attenuation and kernel length.
    "db4": [
        0.23037781330885523, 0.7148465705525415, 0.6308807679295904,
        -0.02798376941698385, -0.18703481171888114, 0.030841381835986965,
        0.032883011666982945, -0.010597401784997278,
    ],
    # sym4 is the "least asymmetric" 8-tap orthogonal wavelet: nearly linear phase,
    # which means less temporal smearing of event-related features than db4.
    # Preferable when the *timing* of a transient matters (ERPs, QRS complexes).
    "sym4": [
        0.032223100604042702, -0.012603967262037833, -0.09921954357684722,
        0.29785779560527736, 0.8037387518059161, 0.49761866763201545,
        -0.02963552764599851, -0.07576571478927333,
    ],
}


# ---------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------
def _get(config: dict, key: str, default=None, aliases: Sequence[str] = ()):
    """
    Fetch ``key`` from ``config``, accepting alternative spellings.

    Rationale: EEGNet-style configs use names like ``p_kernel_1``; this lets the same
    layer be dropped into an existing config dict without renaming anything.
    """
    for k in (key, *aliases):
        if k in config and config[k] is not None:
            return config[k]
    return default


def _as_decimation_factor(value) -> int:
    """
    Normalise the decimation factor to a positive int.

    Accepts ``4``, ``(4,)``, ``(1, 4)`` or ``[1, 4]``. The ``(1, M)`` form is accepted
    because EEGNet configs specify 2D pooling kernels ``(1, M)`` that act on the time
    axis only; this way an existing config value can be passed straight through.
    """
    if isinstance(value, (tuple, list)):
        vals = [int(v) for v in value]
        if len(vals) == 1:
            factor = vals[0]
        elif len(vals) == 2:
            # (height, width) with height == 1 -> purely temporal pooling
            if vals[0] != 1:
                raise ValueError(
                    f"2D pooling kernel {tuple(vals)} pools across the non-temporal "
                    "axis too; these 1D layers only decimate along time. Pass an int "
                    "or (1, M)."
                )
            factor = vals[1]
        else:
            raise ValueError(f"cannot interpret {value!r} as a 1D decimation factor")
    else:
        factor = int(value)
    if factor < 1:
        raise ValueError(f"decimation factor must be >= 1, got {factor}")
    return factor


def _binomial_kernel(length: int) -> np.ndarray:
    """
    Binomial ("Pascal row") kernel of the requested length, normalised to sum 1.

    Built by repeatedly convolving ``[1, 1]`` with itself, which is both the
    definition of a binomial coefficient row and the reason these kernels converge
    to a Gaussian as ``length`` grows (central limit theorem applied to the kernel).

    Frequency response has the closed form  |H(w)| = |cos(w / 2)| ** (length - 1):
    monotonically decreasing on [0, pi], with NO sidelobes and no passband ripple -
    the qualitative improvement over the boxcar. The price is a slow roll-off.
    """
    if length < 1:
        raise ValueError("binomial kernel length must be >= 1")
    k = np.ones(1, dtype=np.float64)
    for _ in range(length - 1):
        k = np.convolve(k, np.ones(2, dtype=np.float64))
    return k / k.sum()


def _gaussian_kernel(length: int, sigma: Optional[float] = None) -> np.ndarray:
    """
    Truncated Gaussian kernel, normalised to sum 1.

    If ``sigma`` is not given it is set so that the kernel is truncated at +-3 sigma,
    which keeps the truncation error (and hence the spurious sidelobes it creates)
    negligible. A Gaussian has a Gaussian frequency response: strictly positive, no
    sidelobes at all, but the slowest roll-off of the fixed kernels here.
    """
    if length < 1:
        raise ValueError("gaussian kernel length must be >= 1")
    if sigma is None:
        sigma = max(length / 6.0, 1e-3)  # +-3 sigma inside the support
    n = np.arange(length, dtype=np.float64) - (length - 1) / 2.0
    k = np.exp(-0.5 * (n / sigma) ** 2)
    return k / k.sum()


def _upsample_zeros(h: np.ndarray, factor: int) -> np.ndarray:
    """
    Insert ``factor - 1`` zeros between consecutive taps of ``h``.

    This implements the "noble identity" used to collapse a multi-stage decimating
    filter bank into one equivalent single-stage filter (see
    ``DWTPool1d.effective_kernel``). Upsampling a filter by L compresses its
    frequency response by L along the frequency axis.
    """
    if factor == 1:
        return h.copy()
    out = np.zeros((len(h) - 1) * factor + 1, dtype=h.dtype)
    out[::factor] = h
    return out


# =================================================================================
# Base class: dense FIR filtering followed by subsampling
# =================================================================================
class _FIRDecimationPool1d(nn.Module):
    """
    Base class implementing ``pooling = dense FIR filtering -> subsampling``.

    Subclasses only have to build the low-pass kernel; everything else (padding,
    depthwise application, phase/offset selection, output-length guarantees and the
    frequency-domain analysis utilities) lives here.

    Shared configuration keys
    -------------------------
    decimation_factor : int or (1, int)        [required]
        The factor M by which the time axis is reduced. Aliases accepted:
        ``pool_size``, ``factor``, ``p_kernel``. NOTE: ``kernel_size`` is
        deliberately NOT an alias here - it always means the *filter length*.
    padding_mode : str, default 'reflect'
        How to extend the signal so the filtered output keeps length T before
        subsampling. One of 'reflect', 'replicate', 'constant' (zeros), 'circular'.
        'reflect' is the default because EEG trials are short: zero-padding would
        inject an artificial step at both ends, whose broadband spectrum is precisely
        the high-frequency content we are trying to suppress.
    phase : {'center', 'first'} or int, default 'center'
        Which sample of each block of M to retain after filtering.
        'center' -> offset (M - 1) // 2, which reproduces the sample alignment of
        ``nn.AvgPool1d`` exactly (each output is centred on its input block).
        'first'  -> offset 0.
        Keep this fixed across an ablation: changing it shifts features in time.
    gain : {'dc', 'orthonormal', 'none'}, default 'dc'
        Kernel normalisation. 'dc' -> sum(h) = 1 (unit DC gain, comparable with an
        average); 'orthonormal' -> ||h||_2 = 1 (energy preserving); 'none' -> leave
        the kernel as constructed. Use 'dc' for a controlled comparison.
    warn_if_worse_than_boxcar : bool, default True
        Emit a warning at construction if the chosen filter passes *more* energy
        above the post-decimation Nyquist than a plain boxcar of length M would.
        This catches the "short BlurPool kernel at large M" trap described in the
        module docstring.
    """

    #: subclasses set this for nicer error messages / __repr__
    _kind: str = "fir"

    def __init__(self, config: dict):
        super().__init__()

        # -- decimation factor ----------------------------------------------------
        raw_factor = _get(
            config, "decimation_factor",
            aliases=("pool_size", "factor", "p_kernel"),
        )
        if raw_factor is None:
            raise KeyError(
                "config must contain 'decimation_factor' (or an alias: 'pool_size', "
                "'factor', 'p_kernel'). Note 'kernel_size' is NOT an alias: it means "
                "the filter length."
            )
        self.decimation_factor: int = _as_decimation_factor(raw_factor)

        # -- generic options ------------------------------------------------------
        self.padding_mode: str = _get(config, "padding_mode", "reflect")
        if self.padding_mode not in ("reflect", "replicate", "constant", "circular"):
            raise ValueError(f"unsupported padding_mode {self.padding_mode!r}")

        self.gain: str = _get(config, "gain", "dc")
        if self.gain not in ("dc", "orthonormal", "none"):
            raise ValueError(f"unsupported gain {self.gain!r}")

        phase = _get(config, "phase", "center")
        if phase == "center":
            # Matches nn.AvgPool1d: output n corresponds to input block [nM, nM+M).
            self.offset: int = (self.decimation_factor - 1) // 2
        elif phase == "first":
            self.offset = 0
        elif isinstance(phase, int):
            if not 0 <= phase < self.decimation_factor:
                raise ValueError(
                    f"integer phase must lie in [0, {self.decimation_factor}), got {phase}"
                )
            self.offset = phase
        else:
            raise ValueError(f"unsupported phase {phase!r}")

        # -- build the kernel (subclass responsibility) ---------------------------
        kernel = np.asarray(self._build_kernel(config), dtype=np.float64).ravel()
        if kernel.size == 0:
            raise ValueError("empty kernel")
        kernel = self._normalise(kernel)

        # Registered as a NON-persistent buffer: it follows .to(device)/.float()/
        # .half() like a parameter would, but is excluded from state_dict(). That
        # keeps checkpoints free of constants and avoids shape-mismatch errors when
        # reloading a run with a different kernel length.
        self.register_buffer(
            "kernel", torch.as_tensor(kernel, dtype=torch.float32), persistent=False
        )

        # -- sanity check against the boxcar baseline ----------------------------
        if _get(config, "warn_if_worse_than_boxcar", True) and self.decimation_factor > 1:
            self._check_against_boxcar()

    # -- subclass hook ------------------------------------------------------------
    def _build_kernel(self, config: dict) -> np.ndarray:
        raise NotImplementedError

    # -- normalisation ------------------------------------------------------------
    def _normalise(self, k: np.ndarray) -> np.ndarray:
        """Apply the requested gain convention (see the ``gain`` config key)."""
        if self.gain == "dc":
            s = k.sum()
            if abs(s) < 1e-12:
                raise ValueError(
                    "kernel has (near) zero DC gain and cannot be DC-normalised; "
                    "this is not a low-pass filter"
                )
            return k / s
        if self.gain == "orthonormal":
            return k / np.sqrt((k**2).sum())
        return k

    def _check_against_boxcar(self) -> None:
        """
        Warn if this filter is a *worse* anti-aliasing filter than plain averaging.

        A short fixed kernel (e.g. binomial [1,2,1]) has a bandwidth set by its own
        length, not by M. As M grows the required cutoff pi/M shrinks, and the short
        kernel eventually lets through more out-of-band energy than a boxcar of
        length M - which is what average pooling already does for free.
        """
        M = self.decimation_factor
        mine = self.stopband_leakage()
        box = _leakage_of(np.ones(M) / M, M)
        if mine > box:
            warnings.warn(
                f"{type(self).__name__}: this configuration is a WORSE anti-aliasing "
                f"filter than plain average pooling at M={M} "
                f"(stopband leakage {mine * 100:.2f}% vs boxcar {box * 100:.2f}%). "
                f"The kernel is too short for this decimation factor - the required "
                f"cutoff is pi/{M}. Increase the kernel length (a useful rule of "
                f"thumb is length >= 4*M+1) or use SincPool1d, which designs the "
                f"filter for the cutoff directly. Set "
                f"warn_if_worse_than_boxcar=False to silence this.",
                RuntimeWarning,
                stacklevel=3,
            )

    # -- the actual computation ---------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, D, T)

        Returns
        -------
        torch.Tensor of shape (B, D, T // M)
        """
        if x.dim() != 3:
            raise ValueError(f"expected a 3D tensor (B, D, T), got shape {tuple(x.shape)}")
        return _filter_and_subsample(
            x,
            kernel=self.kernel,
            factor=self.decimation_factor,
            offset=self.offset,
            padding_mode=self.padding_mode,
        )

    # -- analysis utilities (no autograd; for figures and tables) -----------------
    def effective_kernel(self) -> np.ndarray:
        """
        The single-stage impulse response equivalent to this layer, as a NumPy array.

        For the FIR layers this is just the kernel; ``DWTPool1d`` overrides it to
        collapse its multi-level cascade via the noble identity.
        """
        return self.kernel.detach().cpu().numpy().astype(np.float64)

    def frequency_response(self, n_fft: int = 8192) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discrete-time frequency response of the effective kernel.

        Returns
        -------
        w : ndarray, normalised angular frequency in [0, pi)
        H : complex ndarray, H(w)

        To convert to Hz at an input sampling rate ``fs``: ``f = w / (2 * pi) * fs``.
        """
        h = self.effective_kernel()
        H = np.fft.rfft(h, n=n_fft)
        w = np.linspace(0.0, np.pi, H.size, endpoint=True)
        return w, H

    def stopband_leakage(self, n_fft: int = 8192) -> float:
        """
        Fraction of the filter's energy that sits ABOVE the post-decimation Nyquist.

            leakage = int_{pi/M}^{pi} |H(w)|^2 dw  /  int_0^{pi} |H(w)|^2 dw

        This is the single most useful scalar figure of merit for the present study:
        it is exactly the portion of the response that will be folded back into the
        retained band by the subsequent subsampling. 0 = perfect anti-aliasing.

        Note this is a property of the *filter*, independent of the signal. A
        signal-dependent version (energy of the actual activations above pi/M) is the
        complementary per-layer diagnostic to log during training.
        """
        return _leakage_of(self.effective_kernel(), self.decimation_factor, n_fft)

    def passband_ripple_db(self, passband_edge: float = 0.8, n_fft: int = 8192) -> float:
        """
        Peak-to-peak variation of |H| in dB over the passband [0, passband_edge * pi/M].

        ``passband_edge < 1`` because a realisable filter needs a transition band;
        measuring right up to the cutoff would just report the roll-off.

        INTERPRETATION: for a filter with an oscillating response (boxcar, windowed
        sinc) this is genuine ripple. For a *monotone* response (binomial, Gaussian)
        there is no oscillation at all and the number instead measures in-band DROOP,
        i.e. how much of the signal you are throwing away inside the band you meant to
        keep. A large value there is a warning that the kernel is over-filtering: the
        default Gaussian (sigma = L/6) reads ~14 dB, meaning it attenuates the top of
        its own passband by a factor of 5. Read this column together with
        ``stopband_leakage()``: low leakage bought with high droop is not a better
        filter, just a narrower one.
        """
        w, H = self.frequency_response(n_fft)
        mag = np.abs(H)
        band = w <= passband_edge * np.pi / self.decimation_factor
        if band.sum() < 2:
            return 0.0
        lo, hi = mag[band].min(), mag[band].max()
        if lo <= 0:
            return float("inf")
        return float(20.0 * np.log10(hi / lo))

    def stopband_attenuation_db(self, stopband_edge: float = 1.0, n_fft: int = 8192) -> float:
        """
        Minimum attenuation in dB (relative to DC) over [stopband_edge * pi/M, pi].

        Larger is better. CAVEAT: with the default ``stopband_edge=1.0`` this measures
        from the folding frequency ``pi/M`` itself. Any design whose cutoff is placed
        *below* ``pi/M`` (i.e. ``SincPool1d`` with ``cutoff_scale < 1``) still has its
        transition band straddling that point, so the number reported here is the
        attenuation partway down the transition slope, NOT the true stopband floor -
        e.g. the default Kaiser design reads only ~9.5 dB while actually leaking just
        1.3% of its energy. Pass ``stopband_edge=1.2`` or higher to measure past the
        transition, and prefer ``stopband_leakage()`` as the headline figure of merit,
        since it integrates the whole folded region rather than reporting one point.
        """
        w, H = self.frequency_response(n_fft)
        mag = np.abs(H)
        ref = mag[0] if mag[0] > 0 else mag.max()
        band = w >= stopband_edge * np.pi / self.decimation_factor
        if band.sum() == 0:
            return float("inf")
        peak = mag[band].max()
        if peak <= 0:
            return float("inf")
        return float(-20.0 * np.log10(peak / ref))

    def group_delay_samples(self) -> float:
        """
        Centre of mass of |h|, in input samples, relative to the kernel centre.

        Zero for any symmetric (linear-phase) kernel - boxcar, binomial, Gaussian and
        windowed-sinc are all symmetric. Non-zero for Daubechies wavelets, which are
        asymmetric and therefore smear transients asymmetrically in time. Worth
        reporting whenever event timing matters.
        """
        h = np.abs(self.effective_kernel())
        n = np.arange(h.size, dtype=np.float64)
        return float((h * n).sum() / h.sum() - (h.size - 1) / 2.0)

    def summary(self) -> dict:
        """All figures of merit in one dict, convenient for building result tables."""
        return {
            "layer": type(self).__name__,
            "kind": self._kind,
            "decimation_factor": self.decimation_factor,
            "kernel_length": int(self.effective_kernel().size),
            "stopband_leakage_pct": 100.0 * self.stopband_leakage(),
            "stopband_attenuation_db": self.stopband_attenuation_db(),
            "passband_ripple_db": self.passband_ripple_db(),
            "group_delay_samples": self.group_delay_samples(),
            "dc_gain": float(self.effective_kernel().sum()),
        }

    def extra_repr(self) -> str:
        return (
            f"M={self.decimation_factor}, L={self.kernel.numel()}, "
            f"padding_mode={self.padding_mode}, offset={self.offset}, gain={self.gain}"
        )


# ---------------------------------------------------------------------------------
# Shared functional core
# ---------------------------------------------------------------------------------
def _filter_and_subsample(
    x: torch.Tensor,
    kernel: torch.Tensor,
    factor: int,
    offset: int,
    padding_mode: str,
) -> torch.Tensor:
    """
    Depthwise FIR filtering at stride 1, then subsampling by ``factor``.

    Implementation notes
    --------------------
    * The (B, D, T) tensor is reshaped to (B*D, 1, T) so a single-channel kernel can
      be used. This is equivalent to a depthwise convolution with ``groups=D`` but
      avoids materialising D copies of the kernel and sidesteps the requirement that
      grouped-conv weights be contiguous.
    * ``F.conv1d`` computes cross-correlation, not convolution. For the symmetric
      kernels (boxcar / binomial / Gaussian / windowed-sinc) the two coincide. The
      wavelet layer handles the asymmetric case by pre-flipping its kernel.
    * Filtering is done at stride 1 and subsampling applied afterwards, rather than
      folding the stride into the convolution. It is ~M times more FLOPs, but it (a)
      keeps the two conceptual steps separate and inspectable, (b) makes the exact
      output length trivial to guarantee, and (c) is negligible next to the rest of
      the network at these kernel sizes.
    """
    B, D, T = x.shape
    L = int(kernel.numel())

    # Centred padding so that the stride-1 filtered signal has length exactly T.
    pad_left = (L - 1) // 2
    pad_right = L - 1 - pad_left

    if padding_mode == "reflect" and max(pad_left, pad_right) >= T:
        raise ValueError(
            f"reflect padding requires pad < T, but pad={max(pad_left, pad_right)} "
            f"and T={T}. Use a shorter kernel or padding_mode='replicate'."
        )

    xf = x.reshape(B * D, 1, T)
    if L > 1:
        if padding_mode == "constant":
            xf = F.pad(xf, (pad_left, pad_right), mode="constant", value=0.0)
        else:
            xf = F.pad(xf, (pad_left, pad_right), mode=padding_mode)

    # Cast the kernel to the input dtype so this works under autocast / half / double.
    w = kernel.to(dtype=xf.dtype, device=xf.device).view(1, 1, L)
    y = F.conv1d(xf, w, bias=None, stride=1)          # (B*D, 1, T)
    y = y.reshape(B, D, -1)

    # Subsample. Slicing then truncating guarantees exactly T // factor samples for
    # any (T, factor, offset) combination, matching nn.AvgPool1d(factor, factor).
    n_out = T // factor
    if factor > 1:
        y = y[..., offset::factor]
    return y[..., :n_out].contiguous()


def _leakage_of(h: np.ndarray, factor: int, n_fft: int = 8192) -> float:
    """Stopband leakage of an arbitrary kernel; see ``stopband_leakage`` for the math."""
    h = np.asarray(h, dtype=np.float64)
    H = np.fft.rfft(h, n=n_fft)
    w = np.linspace(0.0, np.pi, H.size, endpoint=True)
    p = np.abs(H) ** 2
    total = p.sum()
    if total <= 0:
        return 0.0
    return float(p[w >= np.pi / factor].sum() / total)


# =================================================================================
# 1. Average pooling - the boxcar baseline
# =================================================================================
class AvgPool1d(_FIRDecimationPool1d):
    """
    Standard average pooling, expressed explicitly as boxcar-filter + subsample.

    This is the BASELINE of the study, and writing it in the same framework as the
    others makes the comparison airtight: it differs from ``BlurPool1d`` only in the
    choice of ``h``.

    The boxcar of length M has the closed-form response (Dirichlet kernel)

        H(w) = (1/M) * sin(w M / 2) / sin(w / 2) * exp(-j w (M-1)/2)

    whose first zero sits exactly at the post-decimation Nyquist ``w = 2 pi / M``.
    That placement is why averaging is not a *disaster* as an anti-aliasing filter -
    but its sidelobes decay only as 1/w, with the first sidelobe at about -13 dB, so
    roughly 20% of its energy lies above ``pi/M`` and is folded back on subsampling.
    That residual folding is the phenomenon this study measures.

    Config
    ------
    decimation_factor : int or (1, int)     [required]
    (plus all shared keys: padding_mode, phase, gain, ...)

    Note
    ----
    With ``padding_mode='constant'`` and ``phase='center'`` this layer is numerically
    identical to ``nn.AvgPool1d(M, stride=M)`` on the interior; boundary samples
    differ because ``nn.AvgPool1d`` does not pad at all. ``self_test()`` verifies the
    exact agreement.
    """

    _kind = "boxcar"

    def _build_kernel(self, config: dict) -> np.ndarray:
        # Length equals the decimation factor: this is what "averaging over the
        # pooling window" means. Note the length therefore scales with M
        # automatically - the property that short fixed binomial kernels lack.
        M = self.decimation_factor
        return np.ones(M, dtype=np.float64) / M

    def _check_against_boxcar(self) -> None:
        # It *is* the boxcar; nothing to warn about.
        return


# =================================================================================
# 2. Max pooling - nonlinear reference arm
# =================================================================================
class MaxPool1d(nn.Module):
    """
    Plain max pooling. Included only as the *un-anti-aliased nonlinear reference*.

    Max pooling does not fit the "filter then subsample" decomposition, because the
    max operator is nonlinear: it has no frequency response, and worse, it can
    *create* new high-frequency content that was not present in its input (the
    envelope of a rectified oscillation contains harmonics). So it both fails to
    suppress out-of-band energy and manufactures more of it before subsampling.
    Several works report that average pooling anti-aliases *better* than max pooling
    for exactly this reason.

    It is here so the ablation can report the operator most widely used in practice,
    and so the ordering (max < avg < properly filtered) can be demonstrated rather
    than assumed.

    Config
    ------
    decimation_factor : int or (1, int)     [required]
    ceil_mode : bool, default False
        Passed to ``nn.MaxPool1d``. Left at False so the output length is ``T // M``,
        matching every other layer here.
    """

    def __init__(self, config: dict):
        super().__init__()
        raw = _get(
            config, "decimation_factor",
            aliases=("pool_size", "factor", "p_kernel"),
        )
        if raw is None:
            raise KeyError("config must contain 'decimation_factor' (or an alias)")
        self.decimation_factor = _as_decimation_factor(raw)
        self.pool = nn.MaxPool1d(
            kernel_size=self.decimation_factor,
            stride=self.decimation_factor,
            ceil_mode=bool(_get(config, "ceil_mode", False)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (B, D, T), got {tuple(x.shape)}")
        return self.pool(x)

    def summary(self) -> dict:
        # No linear frequency response exists, so the spectral metrics are undefined.
        return {
            "layer": "MaxPool1d",
            "kind": "nonlinear",
            "decimation_factor": self.decimation_factor,
            "kernel_length": self.decimation_factor,
            "stopband_leakage_pct": float("nan"),
            "stopband_attenuation_db": float("nan"),
            "passband_ripple_db": float("nan"),
            "group_delay_samples": 0.0,
            "dc_gain": 1.0,
        }

    def extra_repr(self) -> str:
        return f"M={self.decimation_factor} (nonlinear, no anti-aliasing)"


# =================================================================================
# 3. BlurPool - fixed heuristic kernel (binomial / Gaussian / boxcar)
# =================================================================================
class BlurPool1d(_FIRDecimationPool1d):
    """
    Fixed-kernel anti-aliased pooling: the 1D analogue of Zhang's BlurPool (2019).

    Same structure as average pooling, better kernel. The binomial (Pascal-row)
    kernel has response

        |H(w)| = |cos(w / 2)| ** (L - 1)

    which is monotonically decreasing with NO sidelobes and no passband ripple - a
    qualitative improvement over the boxcar's oscillating Dirichlet response. The
    cost is a slow roll-off: suppression is bought only through length.

    THE ONE THING TO GET RIGHT
    --------------------------
    The kernel length must scale with the decimation factor. ``|cos(w/2)|**(L-1)``
    has a bandwidth set by L alone, while the cutoff you need is ``pi / M``. The
    canonical 2D kernels ``[1,2,1]`` (L=3) and ``[1,4,6,4,1]`` (L=5) were chosen for
    stride-2 layers; reusing them at M=8 gives *worse* aliasing than doing nothing
    beyond averaging:

        M = 4   boxcar leakage 21.6%   |  L=3: 39.7%   L=5: 24.5%   L=9: 10.6%
        M = 8   boxcar leakage 22.4%   |  L=3: 67.5%   L=5: 56.5%   L=25: 17.0%

    The default here is therefore ``kernel_size = 4*M + 1``, and a warning fires if
    the configuration is worse than the boxcar. If you want to reproduce the literal
    2D BlurPool setting, pass ``kernel_size=3`` explicitly and expect the warning.

    Config
    ------
    decimation_factor : int or (1, int)     [required]
    kernel_size : int, default 4*M + 1
        Length L of the blur kernel. Odd values keep the kernel symmetric about a
        sample, giving exactly linear phase and zero net delay.
    kernel : {'binomial', 'gaussian', 'boxcar'}, default 'binomial'
        'binomial' - Pascal row; the standard BlurPool choice.
        'gaussian' - truncated Gaussian; no sidelobes at all, slowest roll-off.
        'boxcar'   - moving average of length ``kernel_size``; provided so that
                     ``BlurPool1d(kernel='boxcar', kernel_size=M)`` reproduces
                     ``AvgPool1d`` and the family is provably nested.
    sigma : float, optional
        Standard deviation for the Gaussian kernel; defaults to ``kernel_size / 6``.
    (plus all shared keys)
    """

    _kind = "binomial"

    def _build_kernel(self, config: dict) -> np.ndarray:
        M = self.decimation_factor
        kind = str(_get(config, "kernel", "binomial")).lower()
        # Default length scales with M - see the class docstring.
        length = int(_get(config, "kernel_size", 4 * M + 1, aliases=("blur_kernel_size", "L")))
        if length < 1:
            raise ValueError("kernel_size must be >= 1")

        self._kind = kind
        if kind == "binomial":
            return _binomial_kernel(length)
        if kind == "gaussian":
            return _gaussian_kernel(length, _get(config, "sigma", None))
        if kind == "boxcar":
            return np.ones(length, dtype=np.float64) / length
        raise ValueError(
            f"unknown kernel {kind!r}; expected 'binomial', 'gaussian' or 'boxcar'"
        )


# =================================================================================
# 4. SincPool - properly designed windowed-sinc (Kaiser) decimation filter
# =================================================================================
class SincPool1d(_FIRDecimationPool1d):
    """
    Windowed-sinc anti-aliasing filter: what a DSP engineer would actually use.

    This is the *principled ceiling* arm of the ablation. The ideal anti-aliasing
    filter for decimation by M is a brick wall at ``w_c = pi / M``, whose impulse
    response is the (doubly infinite) sinc

        h_ideal[n] = (1 / M) * sinc(n / M).

    Truncating it produces Gibbs oscillations (a fixed ~-21 dB first sidelobe no
    matter how long the filter). Multiplying by a smooth *window* trades transition
    width for stopband attenuation and removes that floor. The Kaiser window is the
    standard choice because its single parameter ``beta`` sets the attenuation
    directly, with the classical design formulas

        beta  ~  0.1102 (A - 8.7)                        for A > 50 dB
        N     ~  (A - 8) / (2.285 * delta_w)             taps

    where A is the desired stopband attenuation in dB and ``delta_w`` the transition
    width in rad/sample. Both are implemented here, so you can specify the filter
    the way a specification is actually written - "60 dB stopband, 10% transition" -
    instead of guessing a kernel length.

    Config
    ------
    decimation_factor : int or (1, int)     [required]

    Either specify the design directly ...
    num_taps : int, default 8*M + 1
        Filter length N. Forced to odd (Type-I linear phase, exactly zero delay).
    beta : float, default 8.6 (~ 80 dB)
        Kaiser window shape parameter.

    ... or give a specification and let ``scipy.signal.kaiserord`` size it:
    stopband_attenuation_db : float, optional
        Target attenuation A in dB. If given, ``beta`` is derived from it.
    transition_width : float, optional
        Transition band width as a FRACTION of the cutoff ``pi/M`` (e.g. 0.2 -> the
        transition spans 20% of the cutoff frequency). If given together with
        ``stopband_attenuation_db``, ``num_taps`` is derived rather than assumed.
    max_taps : int, optional
        Safety cap on a derived ``num_taps`` (long filters cost FLOPs and, with
        reflect padding, need T > N/2).

    cutoff_scale : float, default 0.9
        Places the -6 dB cutoff at ``cutoff_scale * pi / M`` instead of exactly
        ``pi / M``. A real filter has a transition band straddling its cutoff; putting
        the cutoff slightly below Nyquist keeps the whole transition inside the
        retained band, at the price of attenuating the top ~10% of the passband.
        This is the single most consequential knob for a signal with meaningful
        high-frequency content, and is exactly the anti-aliasing / signal-loss
        trade-off the study should characterise. Set it to 1.0 for a textbook cutoff.
    window : str or tuple, default ('kaiser', beta)
        Any window accepted by ``scipy.signal.firwin`` ('hamming', 'blackman',
        ('kaiser', 8.6), ...), for comparing window choices.
    (plus all shared keys)
    """

    _kind = "windowed-sinc"

    def _build_kernel(self, config: dict) -> np.ndarray:
        try:
            from scipy import signal as sps
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SincPool1d needs SciPy for filter design (scipy.signal.firwin). "
                "Install scipy, or use BlurPool1d, whose kernels are closed-form."
            ) from exc

        M = self.decimation_factor
        if M == 1:
            return np.ones(1, dtype=np.float64)

        cutoff_scale = float(_get(config, "cutoff_scale", 0.9))
        if not 0.0 < cutoff_scale <= 1.0:
            raise ValueError("cutoff_scale must lie in (0, 1]")
        # firwin with fs=2.0 expresses cutoff in units where 1.0 == Nyquist == pi.
        # The anti-aliasing cutoff for decimation by M is pi/M -> 1/M in these units.
        cutoff = cutoff_scale / M

        atten = _get(config, "stopband_attenuation_db", None)
        trans = _get(config, "transition_width", None)
        window = _get(config, "window", None)
        num_taps = _get(config, "num_taps", None, aliases=("kernel_size", "N", "L"))

        # --- resolve the window -------------------------------------------------
        if window is None:
            if atten is not None:
                # Classical Kaiser formula: beta from the attenuation spec.
                beta = float(sps.kaiser_beta(float(atten)))
            else:
                beta = float(_get(config, "beta", 8.6))
            window = ("kaiser", beta)
        self._design_window = window

        # --- resolve the length -------------------------------------------------
        if num_taps is None:
            if atten is not None and trans is not None:
                # kaiserord wants the transition width in the same normalised units
                # as `cutoff` (Nyquist == 1).
                width = float(trans) * cutoff
                n_est, _beta = sps.kaiserord(float(atten), width)
                num_taps = int(n_est)
                cap = _get(config, "max_taps", None)
                if cap is not None:
                    num_taps = min(num_taps, int(cap))
            else:
                # Default: 8*M + 1. Scales with M (mandatory, see BlurPool1d docs)
                # and is long enough for a usefully sharp transition.
                num_taps = 8 * M + 1
        num_taps = int(num_taps)
        if num_taps < 1:
            raise ValueError("num_taps must be >= 1")
        if num_taps % 2 == 0:
            # Odd length -> Type-I linear-phase FIR, symmetric about a sample,
            # integer group delay (N-1)/2 which our centred padding cancels exactly.
            num_taps += 1

        self._num_taps = num_taps
        self._cutoff_normalised = cutoff
        h = sps.firwin(num_taps, cutoff, window=window, fs=2.0, pass_zero=True)
        return np.asarray(h, dtype=np.float64)

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return (
            f"{base}, window={getattr(self, '_design_window', None)}, "
            f"cutoff={getattr(self, '_cutoff_normalised', float('nan')):.4f}*Nyquist"
        )


# =================================================================================
# 5. DWTPool - orthogonal wavelet filter bank
# =================================================================================
class DWTPool1d(_FIRDecimationPool1d):
    """
    Decimation by the low-pass branch of an orthogonal wavelet filter bank (DWT).

    Mechanistically distinct from the two FIR layers above. Rather than *designing* a
    low-pass filter to approximate a brick wall, the DWT uses a filter pair
    (``h`` low-pass, ``g`` high-pass) forming a two-channel *perfect reconstruction*
    quadrature-mirror bank: the two subbands together lose no information, so the
    aliasing introduced by decimating each branch by 2 is exactly cancelled when they
    are recombined. Keeping only the approximation (low-pass) branch is a principled
    band-split rather than an ad hoc blur - the point WaveCNet makes against
    BlurPool's Pascal-triangle kernels being "ad hoc with no theoretical
    justification".

    Decimation by M is obtained by cascading ``log2(M)`` single-level stages, so
    **M must be a power of two**.

    Trade-offs worth knowing
    ------------------------
    * Daubechies filters are ASYMMETRIC, hence nonlinear phase: transients get
      smeared asymmetrically in time. If event timing matters (ERPs, QRS complexes)
      prefer ``wavelet='sym4'``, the least-asymmetric 8-tap orthogonal wavelet.
      ``group_delay_samples()`` quantifies this.
    * ``wavelet='haar'`` is a 2-tap boxcar and is a *deliberately weak* filter;
      useful as a lower bound, not as a serious candidate.
    * The effective filter of the cascade is longer and sharper than one stage: at
      M=8 the db4 cascade has a 50-tap equivalent response with ~9% leakage, between
      a mid-length binomial and a good Kaiser design.

    Config
    ------
    decimation_factor : int or (1, int)     [required, power of two]
    wavelet : str, default 'db4'
        One of ``WAVELET_FILTERS``: 'haar'/'db1', 'db2', 'db3', 'db4', 'sym4'.
        If PyWavelets is installed, any ``pywt`` orthogonal wavelet name also works.
    (plus all shared keys)

    Implementation note
    -------------------
    The stored kernel is the time reverse of the tabulated reconstruction filter,
    ``dec_lo = rec_lo[::-1]``. Because ``F.conv1d`` cross-correlates, correlating
    with ``dec_lo`` implements true *convolution* with ``rec_lo``, matching the
    standard DWT analysis convention (``pywt``'s ``dec_lo``).
    """

    _kind = "wavelet"

    def __init__(self, config: dict):
        # The number of cascade levels must be known before the base class builds the
        # kernel, because the kernel here is a single *stage* while the decimation
        # factor describes the whole cascade.
        raw = _get(
            config, "decimation_factor",
            aliases=("pool_size", "factor", "p_kernel"),
        )
        if raw is None:
            raise KeyError("config must contain 'decimation_factor' (or an alias)")
        factor = _as_decimation_factor(raw)
        if factor < 1 or (factor & (factor - 1)) != 0:
            raise ValueError(
                f"DWTPool1d requires a power-of-two decimation factor, got {factor}. "
                "Each DWT level decimates by exactly 2. Use SincPool1d or BlurPool1d "
                "for other factors."
            )
        self.levels = int(round(math.log2(factor))) if factor > 1 else 0
        super().__init__(config)

    def _build_kernel(self, config: dict) -> np.ndarray:
        name = str(_get(config, "wavelet", "db4")).lower()
        self.wavelet_name = name

        if name in WAVELET_FILTERS:
            rec_lo = np.asarray(WAVELET_FILTERS[name], dtype=np.float64)
        else:
            # Optional escape hatch: any orthogonal wavelet available via PyWavelets.
            try:
                import pywt
            except ImportError as exc:
                raise ValueError(
                    f"unknown wavelet {name!r}. Built-in options: "
                    f"{sorted(WAVELET_FILTERS)}. Install PyWavelets for more."
                ) from exc
            rec_lo = np.asarray(pywt.Wavelet(name).rec_lo, dtype=np.float64)

        # Time-reverse so that cross-correlation implements convolution with rec_lo
        # (i.e. the standard analysis/decomposition filter dec_lo).
        return rec_lo[::-1].copy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``levels`` successive (filter, decimate-by-2) stages."""
        if x.dim() != 3:
            raise ValueError(f"expected (B, D, T), got {tuple(x.shape)}")
        y = x
        for _ in range(self.levels):
            y = _filter_and_subsample(
                y,
                kernel=self.kernel,
                factor=2,
                # Offset 0 at every level. The single-stage 'center' offset for M=2
                # is (2-1)//2 = 0 anyway, so this is consistent with the FIR layers.
                offset=0,
                padding_mode=self.padding_mode,
            )
        return y

    def effective_kernel(self) -> np.ndarray:
        """
        Collapse the multi-level cascade into one equivalent impulse response.

        Uses the noble identity: decimating by 2 then filtering with ``h`` equals
        filtering with ``h`` upsampled by 2 then decimating by 2. Applying it
        repeatedly gives

            h_eff = h * (h upsampled by 2) * (h upsampled by 4) * ...

        so the whole cascade is a single FIR filter followed by decimation by
        ``2**levels``. This is what makes the wavelet arm directly comparable with
        the FIR arms: all the analysis methods inherited from the base class operate
        on this equivalent kernel.

        Verified numerically: the cascade and the equivalent single-stage filter agree
        to ~1e-16 (with circular padding, where the equivalence is exactly LTI), using
        the sampling offset ``(len(h_eff) - 1)//2 - ((L - 1)//2) * (2**levels - 1)``.
        """
        h = self.kernel.detach().cpu().numpy().astype(np.float64)
        if self.levels <= 1:
            return h
        h_eff = h.copy()
        for level in range(1, self.levels):
            h_eff = np.convolve(h_eff, _upsample_zeros(h, 2**level))
        return h_eff

    def extra_repr(self) -> str:
        return (
            f"M={self.decimation_factor} ({self.levels} levels), "
            f"wavelet={getattr(self, 'wavelet_name', '?')}, "
            f"L_stage={self.kernel.numel()}, L_eff={self.effective_kernel().size}, "
            f"padding_mode={self.padding_mode}, gain={self.gain}"
        )


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
    Verify the invariants this file relies on. Run once after installing.

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

if __name__ == "__main__":  # pragma: no cover
    self_test()

    print("\n" + "=" * 78)
    print("Figures of merit at the two EEGNet pooling stages (BCI IV-2a, 250 Hz)")
    print("=" * 78)
    print("Primary metric is 'leak%' = fraction of filter energy above pi/M that gets")
    print("folded back by subsampling (lower is better). 'att@pi/M' is measured at the")
    print("folding frequency itself, so it reads conservatively for cutoff_scale < 1.")
    print("'minT' = shortest input this kernel can be used on with reflect padding.")
    header = (f"{'layer':34s} {'M':>2s} {'L':>4s} {'leak%':>8s} {'att@pi/M':>9s} "
              f"{'delay':>7s} {'minT':>5s}")
    for M in (4, 8):
        fs_in = 250.0 if M == 4 else 62.5
        print(f"\n-- decimation M = {M}: input rate {fs_in:g} Hz -> output rate "
              f"{fs_in / M:g} Hz, folding frequency {fs_in / (2 * M):g} Hz --")
        print(header)
        arms = [
            ("AvgPool (boxcar)", {"pooling_type": "avg"}),
            ("BlurPool binomial L=3 (2D default)", {"pooling_type": "blur", "kernel_size": 3,
                                                    "warn_if_worse_than_boxcar": False}),
            ("BlurPool binomial L=4M+1", {"pooling_type": "blur"}),
            ("BlurPool gaussian L=4M+1", {"pooling_type": "blur", "kernel": "gaussian"}),
            ("SincPool Kaiser N=8M+1", {"pooling_type": "sinc"}),
            # A spec-driven design; max_taps keeps it usable on a 250-sample stage-2
            # input (an uncapped 60 dB / 25% design asks for 259 taps at M=8).
            ("SincPool 60dB/25% spec (capped)", {"pooling_type": "sinc",
                                                 "stopband_attenuation_db": 60,
                                                 "transition_width": 0.25,
                                                 "max_taps": 61}),
            ("DWTPool db4", {"pooling_type": "dwt", "wavelet": "db4"}),
            ("DWTPool sym4", {"pooling_type": "dwt", "wavelet": "sym4"}),
        ]
        for label, cfg in arms:
            cfg = dict(cfg, decimation_factor=M)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                layer = get_pooling_layer(cfg)
            s = layer.summary()
            # reflect padding needs pad < T, with pad = (L - 1) // 2
            min_T = (s["kernel_length"] - 1) // 2 + 1
            print(f"{label:34s} {s['decimation_factor']:2d} {s['kernel_length']:4d} "
                  f"{s['stopband_leakage_pct']:8.3f} {s['stopband_attenuation_db']:9.1f} "
                  f"{s['group_delay_samples']:7.2f} {min_T:5d}")
