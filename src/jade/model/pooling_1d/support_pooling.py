"""
Shared building blocks for the 1D anti-aliased pooling layers.

This module holds everything the individual pooling layers have in common:

* small kernel/utility helpers (``_get``, ``_as_decimation_factor``, ``_binomial_kernel``, ``_gaussian_kernel``, ``_upsample_zeros``),
* the functional core ``_filter_and_subsample`` and the figure-of-merit helper ``_leakage_of``,
* the base class :class:`~jade.model.pooling_1d.support_pooling._fir_decimation_pool_1d` implementing ``pooling = dense FIR filtering -> subsampling``.

See the package ``__init__`` docstring for the motivation and the tensor convention used throughout.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Full module imports
import warnings

# Specific imports
from typing import Optional, Sequence, Tuple

# Third party imports
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Small helpers

def _get(config : dict, key : str, default = None, aliases : Sequence[str] = ()) :
    """
    Fetch ``key`` from ``config``, accepting alternative spellings.

    Rationale: EEGNet-style configs use names like ``p_kernel_1``; this lets the same layer be dropped into an existing config dict without renaming anything.

    Parameters
    ----------
    config : dict
        The configuration dictionary.
    key : str
        The primary key to look up.
    default : Any, default None
        The value returned when neither ``key`` nor any alias is present.
    aliases : sequence of str, default ()
        Alternative spellings accepted for ``key``.

    Returns
    -------
    Any
        The first non-``None`` value found, or ``default``.
    """

    for candidate_key in (key, *aliases) :
        if candidate_key in config and config[candidate_key] is not None :
            return config[candidate_key]
    return default


def _as_decimation_factor(value) -> int :
    """
    Normalize the decimation factor to a positive int.

    It accepts ``4``, ``(4,)``, ``(1, 4)`` or ``[1, 4]``.
    The ``(1, M)`` form is accepted because EEGNet configs specify 2D pooling kernels ``(1, M)`` that act on the time axis only; this way an existing config value can be passed straight through.

    Parameters
    ----------
    value : int or tuple or list
        The raw decimation factor.

    Returns
    -------
    int
        The positive integer decimation factor.
    """

    if isinstance(value, (tuple, list)) :
        values = [int(single_value) for single_value in value]
        if len(values) == 1 :
            factor = values[0]
        elif len(values) == 2 :
            # (height, width) with height == 1 -> purely temporal pooling
            if values[0] != 1 :
                raise ValueError(
                    f"2D pooling kernel {tuple(values)} pools across the non-temporal axis too; "
                    "these 1D layers only decimate along time. Pass an int or (1, M)."
                )
            factor = values[1]
        else :
            raise ValueError(f"cannot interpret {value!r} as a 1D decimation factor")
    else :
        factor = int(value)

    if factor < 1 :
        raise ValueError(f"decimation factor must be >= 1, got {factor}")
    return factor


def _binomial_kernel(length : int) -> np.ndarray :
    """
    Binomial ("Pascal row") kernel of the requested length, normalized to sum 1.

    It is built by repeatedly convolving ``[1, 1]`` with itself, which is both the definition of a binomial coefficient row and the reason these kernels converge to a Gaussian as ``length`` grows (central limit theorem applied to the kernel).

    Its frequency response has the closed form  |H(w)| = |cos(w / 2)| ** (length - 1): monotonically decreasing on [0, pi], with NO sidelobes and no passband ripple - the qualitative improvement over the boxcar. The price is a slow roll-off.

    Parameters
    ----------
    length : int
        The kernel length (number of taps).

    Returns
    -------
    numpy.ndarray
        The normalized binomial kernel.
    """

    if length < 1 :
        raise ValueError("binomial kernel length must be >= 1")

    kernel = np.ones(1, dtype = np.float64)
    for _ in range(length - 1) :
        kernel = np.convolve(kernel, np.ones(2, dtype = np.float64))
    return kernel / kernel.sum()


def _gaussian_kernel(length : int, sigma : Optional[float] = None) -> np.ndarray :
    """
    Truncated Gaussian kernel, normalized to sum 1.

    If ``sigma`` is not given it is set so that the kernel is truncated at +-3 sigma, which keeps the truncation error (and hence the spurious sidelobes it creates) negligible.
    A Gaussian has a Gaussian frequency response: strictly positive, no sidelobes at all, but the slowest roll-off of the fixed kernels here.

    Parameters
    ----------
    length : int
        The kernel length (number of taps).
    sigma : float or None, default None
        The Gaussian standard deviation. If ``None`` it is set to ``length / 6``.

    Returns
    -------
    numpy.ndarray
        The normalized Gaussian kernel.
    """

    if length < 1 :
        raise ValueError("gaussian kernel length must be >= 1")

    if sigma is None :
        sigma = max(length / 6.0, 1e-3)  # +-3 sigma inside the support

    positions = np.arange(length, dtype = np.float64) - (length - 1) / 2.0
    kernel = np.exp(-0.5 * (positions / sigma) ** 2)
    return kernel / kernel.sum()


def _upsample_zeros(h : np.ndarray, factor : int) -> np.ndarray :
    """
    Insert ``factor - 1`` zeros between consecutive taps of ``h``.

    This implements the "noble identity" used to collapse a multi-stage decimating filter bank into one equivalent single-stage filter (see :meth:`~jade.model.pooling_1d.dwt_pool.dwt_pool_1d.effective_kernel`).
    Upsampling a filter by L compresses its frequency response by L along the frequency axis.

    Parameters
    ----------
    h : numpy.ndarray
        The kernel to upsample.
    factor : int
        The upsampling factor.

    Returns
    -------
    numpy.ndarray
        The zero-stuffed kernel.
    """

    if factor == 1 :
        return h.copy()

    output = np.zeros((len(h) - 1) * factor + 1, dtype = h.dtype)
    output[::factor] = h
    return output

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Base class: dense FIR filtering followed by subsampling

class _fir_decimation_pool_1d(nn.Module) :
    """
    Base class implementing ``pooling = dense FIR filtering -> subsampling``.

    Subclasses only have to build the low-pass kernel; everything else (padding, depthwise application, phase/offset selection, output-length guarantees and the frequency-domain analysis utilities) lives here.

    Shared configuration keys
    -------------------------
    decimation_factor : int or (1, int)        [required]
        The factor M by which the time axis is reduced.
        Aliases accepted: ``pool_size``, ``factor``, ``p_kernel``.
        NOTE: ``kernel_size`` is deliberately NOT an alias here - it always means the *filter length*.
    padding_mode : str, default 'reflect'
        How to extend the signal so the filtered output keeps length T before subsampling.
        One of 'reflect', 'replicate', 'constant' (zeros), 'circular'.
        'reflect' is the default because EEG trials are short: zero-padding would inject an artificial step at both ends, whose broadband spectrum is precisely the high-frequency content we are trying to suppress.
    phase : {'center', 'first'} or int, default 'center'
        Which sample of each block of M to retain after filtering.
        'center' -> offset (M - 1) // 2, which reproduces the sample alignment of ``nn.AvgPool1d`` exactly (each output is centred on its input block).
        'first'  -> offset 0.
        Keep this fixed across an ablation: changing it shifts features in time.
    gain : {'dc', 'orthonormal', 'none'}, default 'dc'
        Kernel normalization.
        'dc' -> sum(h) = 1 (unit DC gain, comparable with an average); 'orthonormal' -> ||h||_2 = 1 (energy preserving); 'none' -> leave the kernel as constructed.
        Use 'dc' for a controlled comparison.
    warn_if_worse_than_boxcar : bool, default True
        Emit a warning at construction if the chosen filter passes *more* energy above the post-decimation Nyquist than a plain boxcar of length M would.
        This catches the "short blur_pool_1d kernel at large M" trap described in the module docstring.
    """

    #: subclasses set this for nicer error messages / __repr__
    _kind : str = "fir"

    def __init__(self, config : dict) :
        super().__init__()

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Decimation factor

        raw_factor = _get(
            config, "decimation_factor",
            aliases = ("pool_size", "factor", "p_kernel"),
        )
        if raw_factor is None :
            raise KeyError(
                "config must contain 'decimation_factor' (or an alias: 'pool_size', "
                "'factor', 'p_kernel'). Note 'kernel_size' is NOT an alias: it means "
                "the filter length."
            )
        self.decimation_factor : int = _as_decimation_factor(raw_factor)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Generic options

        self.padding_mode : str = _get(config, "padding_mode", "reflect")
        if self.padding_mode not in ("reflect", "replicate", "constant", "circular") :
            raise ValueError(f"unsupported padding_mode {self.padding_mode!r}")

        self.gain : str = _get(config, "gain", "dc")
        if self.gain not in ("dc", "orthonormal", "none") :
            raise ValueError(f"unsupported gain {self.gain!r}")

        phase = _get(config, "phase", "center")
        if phase == "center" :
            # Matches nn.AvgPool1d: output n corresponds to input block [nM, nM+M).
            self.offset : int = (self.decimation_factor - 1) // 2
        elif phase == "first" :
            self.offset = 0
        elif isinstance(phase, int) :
            if not 0 <= phase < self.decimation_factor :
                raise ValueError(f"integer phase must lie in [0, {self.decimation_factor}), got {phase}")
            self.offset = phase
        else :
            raise ValueError(f"unsupported phase {phase!r}")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Build the kernel (subclass responsibility)

        kernel = np.asarray(self._build_kernel(config), dtype = np.float64).ravel()
        if kernel.size == 0 :
            raise ValueError("empty kernel")
        kernel = self._normalize(kernel)

        # Registered as a NON-persistent buffer: it follows .to(device)/.float()/.half() like a parameter would, but is excluded from state_dict().
        # That keeps checkpoints free of constants and avoids shape-mismatch errors when reloading a run with a different kernel length.
        self.register_buffer(
            "kernel", torch.as_tensor(kernel, dtype = torch.float32), persistent = False
        )

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Sanity check against the boxcar baseline

        if _get(config, "warn_if_worse_than_boxcar", True) and self.decimation_factor > 1 :
            self._check_against_boxcar()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Subclass hook

    def _build_kernel(self, config : dict) -> np.ndarray :
        """
        Build the low-pass kernel. Subclasses must implement this.
        """

        raise NotImplementedError

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Normalization

    def _normalize(self, kernel : np.ndarray) -> np.ndarray :
        """
        Apply the requested gain convention (see the ``gain`` config key).

        Parameters
        ----------
        kernel : numpy.ndarray
            The kernel as constructed by the subclass.

        Returns
        -------
        numpy.ndarray
            The normalized kernel.
        """

        if self.gain == "dc" :
            kernel_sum = kernel.sum()
            if abs(kernel_sum) < 1e-12 :
                raise ValueError(
                    "kernel has (near) zero DC gain and cannot be DC-normalized; "
                    "this is not a low-pass filter"
                )
            return kernel / kernel_sum
        if self.gain == "orthonormal" :
            return kernel / np.sqrt((kernel ** 2).sum())
        return kernel


    def _check_against_boxcar(self) -> None :
        """
        Warn if this filter is a *worse* anti-aliasing filter than plain averaging.

        A short fixed kernel (e.g. binomial [1,2,1]) has a bandwidth set by its own length, not by M.
        As M grows the required cutoff pi/M shrinks, and the short kernel eventually lets through more out-of-band energy than a boxcar of length M - which is what average pooling already does for free.
        """

        M = self.decimation_factor
        mine = self.stopband_leakage()
        box = _leakage_of(np.ones(M) / M, M)
        if mine > box :
            warnings.warn(
                f"{type(self).__name__}: this configuration is a WORSE anti-aliasing filter than plain "
                f"average pooling at M={M} (stopband leakage {mine * 100:.2f}% vs boxcar {box * 100:.2f}%). "
                "The kernel is too short for this decimation factor - the required "
                f"cutoff is pi/{M}. Increase the kernel length (a useful rule of thumb is "
                "length >= 4*M+1) or use sinc_pool_1d, which designs the filter for the "
                "cutoff directly. Set warn_if_worse_than_boxcar=False to silence this.",
                RuntimeWarning,
                stacklevel = 3,
            )

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # The actual computation

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        """
        Filter then subsample the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, D, T).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, D, T // M).
        """

        if x.dim() != 3 :
            raise ValueError(f"expected a 3D tensor (B, D, T), got shape {tuple(x.shape)}")
        return _filter_and_subsample(
            x,
            kernel = self.kernel,
            factor = self.decimation_factor,
            offset = self.offset,
            padding_mode = self.padding_mode,
        )

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Analysis utilities (no autograd; for figures and tables)

    def effective_kernel(self) -> np.ndarray :
        """
        The single-stage impulse response equivalent to this layer, as a NumPy array.

        For the FIR layers this is just the kernel; :class:`~jade.model.pooling_1d.dwt_pool.dwt_pool_1d` overrides it to collapse its multi-level cascade via the noble identity.

        Returns
        -------
        numpy.ndarray
            The equivalent single-stage kernel.
        """

        return self.kernel.detach().cpu().numpy().astype(np.float64)


    def frequency_response(self, n_fft : int = 8192) -> Tuple[np.ndarray, np.ndarray] :
        """
        Discrete-time frequency response of the effective kernel.

        To convert to Hz at an input sampling rate ``fs``: ``f = w / (2 * pi) * fs``.

        Parameters
        ----------
        n_fft : int, default 8192
            The FFT length.

        Returns
        -------
        w : numpy.ndarray
            Normalized angular frequency in [0, pi).
        H : numpy.ndarray
            Complex frequency response H(w).
        """

        h = self.effective_kernel()
        H = np.fft.rfft(h, n = n_fft)
        w = np.linspace(0.0, np.pi, H.size, endpoint = True)
        return w, H


    def stopband_leakage(self, n_fft : int = 8192) -> float :
        """
        Fraction of the filter's energy that sits ABOVE the post-decimation Nyquist.

            leakage = int_{pi/M}^{pi} |H(w)|^2 dw  /  int_0^{pi} |H(w)|^2 dw

        This is the single most useful scalar figure of merit for the present study: it is exactly the portion of the response that will be folded back into the retained band by the subsequent subsampling.
        0 = perfect anti-aliasing.

        Note this is a property of the *filter*, independent of the signal.
        A signal-dependent version (energy of the actual activations above pi/M) is the complementary per-layer diagnostic to log during training.

        Parameters
        ----------
        n_fft : int, default 8192
            The FFT length.

        Returns
        -------
        float
            The stopband leakage fraction.
        """

        return _leakage_of(self.effective_kernel(), self.decimation_factor, n_fft)


    def passband_ripple_db(self, passband_edge : float = 0.8, n_fft : int = 8192) -> float :
        """
        Peak-to-peak variation of |H| in dB over the passband [0, passband_edge * pi/M].

        ``passband_edge < 1`` because a realizable filter needs a transition band; measuring right up to the cutoff would just report the roll-off.

        INTERPRETATION: for a filter with an oscillating response (boxcar, windowed sinc) this is genuine ripple.
        For a *monotone* response (binomial, Gaussian) there is no oscillation at all and the number instead measures in-band DROOP, i.e. how much of the signal you are throwing away inside the band you meant to keep.
        A large value there is a warning that the kernel is over-filtering: the default Gaussian (sigma = L/6) reads ~14 dB, meaning it attenuates the top of its own passband by a factor of 5.
        Read this column together with :meth:`stopband_leakage`: low leakage bought with high droop is not a better filter, just a narrower one.

        Parameters
        ----------
        passband_edge : float, default 0.8
            Fraction of the cutoff at which the passband measurement stops.
        n_fft : int, default 8192
            The FFT length.

        Returns
        -------
        float
            The peak-to-peak passband variation in dB.
        """

        w, H = self.frequency_response(n_fft)
        magnitude = np.abs(H)
        band = w <= passband_edge * np.pi / self.decimation_factor
        if band.sum() < 2 :
            return 0.0
        low, high = magnitude[band].min(), magnitude[band].max()
        if low <= 0 :
            return float("inf")
        return float(20.0 * np.log10(high / low))


    def stopband_attenuation_db(self, stopband_edge : float = 1.0, n_fft : int = 8192) -> float :
        """
        Minimum attenuation in dB (relative to DC) over [stopband_edge * pi/M, pi].

        Larger is better.
        CAVEAT: with the default ``stopband_edge=1.0`` this measures from the folding frequency ``pi/M`` itself.
        Any design whose cutoff is placed *below* ``pi/M`` (i.e. :class:`~jade.model.pooling_1d.sinc_pool.sinc_pool_1d` with ``cutoff_scale < 1``) still has its transition band straddling that point, so the number reported here is the attenuation partway down the transition slope, NOT the true stopband floor - e.g. the default Kaiser design reads only ~9.5 dB while actually leaking just 1.3% of its energy.
        Pass ``stopband_edge=1.2`` or higher to measure past the transition, and prefer :meth:`stopband_leakage` as the headline figure of merit, since it integrates the whole folded region rather than reporting one point.

        Parameters
        ----------
        stopband_edge : float, default 1.0
            Fraction of the cutoff at which the stopband measurement starts.
        n_fft : int, default 8192
            The FFT length.

        Returns
        -------
        float
            The minimum stopband attenuation in dB.
        """

        w, H = self.frequency_response(n_fft)
        magnitude = np.abs(H)
        reference = magnitude[0] if magnitude[0] > 0 else magnitude.max()
        band = w >= stopband_edge * np.pi / self.decimation_factor
        if band.sum() == 0 :
            return float("inf")
        peak = magnitude[band].max()
        if peak <= 0 :
            return float("inf")
        return float(-20.0 * np.log10(peak / reference))


    def group_delay_samples(self) -> float :
        """
        Centre of mass of |h|, in input samples, relative to the kernel centre.

        Zero for any symmetric (linear-phase) kernel - boxcar, binomial, Gaussian and windowed-sinc are all symmetric.
        Non-zero for Daubechies wavelets, which are asymmetric and therefore smear transients asymmetrically in time.
        Worth reporting whenever event timing matters.

        Returns
        -------
        float
            The group delay in input samples.
        """

        h = np.abs(self.effective_kernel())
        positions = np.arange(h.size, dtype = np.float64)
        return float((h * positions).sum() / h.sum() - (h.size - 1) / 2.0)


    def summary(self) -> dict :
        """
        All figures of merit in one dict, convenient for building result tables.

        Returns
        -------
        dict
            A dictionary of the layer's figures of merit.
        """

        return {
            "layer"                    : type(self).__name__,
            "kind"                     : self._kind,
            "decimation_factor"        : self.decimation_factor,
            "kernel_length"            : int(self.effective_kernel().size),
            "stopband_leakage_pct"     : 100.0 * self.stopband_leakage(),
            "stopband_attenuation_db"  : self.stopband_attenuation_db(),
            "passband_ripple_db"       : self.passband_ripple_db(),
            "group_delay_samples"      : self.group_delay_samples(),
            "dc_gain"                  : float(self.effective_kernel().sum()),
        }


    def extra_repr(self) -> str :
        """
        Short description used by :meth:`torch.nn.Module.__repr__`.
        """

        return (
            f"M={self.decimation_factor}, L={self.kernel.numel()}, "
            f"padding_mode={self.padding_mode}, offset={self.offset}, gain={self.gain}"
        )

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Shared functional core

def _filter_and_subsample(
    x : torch.Tensor,
    kernel : torch.Tensor,
    factor : int,
    offset : int,
    padding_mode : str,
) -> torch.Tensor :
    """
    Depthwise FIR filtering at stride 1, then subsampling by ``factor``.

    Implementation notes
    --------------------
    * The (B, D, T) tensor is reshaped to (B*D, 1, T) so a single-channel kernel can be used.
      This is equivalent to a depthwise convolution with ``groups=D`` but avoids materializing D copies of the kernel and sidesteps the requirement that grouped-conv weights be contiguous.
    * ``F.conv1d`` computes cross-correlation, not convolution.
      For the symmetric kernels (boxcar / binomial / Gaussian / windowed-sinc) the two coincide.
      The wavelet layer handles the asymmetric case by pre-flipping its kernel.
    * Filtering is done at stride 1 and subsampling applied afterwards, rather than folding the stride into the convolution.
      It is ~M times more FLOPs, but it (a) keeps the two conceptual steps separate and inspectable, (b) makes the exact output length trivial to guarantee, and (c) is negligible next to the rest of the network at these kernel sizes.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, D, T).
    kernel : torch.Tensor
        The 1D low-pass kernel.
    factor : int
        The subsampling factor.
    offset : int
        The sample of each block of ``factor`` to retain.
    padding_mode : str
        One of 'reflect', 'replicate', 'constant', 'circular'.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (B, D, T // factor).
    """

    B, D, T = x.shape
    L = int(kernel.numel())

    # Centred padding so that the stride-1 filtered signal has length exactly T.
    pad_left = (L - 1) // 2
    pad_right = L - 1 - pad_left

    if padding_mode == "reflect" and max(pad_left, pad_right) >= T :
        raise ValueError(
            f"reflect padding requires pad < T, but pad={max(pad_left, pad_right)} and T={T}. Use a shorter "
            "kernel or padding_mode='replicate'."
        )

    xf = x.reshape(B * D, 1, T)
    if L > 1 :
        if padding_mode == "constant" :
            xf = F.pad(xf, (pad_left, pad_right), mode = "constant", value = 0.0)
        else :
            xf = F.pad(xf, (pad_left, pad_right), mode = padding_mode)

    # Cast the kernel to the input dtype so this works under autocast / half / double.
    w = kernel.to(dtype = xf.dtype, device = xf.device).view(1, 1, L)
    y = F.conv1d(xf, w, bias = None, stride = 1)          # (B*D, 1, T)
    y = y.reshape(B, D, -1)

    # Subsample.
    # Slicing then truncating guarantees exactly T // factor samples for any (T, factor, offset) combination, matching nn.AvgPool1d(factor, factor).
    n_out = T // factor
    if factor > 1 :
        y = y[..., offset::factor]
    return y[..., :n_out].contiguous()


def _leakage_of(h : np.ndarray, factor : int, n_fft : int = 8192) -> float :
    """
    Stopband leakage of an arbitrary kernel; see :meth:`~jade.model.pooling_1d.support_pooling._fir_decimation_pool_1d.stopband_leakage` for the math.

    Parameters
    ----------
    h : numpy.ndarray
        The kernel to analyze.
    factor : int
        The decimation factor.
    n_fft : int, default 8192
        The FFT length.

    Returns
    -------
    float
        The stopband leakage fraction.
    """

    h = np.asarray(h, dtype = np.float64)
    H = np.fft.rfft(h, n = n_fft)
    w = np.linspace(0.0, np.pi, H.size, endpoint = True)
    power = np.abs(H) ** 2
    total = power.sum()
    if total <= 0 :
        return 0.0
    return float(power[w >= np.pi / factor].sum() / total)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Multichannel wrapper: apply a 1D pooling layer channel-wise to (B, D, C, T)

class _multichannel_pool_1d(nn.Module) :
    """
    Apply a 1D pooling layer channel-wise to a multichannel EEG tensor.

    The 1D pooling layers in this package operate on ``(B, D, T)`` signals, decimating the time axis of every ``(batch, depth)`` row independently.
    Models such as EEGNet instead work on 4D tensors ``(B, D, C, T)`` (batch, depth, EEG channels, time) and need the pooling applied to each EEG channel independently.

    This thin wrapper flattens the ``(D, C)`` axes into a single depthwise axis, runs the wrapped 1D layer once, and restores the original layout::

        (B, D, C, T)  ->  reshape (B, D*C, T)  ->  pool  ->  (B, D*C, T // M)  ->  (B, D, C, T // M)

    Because the 1D layer already pools every row of its second axis independently, flattening ``(D, C)`` makes it pool every ``(depth, channel)`` pair independently - exactly equivalent to looping over channels, but vectorized into a single call.

    Subclasses (one per pooling operator, e.g. :class:`~jade.model.pooling_1d.avg_pool.avg_pool_1d_multichannel`) just build the wrapped 1D layer and hand it to ``super().__init__``.

    Parameters
    ----------
    pool_1d : torch.nn.Module
        The 1D pooling layer to apply channel-wise. It must map ``(B, D, T) -> (B, D, T // M)``.
    """

    def __init__(self, pool_1d : nn.Module) :
        super().__init__()
        self.pool = pool_1d

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        """
        Apply the wrapped 1D pooling layer to every EEG channel independently.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, D, C, T).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, D, C, T // M).
        """

        if x.dim() != 4 :
            raise ValueError(f"expected a 4D tensor (B, D, C, T), got shape {tuple(x.shape)}")
        B, D, C, T = x.shape
        x = x.reshape(B, D * C, T)
        x = self.pool(x)
        return x.reshape(B, D, C, x.shape[-1])
