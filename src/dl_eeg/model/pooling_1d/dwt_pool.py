"""
DWTPool - orthogonal wavelet filter bank, plus its filter-bank coefficients.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Full module imports
import math

# Specific imports
from typing import Dict, Sequence

# Third party imports
import numpy as np
import torch

# Internal imports
from dl_eeg.model.pooling_1d.support_pooling import (
    _fir_decimation_pool_1d,
    _as_decimation_factor,
    _filter_and_subsample,
    _get,
    _upsample_zeros,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Wavelet filter bank coefficients

# These are the *reconstruction* low-pass (scaling) filters ``rec_lo`` in the standard orthonormal convention, i.e. they satisfy
#       sum(h)      = sqrt(2)                      (DC gain, unit-norm convention)
#       sum(h**2)   = 1                            (unit energy)
#       sum_n h[n] * h[n + 2k] = 0  for k != 0     (double-shift orthogonality)
# All five sets below were verified numerically against those three conditions to ~1e-12.
# The analysis (decomposition) filter used by the DWT is the time reverse, ``dec_lo = rec_lo[::-1]``; see dwt_pool_1d for how that interacts with the fact that ``F.conv1d`` computes cross-correlation rather than convolution.
WAVELET_FILTERS : Dict[str, Sequence[float]] = {
    # Haar / db1: the shortest orthogonal wavelet.
    # Its low-pass is a 2-tap boxcar, so 'haar' DWT pooling is a very weak anti-aliasing filter (included as a deliberate lower bound / sanity arm, not as a serious candidate).
    "haar" : [0.7071067811865476, 0.7071067811865476],
    "db1"  : [0.7071067811865476, 0.7071067811865476],
    "db2"  : [
        0.48296291314469025, 0.836516303737469,
        0.22414386804185735, -0.12940952255092145,
    ],
    "db3"  : [
        0.3326705529509569, 0.8068915093133388, 0.4598775021193313,
        -0.13501102001039084, -0.08544127388224149, 0.035226291882100656,
    ],
    # db4 (8 taps, 4 vanishing moments) is the usual default: a good compromise between stopband attenuation and kernel length.
    "db4"  : [
        0.23037781330885523, 0.7148465705525415, 0.6308807679295904,
        -0.02798376941698385, -0.18703481171888114, 0.030841381835986965,
        0.032883011666982945, -0.010597401784997278,
    ],
    # sym4 is the "least asymmetric" 8-tap orthogonal wavelet: nearly linear phase, which means less temporal smearing of event-related features than db4.
    # Preferable when the *timing* of a transient matters (ERPs, QRS complexes).
    "sym4" : [
        0.032223100604042702, -0.012603967262037833, -0.09921954357684722,
        0.29785779560527736, 0.8037387518059161, 0.49761866763201545,
        -0.02963552764599851, -0.07576571478927333,
    ],
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# DWTPool - orthogonal wavelet filter bank

class dwt_pool_1d(_fir_decimation_pool_1d) :
    """
    Decimation by the low-pass branch of an orthogonal wavelet filter bank (DWT).

    Mechanistically distinct from the two FIR layers above.
    Rather than *designing* a low-pass filter to approximate a brick wall, the DWT uses a filter pair (``h`` low-pass, ``g`` high-pass) forming a two-channel *perfect reconstruction* quadrature-mirror bank: the two subbands together lose no information, so the aliasing introduced by decimating each branch by 2 is exactly cancelled when they are recombined.
    Keeping only the approximation (low-pass) branch is a principled band-split rather than an ad hoc blur - the point WaveCNet makes against BlurPool's Pascal-triangle kernels being "ad hoc with no theoretical justification".

    Decimation by M is obtained by cascading ``log2(M)`` single-level stages, so **M must be a power of two**.

    Trade-offs worth knowing
    ------------------------
    * Daubechies filters are ASYMMETRIC, hence nonlinear phase: transients get smeared asymmetrically in time.
      If event timing matters (ERPs, QRS complexes) prefer ``wavelet='sym4'``, the least-asymmetric 8-tap orthogonal wavelet.
      ``group_delay_samples()`` quantifies this.
    * ``wavelet='haar'`` is a 2-tap boxcar and is a *deliberately weak* filter; useful as a lower bound, not as a serious candidate.
    * The effective filter of the cascade is longer and sharper than one stage: at M=8 the db4 cascade has a 50-tap equivalent response with ~9% leakage, between a mid-length binomial and a good Kaiser design.

    Config
    ------
    decimation_factor : int or (1, int)     [required, power of two]
    wavelet : str, default 'db4'
        One of ``WAVELET_FILTERS``: 'haar'/'db1', 'db2', 'db3', 'db4', 'sym4'.
        If PyWavelets is installed, any ``pywt`` orthogonal wavelet name also works.
    (plus all shared keys)

    Implementation note
    -------------------
    The stored kernel is the time reverse of the tabulated reconstruction filter, ``dec_lo = rec_lo[::-1]``.
    Because ``F.conv1d`` cross-correlates, correlating with ``dec_lo`` implements true *convolution* with ``rec_lo``, matching the standard DWT analysis convention (``pywt``'s ``dec_lo``).
    """

    _kind = "wavelet"

    def __init__(self, config : dict) :
        # The number of cascade levels must be known before the base class builds the kernel, because the kernel here is a single *stage* while the decimation factor describes the whole cascade.
        raw = _get(
            config, "decimation_factor",
            aliases = ("pool_size", "factor", "p_kernel"),
        )
        if raw is None :
            raise KeyError("config must contain 'decimation_factor' (or an alias)")
        factor = _as_decimation_factor(raw)
        if factor < 1 or (factor & (factor - 1)) != 0 :
            raise ValueError(
                "dwt_pool_1d requires a power-of-two decimation factor, got {}. "
                "Each DWT level decimates by exactly 2. Use sinc_pool_1d or blur_pool_1d "
                "for other factors.".format(factor)
            )
        self.levels = int(round(math.log2(factor))) if factor > 1 else 0
        super().__init__(config)


    def _build_kernel(self, config : dict) -> np.ndarray :
        """
        Build a single DWT stage kernel (the time-reversed reconstruction low-pass).

        Parameters
        ----------
        config : dict
            The layer configuration.

        Returns
        -------
        numpy.ndarray
            The single-stage analysis low-pass kernel.
        """

        name = str(_get(config, "wavelet", "db4")).lower()
        self.wavelet_name = name

        if name in WAVELET_FILTERS :
            rec_lo = np.asarray(WAVELET_FILTERS[name], dtype = np.float64)
        else :
            # Optional escape hatch: any orthogonal wavelet available via PyWavelets.
            try :
                import pywt
            except ImportError as exc :
                raise ValueError(
                    "unknown wavelet {}. Built-in options: {}. "
                    "Install PyWavelets for more.".format(repr(name), sorted(WAVELET_FILTERS))
                ) from exc
            rec_lo = np.asarray(pywt.Wavelet(name).rec_lo, dtype = np.float64)

        # Time-reverse so that cross-correlation implements convolution with rec_lo (i.e. the standard analysis/decomposition filter dec_lo).
        return rec_lo[::-1].copy()


    def forward(self, x : torch.Tensor) -> torch.Tensor :
        """
        Apply ``levels`` successive (filter, decimate-by-2) stages.

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
            raise ValueError("expected (B, D, T), got {}".format(tuple(x.shape)))

        y = x
        for _ in range(self.levels) :
            y = _filter_and_subsample(
                y,
                kernel = self.kernel,
                factor = 2,
                # Offset 0 at every level. The single-stage 'center' offset for M=2 is (2-1)//2 = 0 anyway, so this is consistent with the FIR layers.
                offset = 0,
                padding_mode = self.padding_mode,
            )
        return y


    def effective_kernel(self) -> np.ndarray :
        """
        Collapse the multi-level cascade into one equivalent impulse response.

        It uses the noble identity: decimating by 2 then filtering with ``h`` equals filtering with ``h`` upsampled by 2 then decimating by 2.
        Applying it repeatedly gives

            h_eff = h * (h upsampled by 2) * (h upsampled by 4) * ...

        so the whole cascade is a single FIR filter followed by decimation by ``2**levels``.
        This is what makes the wavelet arm directly comparable with the FIR arms: all the analysis methods inherited from the base class operate on this equivalent kernel.

        Verified numerically: the cascade and the equivalent single-stage filter agree to ~1e-16 (with circular padding, where the equivalence is exactly LTI), using the sampling offset ``(len(h_eff) - 1)//2 - ((L - 1)//2) * (2**levels - 1)``.

        Returns
        -------
        numpy.ndarray
            The equivalent single-stage kernel of the whole cascade.
        """

        h = self.kernel.detach().cpu().numpy().astype(np.float64)
        if self.levels <= 1 :
            return h

        h_eff = h.copy()
        for level in range(1, self.levels) :
            h_eff = np.convolve(h_eff, _upsample_zeros(h, 2 ** level))
        return h_eff


    def extra_repr(self) -> str :
        """
        Short description used by :meth:`torch.nn.Module.__repr__`.
        """

        return "M={} ({} levels), wavelet={}, L_stage={}, L_eff={}, padding_mode={}, gain={}".format(
            self.decimation_factor, self.levels, getattr(self, "wavelet_name", "?"),
            self.kernel.numel(), self.effective_kernel().size, self.padding_mode, self.gain
        )
