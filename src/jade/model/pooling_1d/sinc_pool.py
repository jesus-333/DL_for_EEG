"""
SincPool - properly designed windowed-sinc (Kaiser) decimation filter.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Third party imports
import numpy as np

# Internal imports
from jade.model.pooling_1d.support_pooling import _fir_decimation_pool_1d, _get, _multichannel_pool_1d

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# SincPool - properly designed windowed-sinc (Kaiser) decimation filter

class sinc_pool_1d(_fir_decimation_pool_1d) :
    """
    Windowed-sinc anti-aliasing filter: what a DSP engineer would actually use.

    This is the *principled ceiling* arm of the ablation.
    The ideal anti-aliasing filter for decimation by M is a brick wall at ``w_c = pi / M``, whose impulse response is the (doubly infinite) sinc

        h_ideal[n] = (1 / M) * sinc(n / M).

    Truncating it produces Gibbs oscillations (a fixed ~-21 dB first sidelobe no matter how long the filter).
    Multiplying by a smooth *window* trades transition width for stopband attenuation and removes that floor.
    The Kaiser window is the standard choice because its single parameter ``beta`` sets the attenuation directly, with the classical design formulas

        beta  ~  0.1102 (A - 8.7)                        for A > 50 dB
        N     ~  (A - 8) / (2.285 * delta_w)             taps

    where A is the desired stopband attenuation in dB and ``delta_w`` the transition width in rad/sample.
    Both are implemented here, so you can specify the filter the way a specification is actually written - "60 dB stopband, 10% transition" - instead of guessing a kernel length.

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
        Transition band width as a FRACTION of the cutoff ``pi/M`` (e.g. 0.2 -> the transition spans 20% of the cutoff frequency).
        If given together with ``stopband_attenuation_db``, ``num_taps`` is derived rather than assumed.
    max_taps : int, optional
        Safety cap on a derived ``num_taps`` (long filters cost FLOPs and, with reflect padding, need T > N/2).

    cutoff_scale : float, default 0.9
        Places the -6 dB cutoff at ``cutoff_scale * pi / M`` instead of exactly ``pi / M``.
        A real filter has a transition band straddling its cutoff; putting the cutoff slightly below Nyquist keeps the whole transition inside the retained band, at the price of attenuating the top ~10% of the passband.
        This is the single most consequential knob for a signal with meaningful high-frequency content, and is exactly the anti-aliasing / signal-loss trade-off the study should characterize.
        Set it to 1.0 for a textbook cutoff.
    window : str or tuple, default ('kaiser', beta)
        Any window accepted by ``scipy.signal.firwin`` ('hamming', 'blackman', ('kaiser', 8.6), ...), for comparing window choices.
    (plus all shared keys)
    """

    _kind = "windowed-sinc"

    def _build_kernel(self, config : dict) -> np.ndarray :
        """
        Design the windowed-sinc kernel with :func:`scipy.signal.firwin`.

        Parameters
        ----------
        config : dict
            The layer configuration.

        Returns
        -------
        numpy.ndarray
            The designed FIR kernel.
        """

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Import scipy lazily (it is only needed for filter design)

        try :
            from scipy import signal as sps
        except ImportError as exc :  # pragma: no cover
            raise ImportError(
                "sinc_pool_1d needs SciPy for filter design (scipy.signal.firwin). "
                "Install scipy, or use blur_pool_1d, whose kernels are closed-form."
            ) from exc

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Resolve the cutoff

        M = self.decimation_factor
        if M == 1 :
            return np.ones(1, dtype = np.float64)

        cutoff_scale = float(_get(config, "cutoff_scale", 0.9))
        if not 0.0 < cutoff_scale <= 1.0 :
            raise ValueError("cutoff_scale must lie in (0, 1]")

        # firwin with fs=2.0 expresses cutoff in units where 1.0 == Nyquist == pi.
        # The anti-aliasing cutoff for decimation by M is pi/M -> 1/M in these units.
        cutoff = cutoff_scale / M

        atten = _get(config, "stopband_attenuation_db", None)
        trans = _get(config, "transition_width", None)
        window = _get(config, "window", None)
        num_taps = _get(config, "num_taps", None, aliases = ("kernel_size", "N", "L"))

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Resolve the window

        if window is None :
            if atten is not None :
                # Classical Kaiser formula: beta from the attenuation spec.
                beta = float(sps.kaiser_beta(float(atten)))
            else :
                beta = float(_get(config, "beta", 8.6))
            window = ("kaiser", beta)
        self._design_window = window

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Resolve the length

        if num_taps is None :
            if atten is not None and trans is not None :
                # kaiserord wants the transition width in the same normalized units as `cutoff` (Nyquist == 1).
                width = float(trans) * cutoff
                n_est, _beta = sps.kaiserord(float(atten), width)
                num_taps = int(n_est)
                cap = _get(config, "max_taps", None)
                if cap is not None :
                    num_taps = min(num_taps, int(cap))
            else :
                # Default: 8*M + 1. Scales with M (mandatory, see blur_pool_1d docs) and is long enough for a usefully sharp transition.
                num_taps = 8 * M + 1
        num_taps = int(num_taps)
        if num_taps < 1 :
            raise ValueError("num_taps must be >= 1")
        if num_taps % 2 == 0 :
            # Odd length -> Type-I linear-phase FIR, symmetric about a sample, integer group delay (N-1)/2 which our centred padding cancels exactly.
            num_taps += 1

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Design the filter

        self._num_taps = num_taps
        self._cutoff_normalized = cutoff
        h = sps.firwin(num_taps, cutoff, window = window, fs = 2.0, pass_zero = True)
        return np.asarray(h, dtype = np.float64)


    def extra_repr(self) -> str :
        """
        Short description used by :meth:`torch.nn.Module.__repr__`.
        """

        base = super().extra_repr()
        window = getattr(self, "_design_window", None)
        cutoff = getattr(self, "_cutoff_normalized", float("nan"))
        return f"{base}, window={window}, cutoff={cutoff:.4f}*Nyquist"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Multichannel variant

class sinc_pool_1d_multichannel(_multichannel_pool_1d) :
    """
    Channel-wise :class:`sinc_pool_1d` for multichannel EEG tensors of shape ``(B, D, C, T)``.

    Applies :class:`sinc_pool_1d` independently to every EEG channel; see :class:`~jade.model.pooling_1d.support_pooling._multichannel_pool_1d` for the mechanism.
    Takes the same ``config`` dict as :class:`sinc_pool_1d` (and, like it, needs SciPy for filter design).
    """

    def __init__(self, config : dict) :
        super().__init__(sinc_pool_1d(config))
