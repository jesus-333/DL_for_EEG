"""
Support functions for the frequency domain plots of the EEG datasets defined in :mod:`~jade.dataset.dataset_time`.

At the moment the module contains the routine used to move an EEG signal from the time domain to the frequency domain (i.e. the power spectral density estimation).
Three backends are provided:

- ``scipy`` : relies on ``scipy.signal.welch``. It is the default backend.
- ``torch`` : a small implementation of the Welch method based on ``torch.fft``. It is used automatically as default when scipy is not installed.
- ``numpy`` : a small implementation of the Welch method based on ``numpy.fft``. It is used as a backup when the torch backend fails (or when the user explicitly asks for it).

Thanks to this chain the frequency plots keep working even without the optional scipy dependency, and every backend returns plain numpy arrays that matplotlib can plot directly.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports section

from __future__ import annotations

# Full module imports
import warnings

# Specific imports
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Frequency configuration

def check_freq_config(freq_config : dict = None) -> dict :
    """
    Validate the frequency (power spectral density) configuration and fill the missing entries with their default value.

    The parameters mirror the ones of :func:`scipy.signal.welch` so that the three backends (scipy, torch and numpy) behave in the same way.
    If ``freq_config`` is ``None`` or an empty dictionary a brand new dictionary with every default value is returned.

    Parameters
    ----------
    freq_config : dict, optional
        Dictionary with (a subset of) the frequency configuration keys.
        The recognized keys are:

        - ``method`` : str, the backend to use. One of ``'welch'`` (scipy), ``'torch'`` or ``'numpy'``. Default ``'welch'``. When ``'welch'`` is selected but scipy is not installed the torch backend is used instead, and the numpy backend is used as a further backup if the torch one fails.
        - ``window`` : str, the window applied to each segment. Supported values are ``'hann'``, ``'hamming'``, ``'blackman'``, ``'bartlett'`` and ``'boxcar'`` (rectangular). Default ``'hann'``.
        - ``nperseg`` : int or None, the length of each segment. If ``None`` (default) the whole signal is used as a single segment (i.e. a periodogram).
        - ``noverlap`` : int or None, the number of samples of overlap between segments. If ``None`` (default) it is set to ``nperseg // 2``.
        - ``nfft`` : int or None, the length of the FFT used. If ``None`` (default) it is set to ``nperseg``.
        - ``detrend`` : str or bool, the detrend applied to each segment. Either ``'constant'`` (remove the mean, default), ``'linear'`` (remove a linear trend) or ``False`` (no detrend).
        - ``scaling`` : str, the scaling of the spectrum. Either ``'density'`` (power spectral density, unit V**2/Hz, default) or ``'spectrum'`` (power spectrum, unit V**2).

    Returns
    -------
    checked_freq_config : dict
        A new dictionary (the input is never modified in place) with every key set to a valid value.
    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Default configuration

    default_freq_config = dict(
        method   = 'welch',
        window   = 'hann',
        nperseg  = None,
        noverlap = None,
        nfft     = None,
        detrend  = 'constant',
        scaling  = 'density',
    )

    # If nothing (or an empty dict) is passed simply return a copy of the default configuration.
    if freq_config is None or len(freq_config) == 0 :
        return dict(default_freq_config)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Merge the input with the default values

    # Work on a copy so the dictionary passed by the user is never modified.
    checked_freq_config = dict(freq_config)

    # Add the missing keys with their default value.
    for key, default_value in default_freq_config.items() :
        if key not in checked_freq_config or checked_freq_config[key] is None :
            # Note that some keys (nperseg, noverlap, nfft) accept None as a valid value, so for them None is kept.
            if key in ('nperseg', 'noverlap', 'nfft') and key in checked_freq_config :
                continue
            checked_freq_config[key] = default_value

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Validate the values

    if checked_freq_config['method'] not in ('welch', 'torch', 'numpy') :
        raise ValueError("freq_config['method'] must be 'welch', 'torch' or 'numpy'. Current value is {}".format(repr(checked_freq_config['method'])))

    if checked_freq_config['scaling'] not in ('density', 'spectrum') :
        raise ValueError("freq_config['scaling'] must be 'density' or 'spectrum'. Current value is {}".format(repr(checked_freq_config['scaling'])))

    return checked_freq_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Main function (time to frequency transformation)

def compute_frequency_spectrum(signal, fs : float = 1, freq_config : dict = None) :
    """
    Estimate the power spectral density (PSD) of a 1D EEG signal through the Welch method.

    The default backend is ``scipy.signal.welch``.
    If scipy is not installed the torch backend (based on ``torch.fft``) is used instead, and the numpy backend (based on ``numpy.fft``) is used as a further backup if the torch one fails.
    The backend can also be forced through ``freq_config['method']`` (``'welch'``, ``'torch'`` or ``'numpy'``).
    Regardless of the backend the returned arrays are always plain numpy arrays, so they can be plotted directly by matplotlib.
    With the default configuration (``nperseg = None``) the whole signal is used as a single segment, i.e. the estimate reduces to a simple periodogram.

    Parameters
    ----------
    signal : numpy.ndarray or array_like
        The 1D EEG signal (a single channel of a single trial).
    fs : float, optional
        The sampling frequency of the signal. Default is ``1`` (normalized frequency).
    freq_config : dict, optional
        The frequency configuration. See :func:`check_freq_config` for the list of the accepted keys and their default value.

    Returns
    -------
    freqs : numpy.ndarray
        The array with the (one-sided) frequencies. When ``fs = 1`` the values are the normalized frequencies.
    psd : numpy.ndarray
        The estimated power spectral density (unit V**2/Hz when ``scaling = 'density'``).
    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Input check and parameters resolution

    freq_config = check_freq_config(freq_config)

    # Force the signal to be a 1D numpy array of float.
    signal = np.asarray(signal, dtype = float).squeeze()
    if signal.ndim != 1 : raise ValueError("signal must be a 1D array. Current number of dimensions is {}".format(signal.ndim))
    n_samples = signal.shape[0]

    # Resolve the segmentation parameters (None means "use a sensible default that depends on the signal length").
    nperseg = freq_config['nperseg']
    if nperseg is None or nperseg > n_samples :
        # If no segment length is given (or it is longer than the signal) use the whole signal as a single segment.
        nperseg = n_samples

    noverlap = freq_config['noverlap']
    if noverlap is None :
        noverlap = nperseg // 2

    nfft = freq_config['nfft']
    if nfft is None :
        nfft = nperseg

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Compute the spectrum

    method = freq_config['method']

    if method == 'welch' :
        # Try the scipy backend first. If scipy is not installed fall back to the torch backend (the default fallback).
        try :
            from scipy import signal as scipy_signal

            freqs, psd = scipy_signal.welch(
                signal,
                fs            = fs,
                window        = freq_config['window'],
                nperseg       = nperseg,
                noverlap      = noverlap,
                nfft          = nfft,
                detrend       = freq_config['detrend'],
                scaling       = freq_config['scaling'],
                return_onesided = True,
            )
            # scipy already returns numpy arrays, so nothing to convert here.
            return freqs, psd
        except ImportError :
            warnings.warn("scipy is not installed: falling back to the torch implementation of the Welch method.")
            method = 'torch'

    if method == 'torch' :
        # torch backend. If torch is missing (or any other torch related problem happens) fall back to the numpy backend.
        try :
            return _welch_torch(
                signal,
                fs       = fs,
                window   = freq_config['window'],
                nperseg  = nperseg,
                noverlap = noverlap,
                nfft     = nfft,
                detrend  = freq_config['detrend'],
                scaling  = freq_config['scaling'],
            )
        except Exception as torch_error :
            warnings.warn("The torch backend failed ({}): falling back to the numpy implementation of the Welch method.".format(torch_error))
            method = 'numpy'

    # numpy backend (used both when explicitly requested and as the last backup).
    return _welch_numpy(
        signal,
        fs       = fs,
        window   = freq_config['window'],
        nperseg  = nperseg,
        noverlap = noverlap,
        nfft     = nfft,
        detrend  = freq_config['detrend'],
        scaling  = freq_config['scaling'],
    )

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Helper functions (numpy backend)

def _get_window(window : str, nperseg : int) :
    """
    Build a (periodic) window of length ``nperseg``.

    The periodic version of the window is used (as scipy does by default for spectral analysis) by sampling ``nperseg + 1`` points of the symmetric window and dropping the last one.

    Parameters
    ----------
    window : str
        The name of the window. Supported values are ``'hann'``, ``'hamming'``, ``'blackman'``, ``'bartlett'`` and ``'boxcar'`` (rectangular).
    nperseg : int
        The length of the window.

    Returns
    -------
    window_values : numpy.ndarray
        The array with the window values.
    """

    window = str(window).lower()

    # Rectangular window (no tapering).
    if window in ('boxcar', 'rectangular', 'none') :
        return np.ones(nperseg)

    # For the tapered windows use the periodic version (build a symmetric window of length nperseg + 1 and drop the last sample).
    if window == 'hann' :
        return np.hanning(nperseg + 1)[:-1]
    elif window == 'hamming' :
        return np.hamming(nperseg + 1)[:-1]
    elif window == 'blackman' :
        return np.blackman(nperseg + 1)[:-1]
    elif window == 'bartlett' :
        return np.bartlett(nperseg + 1)[:-1]
    else :
        raise ValueError("Unsupported window {}. Supported windows are 'hann', 'hamming', 'blackman', 'bartlett', 'boxcar'.".format(repr(window)))

def _detrend_segment(segment, detrend) :
    """
    Detrend a single segment.

    Parameters
    ----------
    segment : numpy.ndarray
        The segment to detrend.
    detrend : str or bool
        Either ``'constant'`` (remove the mean), ``'linear'`` (remove a linear trend) or ``False`` (no detrend).

    Returns
    -------
    detrended_segment : numpy.ndarray
        The detrended segment.
    """

    if detrend is False or detrend is None :
        return segment
    elif detrend == 'constant' :
        return segment - segment.mean()
    elif detrend == 'linear' :
        # Fit and subtract a first order polynomial (a straight line).
        x_axis = np.arange(segment.shape[0])
        linear_coefficients = np.polyfit(x_axis, segment, deg = 1)
        return segment - np.polyval(linear_coefficients, x_axis)
    else :
        raise ValueError("detrend must be 'constant', 'linear' or False. Current value is {}".format(repr(detrend)))

def _welch_numpy(signal, fs, window, nperseg, noverlap, nfft, detrend, scaling) :
    """
    Pure numpy implementation of the Welch method (used as a fallback when scipy is not installed).

    The signal is split into (possibly overlapping) segments, each segment is detrended, windowed and transformed with the FFT and finally the resulting periodograms are averaged.
    The output is one-sided and scaled exactly as :func:`scipy.signal.welch` so that the two backends can be used interchangeably.

    Parameters
    ----------
    signal : numpy.ndarray
        The 1D signal.
    fs : float
        The sampling frequency.
    window : str
        The name of the window (see :func:`_get_window`).
    nperseg : int
        The length of each segment.
    noverlap : int
        The number of overlapping samples between two consecutive segments.
    nfft : int
        The length of the FFT.
    detrend : str or bool
        The detrend applied to each segment (see :func:`_detrend_segment`).
    scaling : str
        Either ``'density'`` (unit V**2/Hz) or ``'spectrum'`` (unit V**2).

    Returns
    -------
    freqs : numpy.ndarray
        The one-sided array of frequencies.
    psd : numpy.ndarray
        The estimated power spectral density.
    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Build the window and compute the scaling factor

    window_values = _get_window(window, nperseg)

    if scaling == 'density' :
        # Scaling that gives a power spectral density (unit V**2/Hz).
        scale = 1.0 / (fs * np.sum(window_values ** 2))
    elif scaling == 'spectrum' :
        # Scaling that gives a power spectrum (unit V**2).
        scale = 1.0 / (np.sum(window_values) ** 2)
    else :
        raise ValueError("scaling must be 'density' or 'spectrum'. Current value is {}".format(repr(scaling)))

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Split the signal in segments and average the periodograms

    # Distance (in samples) between the start of two consecutive segments.
    step = nperseg - noverlap
    if step <= 0 :
        # Guard against a noverlap greater than or equal to nperseg.
        step = nperseg

    # Start index of every full length segment.
    segment_starts = list(range(0, len(signal) - nperseg + 1, step))
    if len(segment_starts) == 0 :
        # This should not happen because nperseg is clamped to the signal length, but keep it as a safety net.
        segment_starts = [0]

    psd_accumulator = np.zeros(nfft // 2 + 1)
    for start in segment_starts :
        segment = signal[start : start + nperseg]

        # Detrend, apply the window and move to the frequency domain.
        segment = _detrend_segment(segment, detrend)
        segment = segment * window_values
        spectrum = np.fft.rfft(segment, n = nfft)

        # Accumulate the (scaled) periodogram of the segment.
        psd_accumulator += (np.abs(spectrum) ** 2) * scale

    # Average over the segments.
    psd = psd_accumulator / len(segment_starts)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Convert to a one-sided spectrum

    # For a one-sided spectrum every frequency (but DC and, for an even nfft, the Nyquist one) has to be doubled to preserve the total power.
    if nfft % 2 == 0 :
        # Even nfft: the last bin is the Nyquist frequency and must not be doubled.
        psd[1:-1] *= 2
    else :
        # Odd nfft: there is no Nyquist bin, so every bin but DC is doubled.
        psd[1:] *= 2

    freqs = np.fft.rfftfreq(nfft, d = 1.0 / fs)

    return freqs, psd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Helper functions (torch backend)

def _detrend_segment_torch(segment, detrend) :
    """
    Detrend a single (torch) segment.

    It is the torch counterpart of :func:`_detrend_segment`.

    Parameters
    ----------
    segment : torch.Tensor
        The segment to detrend.
    detrend : str or bool
        Either ``'constant'`` (remove the mean), ``'linear'`` (remove a linear trend) or ``False`` (no detrend).

    Returns
    -------
    detrended_segment : torch.Tensor
        The detrended segment.
    """

    import torch

    if detrend is False or detrend is None :
        return segment
    elif detrend == 'constant' :
        return segment - segment.mean()
    elif detrend == 'linear' :
        # Fit and subtract a first order polynomial (a straight line) through a least squares solution.
        x_axis = torch.arange(segment.shape[0], dtype = segment.dtype, device = segment.device)
        design_matrix = torch.stack([x_axis, torch.ones_like(x_axis)], dim = 1)
        linear_coefficients = torch.linalg.lstsq(design_matrix, segment.unsqueeze(1)).solution
        trend = (design_matrix @ linear_coefficients).squeeze(1)
        return segment - trend
    else :
        raise ValueError("detrend must be 'constant', 'linear' or False. Current value is {}".format(repr(detrend)))

def _welch_torch(signal, fs, window, nperseg, noverlap, nfft, detrend, scaling) :
    """
    Implementation of the Welch method based on ``torch.fft``.

    It is the torch counterpart of :func:`_welch_numpy` and follows exactly the same steps (segmentation, detrend, windowing, FFT, averaging and one-sided scaling) so that the two backends return the same result.
    The computation is carried out with torch tensors, but the returned arrays are converted to numpy so that they can be plotted directly by matplotlib.

    Parameters
    ----------
    signal : numpy.ndarray
        The 1D signal.
    fs : float
        The sampling frequency.
    window : str
        The name of the window (see :func:`_get_window`).
    nperseg : int
        The length of each segment.
    noverlap : int
        The number of overlapping samples between two consecutive segments.
    nfft : int
        The length of the FFT.
    detrend : str or bool
        The detrend applied to each segment (see :func:`_detrend_segment_torch`).
    scaling : str
        Either ``'density'`` (unit V**2/Hz) or ``'spectrum'`` (unit V**2).

    Returns
    -------
    freqs : numpy.ndarray
        The one-sided array of frequencies.
    psd : numpy.ndarray
        The estimated power spectral density.
    """

    import torch

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Move the signal to torch and build the window

    # Use float64 to match (as close as possible) the precision of the scipy and numpy backends.
    signal_torch = torch.as_tensor(np.asarray(signal), dtype = torch.float64)

    # The window is built with numpy and then moved to torch to reuse the exact same window definition of the numpy backend.
    window_values = torch.as_tensor(_get_window(window, nperseg), dtype = torch.float64)

    if scaling == 'density' :
        # Scaling that gives a power spectral density (unit V**2/Hz).
        scale = 1.0 / (fs * torch.sum(window_values ** 2))
    elif scaling == 'spectrum' :
        # Scaling that gives a power spectrum (unit V**2).
        scale = 1.0 / (torch.sum(window_values) ** 2)
    else :
        raise ValueError("scaling must be 'density' or 'spectrum'. Current value is {}".format(repr(scaling)))

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Split the signal in segments and average the periodograms

    # Distance (in samples) between the start of two consecutive segments.
    step = nperseg - noverlap
    if step <= 0 :
        # Guard against a noverlap greater than or equal to nperseg.
        step = nperseg

    # Start index of every full length segment.
    segment_starts = list(range(0, signal_torch.shape[0] - nperseg + 1, step))
    if len(segment_starts) == 0 :
        # This should not happen because nperseg is clamped to the signal length, but keep it as a safety net.
        segment_starts = [0]

    psd_accumulator = torch.zeros(nfft // 2 + 1, dtype = torch.float64)
    for start in segment_starts :
        segment = signal_torch[start : start + nperseg]

        # Detrend, apply the window and move to the frequency domain.
        segment = _detrend_segment_torch(segment, detrend)
        segment = segment * window_values
        spectrum = torch.fft.rfft(segment, n = nfft)

        # Accumulate the (scaled) periodogram of the segment.
        psd_accumulator += (torch.abs(spectrum) ** 2) * scale

    # Average over the segments.
    psd = psd_accumulator / len(segment_starts)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Convert to a one-sided spectrum

    # For a one-sided spectrum every frequency (but DC and, for an even nfft, the Nyquist one) has to be doubled to preserve the total power.
    if nfft % 2 == 0 :
        # Even nfft: the last bin is the Nyquist frequency and must not be doubled.
        psd[1:-1] *= 2
    else :
        # Odd nfft: there is no Nyquist bin, so every bin but DC is doubled.
        psd[1:] *= 2

    freqs = torch.fft.rfftfreq(nfft, d = 1.0 / fs)

    # Return plain numpy arrays so that the output can be plotted directly by matplotlib.
    return freqs.numpy(), psd.numpy()

#%% End file
