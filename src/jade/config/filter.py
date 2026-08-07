"""
Filter configuration.

:class:`~jade.config.filter.filter_config` collects the parameters of a single filtering operation as applied by moabb.
moabb filters the raw signal by calling MNE's ``Raw.filter`` (a band-pass by default: ``raw.filter(l_freq = fmin, h_freq = fmax, method = "iir", picks = "data")``); a notch uses ``Raw.notch_filter``.
This config therefore mirrors those MNE parameters.

The ``filter_type`` selects the kind of filter and dictates which cutoff(s) are required; every other parameter has a sensible default.
Concretely:

* ``bandpass``  -> requires ``l_freq`` and ``h_freq`` (``l_freq < h_freq``)
* ``bandstop``  -> requires ``l_freq`` and ``h_freq`` (``l_freq > h_freq``)
* ``lowpass``   -> requires ``h_freq`` (the cutoff); ``l_freq`` must be unset
* ``highpass``  -> requires ``l_freq`` (the cutoff); ``h_freq`` must be unset
* ``notch``     -> requires ``freqs`` (frequency or list of frequencies to remove)

All frequencies must be strictly positive.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from dataclasses import dataclass
from typing import List, Optional, Union

# Internal imports
from jade.config.generic import generic_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module constants

# Filter kinds understood by the filter config.
FILTER_TYPES = ("bandpass", "bandstop", "lowpass", "highpass", "notch")
# Filtering backends accepted by MNE / moabb.
FILTER_METHODS = ("iir", "fir")
# Linear-phase options accepted by MNE.
FILTER_PHASES = ("zero", "zero-double", "minimum", "minimum-half")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Filter config class

@dataclass
class filter_config(generic_config) :
    """
    Parameters of one moabb/MNE filtering operation.

    Parameters
    ----------
    filter_type : str, default 'bandpass'
        One of ``FILTER_TYPES``. It determines which cutoff(s) are required.
    l_freq : float or None
        Low transition-band edge (the high-pass cutoff).
        Required for ``highpass`` and one edge of ``bandpass``/``bandstop``.
        It must be unset for ``lowpass``.
    h_freq : float or None
        High transition-band edge (the low-pass cutoff).
        Required for ``lowpass`` and one edge of ``bandpass``/``bandstop``.
        It must be unset for ``highpass``.
    freqs : float, list of float or None
        Frequencies to remove; required for ``notch`` (and ignored otherwise).
    method : str, default 'iir'
        Filtering backend, ``'iir'`` (moabb's default) or ``'fir'``.
    iir_params : dict or None
        Parameters for the IIR design, forwarded to MNE (e.g. ``{'order' : 4, 'ftype' : 'butter'}``).
        ``None`` lets MNE pick its defaults.
    filter_length : str or int, default 'auto'
        FIR filter length (MNE ``filter_length``). Only relevant for ``method = 'fir'``.
    l_trans_bandwidth : str or float, default 'auto'
        Width of the lower transition band (MNE).
    h_trans_bandwidth : str or float, default 'auto'
        Width of the upper transition band (MNE).
    phase : str, default 'zero'
        Filter phase (MNE), one of ``FILTER_PHASES``.
    fir_window : str, default 'hamming'
        FIR window (MNE). Only relevant for ``method = 'fir'``.
    fir_design : str, default 'firwin'
        FIR design (MNE). Only relevant for ``method = 'fir'``.
    picks : str or None, default 'data'
        Channels to filter (MNE ``picks``). moabb uses ``'data'``.
    """

    filter_type : str = "bandpass"

    l_freq : Optional[float] = None
    h_freq : Optional[float] = None
    freqs  : Optional[Union[float, List[float]]] = None

    method            : str = "iir"
    iir_params        : Optional[dict] = None
    filter_length     : Union[str, int] = "auto"
    l_trans_bandwidth : Union[str, float] = "auto"
    h_trans_bandwidth : Union[str, float] = "auto"
    phase             : str = "zero"
    fir_window        : str = "hamming"
    fir_design        : str = "firwin"
    picks             : Optional[str] = "data"


    def __post_init__(self) -> None :
        """
        Validate the filter parameters after initialization.
        """

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Normalize and check the categorical parameters

        filter_type = str(self.filter_type).lower()
        self.filter_type = filter_type
        if filter_type not in FILTER_TYPES :
            raise ValueError("filter_type must be one of {}, got {}".format(FILTER_TYPES, self.filter_type))

        self.method = str(self.method).lower()
        if self.method not in FILTER_METHODS :
            raise ValueError("method must be one of {}, got {}".format(FILTER_METHODS, self.method))

        if self.phase not in FILTER_PHASES :
            raise ValueError("phase must be one of {}, got {}".format(FILTER_PHASES, self.phase))

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Check the positivity of every provided cutoff

        for name in ("l_freq", "h_freq") :
            value = getattr(self, name)
            if value is not None and value <= 0 :
                raise ValueError("{} must be a positive number, got {}".format(name, value))

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Check the type-specific required cutoffs

        if filter_type == "bandpass" :
            self._require("l_freq")
            self._require("h_freq")
            if self.l_freq >= self.h_freq :
                raise ValueError("bandpass requires l_freq < h_freq, got l_freq = {}, h_freq = {}".format(self.l_freq, self.h_freq))
        elif filter_type == "bandstop" :
            self._require("l_freq")
            self._require("h_freq")
            if self.l_freq <= self.h_freq :
                raise ValueError("bandstop requires l_freq > h_freq, got l_freq = {}, h_freq = {}".format(self.l_freq, self.h_freq))
        elif filter_type == "lowpass" :
            self._require("h_freq")
            if self.l_freq is not None :
                raise ValueError("lowpass takes only h_freq (the cutoff); leave l_freq unset")
        elif filter_type == "highpass" :
            self._require("l_freq")
            if self.h_freq is not None :
                raise ValueError("highpass takes only l_freq (the cutoff); leave h_freq unset")
        elif filter_type == "notch" :
            if self.freqs is None :
                raise ValueError("notch requires 'freqs' (a frequency or list of frequencies)")
            freqs = self.freqs if isinstance(self.freqs, (list, tuple)) else [self.freqs]
            for single_freq in freqs :
                if single_freq is None or single_freq <= 0 :
                    raise ValueError("every notch frequency must be positive, got {}".format(single_freq))


    def _require(self, name : str) -> None :
        """
        Raize a ``ValueError`` if the cutoff ``name`` is not set for the current filter type.

        Parameters
        ----------
        name : str
            The name of the cutoff attribute that must be set (``'l_freq'`` or ``'h_freq'``).
        """

        if getattr(self, name) is None :
            raise ValueError("filter_type = {} requires {} to be set (the cutoff frequency)".format(repr(self.filter_type), repr(name)))
