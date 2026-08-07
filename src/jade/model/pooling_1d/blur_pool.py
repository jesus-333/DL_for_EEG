"""
BlurPool - fixed heuristic kernel (binomial / Gaussian / boxcar).
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Third party imports
import numpy as np

# Internal imports
from jade.model.pooling_1d.support_pooling import (
    _fir_decimation_pool_1d,
    _binomial_kernel,
    _gaussian_kernel,
    _get,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# BlurPool - fixed heuristic kernel (binomial / Gaussian / boxcar)

class blur_pool_1d(_fir_decimation_pool_1d) :
    """
    Fixed-kernel anti-aliased pooling: the 1D analogue of Zhang's BlurPool (2019).

    Same structure as average pooling, better kernel.
    The binomial (Pascal-row) kernel has response

        |H(w)| = |cos(w / 2)| ** (L - 1)

    which is monotonically decreasing with NO sidelobes and no passband ripple - a qualitative improvement over the boxcar's oscillating Dirichlet response.
    The cost is a slow roll-off: suppression is bought only through length.

    THE ONE THING TO GET RIGHT
    --------------------------
    The kernel length must scale with the decimation factor.
    ``|cos(w/2)|**(L-1)`` has a bandwidth set by L alone, while the cutoff you need is ``pi / M``.
    The canonical 2D kernels ``[1,2,1]`` (L=3) and ``[1,4,6,4,1]`` (L=5) were chosen for stride-2 layers; reusing them at M=8 gives *worse* aliasing than doing nothing beyond averaging:

        M = 4   boxcar leakage 21.6%   |  L=3: 39.7%   L=5: 24.5%   L=9: 10.6%
        M = 8   boxcar leakage 22.4%   |  L=3: 67.5%   L=5: 56.5%   L=25: 17.0%

    The default here is therefore ``kernel_size = 4*M + 1``, and a warning fires if the configuration is worse than the boxcar.
    If you want to reproduce the literal 2D BlurPool setting, pass ``kernel_size=3`` explicitly and expect the warning.

    Config
    ------
    decimation_factor : int or (1, int)     [required]
    kernel_size : int, default 4*M + 1
        Length L of the blur kernel.
        Odd values keep the kernel symmetric about a sample, giving exactly linear phase and zero net delay.
    kernel : {'binomial', 'gaussian', 'boxcar'}, default 'binomial'
        'binomial' - Pascal row; the standard BlurPool choice.
        'gaussian' - truncated Gaussian; no sidelobes at all, slowest roll-off.
        'boxcar'   - moving average of length ``kernel_size``; provided so that ``blur_pool_1d(kernel='boxcar', kernel_size=M)`` reproduces :class:`~jade.model.pooling_1d.avg_pool.avg_pool_1d` and the family is provably nested.
    sigma : float, optional
        Standard deviation for the Gaussian kernel; defaults to ``kernel_size / 6``.
    (plus all shared keys)
    """

    _kind = "binomial"

    def _build_kernel(self, config : dict) -> np.ndarray :
        """
        Build the requested fixed kernel (binomial, Gaussian or boxcar).

        Parameters
        ----------
        config : dict
            The layer configuration.

        Returns
        -------
        numpy.ndarray
            The chosen kernel.
        """

        M = self.decimation_factor
        kind = str(_get(config, "kernel", "binomial")).lower()

        # Default length scales with M - see the class docstring.
        length = int(_get(config, "kernel_size", 4 * M + 1, aliases = ("blur_kernel_size", "L")))
        if length < 1 :
            raise ValueError("kernel_size must be >= 1")

        self._kind = kind
        if kind == "binomial" :
            return _binomial_kernel(length)
        if kind == "gaussian" :
            return _gaussian_kernel(length, _get(config, "sigma", None))
        if kind == "boxcar" :
            return np.ones(length, dtype = np.float64) / length
        raise ValueError("unknown kernel {}; expected 'binomial', 'gaussian' or 'boxcar'".format(repr(kind)))
