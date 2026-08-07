"""
Average pooling - the boxcar baseline of the anti-aliased pooling family.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Third party imports
import numpy as np

# Internal imports
from jade.model.pooling_1d.support_pooling import _fir_decimation_pool_1d, _multichannel_pool_1d

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Average pooling - the boxcar baseline

class avg_pool_1d(_fir_decimation_pool_1d) :
    """
    Standard average pooling, expressed explicitly as boxcar-filter + subsample.

    This is the BASELINE of the study, and writing it in the same framework as the others makes the comparison airtight: it differs from :class:`~jade.model.pooling_1d.blur_pool.blur_pool_1d` only in the choice of ``h``.

    The boxcar of length M has the closed-form response (Dirichlet kernel)

        H(w) = (1/M) * sin(w M / 2) / sin(w / 2) * exp(-j w (M-1)/2)

    whose first zero sits exactly at the post-decimation Nyquist ``w = 2 pi / M``.
    That placement is why averaging is not a *disaster* as an anti-aliasing filter - but its sidelobes decay only as 1/w, with the first sidelobe at about -13 dB, so roughly 20% of its energy lies above ``pi/M`` and is folded back on subsampling.
    That residual folding is the phenomenon this study measures.

    Config
    ------
    decimation_factor : int or (1, int)     [required]
    (plus all shared keys: padding_mode, phase, gain, ...)

    Note
    ----
    With ``padding_mode='constant'`` and ``phase='center'`` this layer is numerically identical to ``nn.AvgPool1d(M, stride=M)`` on the interior; boundary samples differ because ``nn.AvgPool1d`` does not pad at all.
    ``self_test()`` verifies the exact agreement.
    """

    _kind = "boxcar"

    def _build_kernel(self, config : dict) -> np.ndarray :
        """
        Build the boxcar kernel of length equal to the decimation factor.

        Note the length therefore scales with M automatically - the property that short fixed binomial kernels lack.

        Parameters
        ----------
        config : dict
            The layer configuration (unused here beyond the decimation factor).

        Returns
        -------
        numpy.ndarray
            The boxcar kernel.
        """

        M = self.decimation_factor
        return np.ones(M, dtype = np.float64) / M


    def _check_against_boxcar(self) -> None :
        """
        Do nothing: this layer *is* the boxcar, so there is nothing to warn about.
        """

        return

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Multichannel variant

class avg_pool_1d_multichannel(_multichannel_pool_1d) :
    """
    Channel-wise :class:`avg_pool_1d` for multichannel EEG tensors of shape ``(B, D, C, T)``.

    Applies :class:`avg_pool_1d` independently to every EEG channel; see :class:`~jade.model.pooling_1d.support_pooling._multichannel_pool_1d` for the mechanism.
    Takes the same ``config`` dict as :class:`avg_pool_1d`.
    """

    def __init__(self, config : dict) :
        super().__init__(avg_pool_1d(config))
