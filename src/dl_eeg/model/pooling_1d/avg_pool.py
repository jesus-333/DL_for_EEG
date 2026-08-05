"""
Average pooling - the boxcar baseline of the anti-aliased pooling family.
"""

from __future__ import annotations

import numpy as np

from dl_eeg.model.pooling_1d.support_pooling import _FIRDecimationPool1d


# =================================================================================
# Average pooling - the boxcar baseline
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
