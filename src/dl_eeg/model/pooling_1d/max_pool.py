"""
Max pooling - the un-anti-aliased nonlinear reference arm.
"""

from __future__ import annotations

import torch
from torch import nn

from dl_eeg.model.pooling_1d.support_pooling import _as_decimation_factor, _get


# =================================================================================
# Max pooling - nonlinear reference arm
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
