"""
Max pooling - the un-anti-aliased nonlinear reference arm.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Third party imports
import torch
from torch import nn

# Internal imports
from jade.model.pooling_1d.support_pooling import _as_decimation_factor, _get, _multichannel_pool_1d

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Max pooling - nonlinear reference arm

class max_pool_1d(nn.Module) :
    """
    Plain max pooling. Included only as the *un-anti-aliased nonlinear reference*.

    Max pooling does not fit the "filter then subsample" decomposition, because the max operator is nonlinear: it has no frequency response, and worse, it can *create* new high-frequency content that was not present in its input (the envelope of a rectified oscillation contains harmonics).
    So it both fails to suppress out-of-band energy and manufactures more of it before subsampling.
    Several works report that average pooling anti-aliases *better* than max pooling for exactly this reason.

    It is here so the ablation can report the operator most widely used in practice, and so the ordering (max < avg < properly filtered) can be demonstrated rather than assumed.

    Config
    ------
    decimation_factor : int or (1, int)     [required]
    ceil_mode : bool, default False
        Passed to ``nn.MaxPool1d``. Left at False so the output length is ``T // M``, matching every other layer here.
    """

    def __init__(self, config : dict) :
        super().__init__()

        # Get the decimation factor (required).
        raw = _get(
            config, "decimation_factor",
            aliases = ("pool_size", "factor", "p_kernel"),
        )
        if raw is None :
            raise KeyError("config must contain 'decimation_factor' (or an alias)")
        self.decimation_factor = _as_decimation_factor(raw)

        # Build the underlying torch max pooling layer.
        self.pool = nn.MaxPool1d(
            kernel_size = self.decimation_factor,
            stride = self.decimation_factor,
            ceil_mode = bool(_get(config, "ceil_mode", False)),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        """
        Apply max pooling.

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
            raise ValueError(f"expected (B, D, T), got {tuple(x.shape)}")
        return self.pool(x)

    def summary(self) -> dict :
        """
        Figures of merit for the max pooling layer.

        No linear frequency response exists, so the spectral metrics are undefined (NaN).

        Returns
        -------
        dict
            A dictionary of the layer's figures of merit.
        """

        return {
            "layer"                    : "max_pool_1d",
            "kind"                     : "nonlinear",
            "decimation_factor"        : self.decimation_factor,
            "kernel_length"            : self.decimation_factor,
            "stopband_leakage_pct"     : float("nan"),
            "stopband_attenuation_db"  : float("nan"),
            "passband_ripple_db"       : float("nan"),
            "group_delay_samples"      : 0.0,
            "dc_gain"                  : 1.0,
        }
    def plot_frequency_response(self, *args, **kwargs) :
        """
        Not available: max pooling is nonlinear and has no frequency response.

        Unlike the FIR pooling layers (avg / blur / sinc / dwt), :class:`max_pool_1d` wraps :class:`torch.nn.MaxPool1d`, a nonlinear operator, so it has no linear/LTI frequency response to plot (its :meth:`summary` already reports ``NaN`` for the spectral figures of merit).
        The method exists so every pooling layer exposes the same interface, but calling it always raises.

        Raises
        ------
        NotImplementedError
            Always, because a nonlinear operator has no frequency response.
        """

        raise NotImplementedError(
            "max_pool_1d is nonlinear (nn.MaxPool1d): it has no frequency response to plot. "
            "Use one of the FIR pooling layers (avg / blur / sinc / dwt) for a frequency-response plot."
        )

    def extra_repr(self) -> str :
        """
        Short description used by :meth:`torch.nn.Module.__repr__`.
        """

        return f"M={self.decimation_factor} (nonlinear, no anti-aliasing)"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Multichannel variant

class max_pool_1d_multichannel(_multichannel_pool_1d) :
    """
    Channel-wise :class:`max_pool_1d` for multichannel EEG tensors of shape ``(B, D, C, T)``.

    Applies :class:`max_pool_1d` independently to every EEG channel; see :class:`~jade.model.pooling_1d.support_pooling._multichannel_pool_1d` for the mechanism.
    Takes the same ``config`` dict as :class:`max_pool_1d`.
    """

    def __init__(self, config : dict) :
        super().__init__(max_pool_1d(config))
