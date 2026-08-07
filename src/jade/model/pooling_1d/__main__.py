"""
Runnable entry point: ``python -m jade.model.pooling_1d``.

It runs the self-test and then prints the figures-of-merit table for the two EEGNet pooling stages, exactly as running the original single-file module directly did.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Full module imports
import warnings

# Internal imports
from jade.model.pooling_1d import get_pooling_layer, self_test

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Main

if __name__ == "__main__" :  # pragma: no cover
    self_test()

    print("\n" + "=" * 78)
    print("Figures of merit at the two EEGNet pooling stages (BCI IV-2a, 250 Hz)")
    print("=" * 78)
    print("Primary metric is 'leak%' = fraction of filter energy above pi/M that gets")
    print("folded back by subsampling (lower is better). 'att@pi/M' is measured at the")
    print("folding frequency itself, so it reads conservatively for cutoff_scale < 1.")
    print("'minT' = shortest input this kernel can be used on with reflect padding.")

    header = "{:34s} {:>2s} {:>4s} {:>8s} {:>9s} {:>7s} {:>5s}".format(
        "layer", "M", "L", "leak%", "att@pi/M", "delay", "minT"
    )

    for M in (4, 8) :
        fs_in = 250.0 if M == 4 else 62.5
        print("\n-- decimation M = {}: input rate {:g} Hz -> output rate {:g} Hz, folding frequency {:g} Hz --".format(
            M, fs_in, fs_in / M, fs_in / (2 * M)
        ))
        print(header)

        arms = [
            ("avg_pool (boxcar)", {"pooling_type" : "avg"}),
            ("blur_pool binomial L=3 (2D default)", {"pooling_type" : "blur", "kernel_size" : 3,
                                                     "warn_if_worse_than_boxcar" : False}),
            ("blur_pool binomial L=4M+1", {"pooling_type" : "blur"}),
            ("blur_pool gaussian L=4M+1", {"pooling_type" : "blur", "kernel" : "gaussian"}),
            ("sinc_pool Kaiser N=8M+1", {"pooling_type" : "sinc"}),
            # A spec-driven design; max_taps keeps it usable on a 250-sample stage-2 input (an uncapped 60 dB / 25% design asks for 259 taps at M=8).
            ("sinc_pool 60dB/25% spec (capped)", {"pooling_type" : "sinc",
                                                  "stopband_attenuation_db" : 60,
                                                  "transition_width" : 0.25,
                                                  "max_taps" : 61}),
            ("dwt_pool db4", {"pooling_type" : "dwt", "wavelet" : "db4"}),
            ("dwt_pool sym4", {"pooling_type" : "dwt", "wavelet" : "sym4"}),
        ]

        for label, cfg in arms :
            cfg = dict(cfg, decimation_factor = M)
            with warnings.catch_warnings() :
                warnings.simplefilter("ignore", RuntimeWarning)
                layer = get_pooling_layer(cfg)
            summary = layer.summary()
            # reflect padding needs pad < T, with pad = (L - 1) // 2
            min_T = (summary["kernel_length"] - 1) // 2 + 1
            print("{:34s} {:2d} {:4d} {:8.3f} {:9.1f} {:7.2f} {:5d}".format(
                label, summary["decimation_factor"], summary["kernel_length"],
                summary["stopband_leakage_pct"], summary["stopband_attenuation_db"],
                summary["group_delay_samples"], min_T
            ))
