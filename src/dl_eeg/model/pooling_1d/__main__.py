"""
Runnable entry point: ``python -m dl_eeg.model.pooling_1d``.

Runs the self-test and then prints the figures-of-merit table for the two EEGNet
pooling stages, exactly as running the original single-file module directly did.
"""

from __future__ import annotations

import warnings

from dl_eeg.model.pooling_1d import get_pooling_layer, self_test

if __name__ == "__main__":  # pragma: no cover
    self_test()

    print("\n" + "=" * 78)
    print("Figures of merit at the two EEGNet pooling stages (BCI IV-2a, 250 Hz)")
    print("=" * 78)
    print("Primary metric is 'leak%' = fraction of filter energy above pi/M that gets")
    print("folded back by subsampling (lower is better). 'att@pi/M' is measured at the")
    print("folding frequency itself, so it reads conservatively for cutoff_scale < 1.")
    print("'minT' = shortest input this kernel can be used on with reflect padding.")
    header = (f"{'layer':34s} {'M':>2s} {'L':>4s} {'leak%':>8s} {'att@pi/M':>9s} "
              f"{'delay':>7s} {'minT':>5s}")
    for M in (4, 8):
        fs_in = 250.0 if M == 4 else 62.5
        print(f"\n-- decimation M = {M}: input rate {fs_in:g} Hz -> output rate "
              f"{fs_in / M:g} Hz, folding frequency {fs_in / (2 * M):g} Hz --")
        print(header)
        arms = [
            ("AvgPool (boxcar)", {"pooling_type": "avg"}),
            ("BlurPool binomial L=3 (2D default)", {"pooling_type": "blur", "kernel_size": 3,
                                                    "warn_if_worse_than_boxcar": False}),
            ("BlurPool binomial L=4M+1", {"pooling_type": "blur"}),
            ("BlurPool gaussian L=4M+1", {"pooling_type": "blur", "kernel": "gaussian"}),
            ("SincPool Kaiser N=8M+1", {"pooling_type": "sinc"}),
            # A spec-driven design; max_taps keeps it usable on a 250-sample stage-2
            # input (an uncapped 60 dB / 25% design asks for 259 taps at M=8).
            ("SincPool 60dB/25% spec (capped)", {"pooling_type": "sinc",
                                                 "stopband_attenuation_db": 60,
                                                 "transition_width": 0.25,
                                                 "max_taps": 61}),
            ("DWTPool db4", {"pooling_type": "dwt", "wavelet": "db4"}),
            ("DWTPool sym4", {"pooling_type": "dwt", "wavelet": "sym4"}),
        ]
        for label, cfg in arms:
            cfg = dict(cfg, decimation_factor=M)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                layer = get_pooling_layer(cfg)
            s = layer.summary()
            # reflect padding needs pad < T, with pad = (L - 1) // 2
            min_T = (s["kernel_length"] - 1) // 2 + 1
            print(f"{label:34s} {s['decimation_factor']:2d} {s['kernel_length']:4d} "
                  f"{s['stopband_leakage_pct']:8.3f} {s['stopband_attenuation_db']:9.1f} "
                  f"{s['group_delay_samples']:7.2f} {min_T:5d}")
