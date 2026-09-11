"""
End-to-end check of the braindecode pooling extension (:mod:`jade.model.braindecode_pooling_extension`).

For a few braindecode models it:
- builds the model through :class:`~jade.model.braindecode_pooling_extension.braindecode_pooling_wrapper` with
  several jade pooling operators (avg / blur / sinc / dwt / max),
- reports the replaced pooling layers, the parameter count and the output shape,
- checks that no stock pooling layer is left and that the model still runs,
- for a non-overlapping model checks that the swap preserves the output shape of the stock model
  (the shape-preserving property the jade pooling family guarantees),
- for the overlapping ShallowFBCSPNet checks that the trailing classifier was re-sized so the model still runs,
- checks the ``keep_pool_nonlin`` flag and the standard ``get_model`` factory path.

It needs no dataset (dummy tensors are used) but does need braindecode installed:
``pip install braindecode`` (or ``pip install jade[braindecode]``).

Run with:
    python test/custom/braindecode_pooling.py

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

# Full module imports
import warnings

# External libraries
import torch
from torch import nn

# Internal libraries
from jade.model import get_model
from jade.model.support_functions import count_trainable_parameters
from jade.config.model import braindecode_wrapper_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

# EEG geometry of the dummy input (batch, channels, time samples).
BATCH    = 4
N_CHANS  = 22
N_TIMES  = 1000
N_OUTPUT = 4

# Pooling operators tried on each model. 'dwt' needs a power-of-two decimation factor, so it is only used on the
# models whose pooling factors are all powers of two (see the per-model lists below).
POOLING_TYPES_POW2 = ["avg", "blur", "sinc", "dwt", "max"]
POOLING_TYPES_ANY  = ["avg", "blur", "sinc", "max"]

# The models exercised in detail: name -> (list of pooling types, is the stock pooling non-overlapping?).
MODELS = {
    "EEGNet"          : (POOLING_TYPES_POW2, True),   # AvgPool2d (1, 4) then (1, 8) - non-overlapping, axis 3
    "Deep4Net"        : (POOLING_TYPES_ANY,  True),   # 4x MaxPool2d (3, 1) - non-overlapping, axis 2
    "ShallowFBCSPNet" : (POOLING_TYPES_ANY,  False),  # AvgPool2d (75, 1) stride (15, 1) - overlapping, axis 2
}

# A wider set of principal braindecode models, checked more lightly (build + one forward) to back the
# "handle all the principal models" requirement. They cover both pooled axes and 3D/4D internal layouts;
# all take the same 3D (B, n_chans, n_times) input (their ensuredims / dimshuffle front layers adapt it).
BROAD_MODELS = ["EEGITNet", "EEGInceptionERP", "EEGNeX", "EEGTCNet", "ATCNet"]

_POOL_TYPES = (nn.AvgPool1d, nn.MaxPool1d, nn.AvgPool2d, nn.MaxPool2d)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Helpers

def _stock_output_shape(model_name : str, x : torch.Tensor) -> tuple :
    """
    Build the untouched braindecode model and return its output shape (baseline for the shape-preserving check).
    """

    import braindecode.models as braindecode_models

    model = getattr(braindecode_models, model_name)(n_chans = N_CHANS, n_outputs = N_OUTPUT, n_times = N_TIMES)
    model.eval()
    with torch.no_grad() :
        return tuple(model(x).shape)


def check_model(model_name : str, pooling_types : list, non_overlapping : bool, x : torch.Tensor) -> None :
    """
    Build ``model_name`` with every pooling type and run the checks described in the module docstring.
    """

    print(f"\n{'=' * 70}\n{model_name}")

    stock_shape = _stock_output_shape(model_name, x)
    print(f"  stock output shape : {stock_shape}")

    for pooling_type in pooling_types :
        config = {
            "name"                   : "braindecode",
            "braindecode_model_name" : model_name,
            "model_kwargs"           : {"n_chans" : N_CHANS, "n_outputs" : N_OUTPUT, "n_times" : N_TIMES},
            "pooling"                : {"pooling_type" : pooling_type},
        }

        # Build the wrapper (the shape-preserving warnings of the pooling family are not interesting here).
        with warnings.catch_warnings() :
            warnings.simplefilter("ignore")
            model = get_model(config)

        model.eval()
        with torch.no_grad() :
            y = model(x)

        # At least one pooling layer must have been replaced, and each swapped path must no longer hold a stock
        # pooling layer (a jade replacement can legitimately contain a nested nn.*Pool1d of its own, so the check
        # is per swapped path, not a global scan).
        assert len(model.swaps) >= 1, f"{model_name}/{pooling_type}: no pooling layer was replaced"
        for swap in model.swaps :
            current = model.model.get_submodule(swap["name"])
            assert not isinstance(current, _POOL_TYPES), f"{model_name}/{pooling_type}: {swap['name']!r} still a stock pool"

        # The model must run and (for a non-overlapping model) keep the stock output shape.
        assert tuple(y.shape) == stock_shape, f"{model_name}/{pooling_type}: output {tuple(y.shape)} != stock {stock_shape}"

        # classify() must return one class index per sample.
        assert tuple(model.classify(x).shape) == (x.shape[0],), f"{model_name}/{pooling_type}: bad classify() shape"

        factors     = [swap["factor"] for swap in model.swaps]
        overlapping = any(swap["overlapping"] for swap in model.swaps)
        n_params    = count_trainable_parameters(model)
        tag         = "overlapping (trailing classifier re-sized)" if overlapping else "non-overlapping"
        print(f"  [OK] {pooling_type:5s} : {len(model.swaps)} layer(s) swapped, factors {factors}, "
              f"{tag}, out {tuple(y.shape)}, params {n_params}")

    # Sanity: the flagged expectation matches what the wrapper detected.
    with warnings.catch_warnings() :
        warnings.simplefilter("ignore")
        probe = get_model({**config, "pooling" : {"pooling_type" : "avg"}})
    detected_overlap = any(swap["overlapping"] for swap in probe.swaps)
    assert detected_overlap == (not non_overlapping), f"{model_name}: overlap detection mismatch"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Main

def main() -> None :
    """
    Run every check and print a final confirmation.
    """

    torch.manual_seed(0)
    x = torch.randn(BATCH, N_CHANS, N_TIMES)

    # 1. Per-model pooling swap checks.
    for model_name, (pooling_types, non_overlapping) in MODELS.items() :
        check_model(model_name, pooling_types, non_overlapping, x)

    # 1b. Broad coverage: build a wider set of principal models (avg pooling) and confirm each runs.
    print(f"\n{'=' * 70}\nbroad model coverage (avg pooling, 3D input)")
    for model_name in BROAD_MODELS :
        config = {
            "name"                   : "braindecode",
            "braindecode_model_name" : model_name,
            "model_kwargs"           : {"n_chans" : N_CHANS, "n_outputs" : N_OUTPUT, "n_times" : N_TIMES},
            "pooling"                : {"pooling_type" : "avg"},
        }
        with warnings.catch_warnings() :
            warnings.simplefilter("ignore")
            model = get_model(config)          # validate_forward (default) already ran a probe forward
        model.eval()
        with torch.no_grad() :
            y = model(x)
        assert tuple(y.shape) == (BATCH, N_OUTPUT), f"{model_name}: output {tuple(y.shape)} != {(BATCH, N_OUTPUT)}"
        print(f"  [OK] {model_name:18s} {len(model.swaps)} layer(s) swapped, out {tuple(y.shape)}")

    # 1c. The build-time validation forward turns an unsupported swap into a clear error rather than a cryptic one
    # (EEGConformer: an overlapping pooling feeding a transformer head the wrapper cannot auto-resize).
    print(f"\n{'=' * 70}\nvalidate_forward turns an unsupported case into a clear error")
    try :
        with warnings.catch_warnings() :
            warnings.simplefilter("ignore")
            get_model({
                "name"                   : "braindecode",
                "braindecode_model_name" : "EEGConformer",
                "model_kwargs"           : {"n_chans" : N_CHANS, "n_outputs" : N_OUTPUT, "n_times" : N_TIMES},
                "pooling"                : {"pooling_type" : "sinc"},
            })
        raise AssertionError("EEGConformer with an overlapping-pooling swap was expected to fail validation")
    except RuntimeError as error :
        assert "probe forward" in str(error), "expected the wrapper's clear validation error"
        print("  [OK] EEGConformer/sinc raised the wrapper's actionable validation error")

    # 2. keep_pool_nonlin flag (ShallowFBCSPNet wraps the pooling with a safe_log 'pool_nonlin_exp').
    print(f"\n{'=' * 70}\nkeep_pool_nonlin flag (ShallowFBCSPNet)")
    base = {
        "name"                   : "braindecode",
        "braindecode_model_name" : "ShallowFBCSPNet",
        "model_kwargs"           : {"n_chans" : N_CHANS, "n_outputs" : N_OUTPUT, "n_times" : N_TIMES},
        "pooling"                : {"pooling_type" : "avg"},
    }
    with warnings.catch_warnings() :
        warnings.simplefilter("ignore")
        kept    = get_model({**base, "keep_pool_nonlin" : True})
        dropped = get_model({**base, "keep_pool_nonlin" : False})
    assert type(kept.model.pool_nonlin_exp).__name__ != "Identity", "pool_nonlin_exp should be kept"
    assert isinstance(dropped.model.pool_nonlin_exp, nn.Identity), "pool_nonlin_exp should be dropped"
    print(f"  [OK] keep -> {type(kept.model.pool_nonlin_exp).__name__}, "
          f"drop -> {type(dropped.model.pool_nonlin_exp).__name__} (dropped {dropped.dropped_pool_nonlin})")

    # 3. Factory + config round-trip.
    print(f"\n{'=' * 70}\nfactory and config round-trip")
    config_object = braindecode_wrapper_config(
        braindecode_model_name = "EEGNet",
        model_kwargs           = {"n_chans" : N_CHANS, "n_outputs" : N_OUTPUT, "n_times" : N_TIMES},
        pooling                = {"pooling_type" : "sinc", "cutoff_scale" : 0.8},
    )
    assert braindecode_wrapper_config.from_dict(config_object.to_dict()) == config_object, "dict round-trip failed"
    with warnings.catch_warnings() :
        warnings.simplefilter("ignore")
        model = get_model(config_object)
    with torch.no_grad() :
        assert tuple(model(x).shape) == (BATCH, N_OUTPUT), "factory-built model output shape wrong"
    print("  [OK] get_model(braindecode_wrapper_config) and dict round-trip")

    print(f"\n{'=' * 70}\nAll braindecode pooling-extension checks passed.")


if __name__ == "__main__" :
    main()
