"""
End-to-end check of the EEG Conformer model (:mod:`jade.model.EEGConformer`).

Using dummy tensors (no dataset needed) it:
- builds the classifier and the plain feature extractor through the standard :func:`~jade.model.get_model` factory,
- runs a forward pass and checks the output shapes (logits ``(B, n_classes)`` for the classifier, a token sequence
  ``(B, n_tokens, emb_size)`` for the feature extractor),
- checks that the default configuration reproduces the original architecture (61 tokens -> 2440 flattened features for
  a 22 x 1000 BCI IV-2a trial),
- swaps the patch-embedding pooling for every custom :mod:`jade.model.pooling_1d` operator (avg / blur / sinc / dwt /
  max) and confirms the model still builds and runs,
- checks the ``classify`` helper, the ``use_softmax`` head, a training step (gradient flow), a different input geometry
  and the config dict/TOML round-trip.

Some pooling operators need optional dependencies (``scipy`` for ``sinc``, ``PyWavelets`` for some ``dwt`` wavelets);
those arms are skipped with a message if the dependency is missing.

Run with:
    python test/custom/EEGConformer.py

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

# Full module imports
import tempfile
import os
import warnings

# External libraries
import torch

# Internal libraries
from jade.model import get_model, EEGConformer, EEGConformer_Classifier
from jade.model.support_functions import count_trainable_parameters
from jade.config.model import EEGConformer_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

# EEG geometry of the dummy input (batch, depth, channels, time samples).
BATCH    = 6
DEPTH    = 1
N_CHANS  = 22
N_TIMES  = 1000
N_OUTPUT = 4

# The single-sample input size the classifier uses to size its MLP head.
INPUT_SIZE = (1, DEPTH, N_CHANS, N_TIMES)

# Pooling operators tried on the patch embedding: name -> (decimation stride, type-specific params).
# 'dwt' needs a power-of-two decimation factor, so it uses stride 16 instead of the default 15.
POOLING_CASES = {
    "default" : (15, {}),
    "avg"     : (15, {}),
    "max"     : (15, {}),
    "blur"    : (15, {}),
    "sinc"    : (15, {"cutoff_scale" : 0.8}),
    "dwt"     : (16, {"wavelet" : "sym4"}),
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Checks

def check_default_architecture(x : torch.Tensor) -> None :
    """
    Build the default classifier and feature extractor and check the shapes and the faithful token count.
    """

    print(f"\n{'=' * 70}\ndefault architecture (22 x 1000 BCI IV-2a trial)")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Classifier

    model = get_model(EEGConformer_config(input_size = INPUT_SIZE))
    model.eval()
    with torch.no_grad() :
        y = model(x)

    assert isinstance(model, EEGConformer_Classifier), "default name must build the classifier"
    assert tuple(y.shape) == (BATCH, N_OUTPUT), f"classifier output {tuple(y.shape)} != {(BATCH, N_OUTPUT)}"
    assert tuple(model.classify(x).shape) == (BATCH,), "classify() must return one index per sample"
    print(f"  [OK] classifier          : out {tuple(y.shape)}, params {count_trainable_parameters(model)}")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Feature extractor

    extractor = get_model(EEGConformer_config(name = "eegconformer", input_size = INPUT_SIZE))
    extractor.eval()
    with torch.no_grad() :
        tokens = extractor(x)

    assert isinstance(extractor, EEGConformer), "the 'eegconformer' name must build the feature extractor"
    # The original reference architecture yields 61 tokens of size 40 for a 22 x 1000 trial (61 * 40 = 2440).
    assert tuple(tokens.shape) == (BATCH, 61, 40), f"token sequence {tuple(tokens.shape)} != {(BATCH, 61, 40)}"
    print(f"  [OK] feature extractor    : tokens {tuple(tokens.shape)} (61 * 40 = 2440, matches the original)")


def check_pooling_operators(x : torch.Tensor) -> None :
    """
    Swap the patch-embedding pooling for every custom pooling operator and confirm the model runs.
    """

    print(f"\n{'=' * 70}\npatch-embedding pooling operators (jade.model.pooling_1d)")

    for pooling_type, (stride, params) in POOLING_CASES.items() :
        config = EEGConformer_config(
            input_size = INPUT_SIZE,
            pooling    = {"pooling_type" : pooling_type, "pooling_stride" : stride, "params" : params},
        )

        # Build the model (the anti-aliasing warnings of the pooling family are not interesting here).
        try :
            with warnings.catch_warnings() :
                warnings.simplefilter("ignore")
                model = get_model(config)
        except ImportError as error :
            # sinc needs scipy, some dwt wavelets need PyWavelets: skip the arm if the optional dep is missing.
            print(f"  [SKIP] {pooling_type:8s} : optional dependency missing ({error})")
            continue

        model.eval()
        with torch.no_grad() :
            y = model(x)

        assert tuple(y.shape) == (BATCH, N_OUTPUT), f"{pooling_type}: output {tuple(y.shape)} != {(BATCH, N_OUTPUT)}"
        assert tuple(model.classify(x).shape) == (BATCH,), f"{pooling_type}: bad classify() shape"
        print(f"  [OK] {pooling_type:8s} : stride {stride:2d}, out {tuple(y.shape)}, params {count_trainable_parameters(model)}")


def check_training_step(x : torch.Tensor) -> None :
    """
    Run one forward + backward pass and check that every gradient is finite (the model trains).
    """

    print(f"\n{'=' * 70}\ntraining step (gradient flow) and use_softmax head")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Logits head + cross-entropy

    model = get_model(EEGConformer_config(input_size = INPUT_SIZE))
    model.train()
    target = torch.randint(0, N_OUTPUT, (BATCH,))
    loss = torch.nn.functional.cross_entropy(model(x), target)
    loss.backward()

    grads_ok = all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters() if parameter.requires_grad
    )
    assert grads_ok, "some gradients are missing or non-finite"
    print(f"  [OK] logits head          : cross-entropy loss {loss.item():.4f}, all gradients finite")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # LogSoftmax head

    model = get_model(EEGConformer_config(use_softmax = True, input_size = INPUT_SIZE))
    model.eval()
    with torch.no_grad() :
        probabilities = model(x).exp().sum(dim = 1)
    assert torch.allclose(probabilities, torch.ones(BATCH), atol = 1e-5), "LogSoftmax head must produce log-probabilities"
    print("  [OK] use_softmax head     : exp(output) sums to 1 (log-probabilities)")


def check_geometry_and_round_trip(x : torch.Tensor) -> None :
    """
    Check a different input geometry and the config dict/TOML round-trip.
    """

    print(f"\n{'=' * 70}\ndifferent geometry and config round-trip")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Different geometry (channels, time, classes)

    other_x = torch.randn(4, 1, 32, 500)
    model = get_model(EEGConformer_config(C = 32, T = 500, n_classes = 3, input_size = (1, 1, 32, 500)))
    model.eval()
    with torch.no_grad() :
        assert tuple(model(other_x).shape) == (4, 3), "different-geometry output shape wrong"
    print("  [OK] geometry             : C=32, T=500, n_classes=3 -> (4, 3)")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Config round-trip (dict and TOML)

    config = EEGConformer_config(
        input_size = INPUT_SIZE,
        pooling    = {"pooling_type" : "blur", "pooling_stride" : 15},
    )
    assert EEGConformer_config.from_dict(config.to_dict()) == config, "dict round-trip failed"

    path = os.path.join(tempfile.gettempdir(), "eegconformer_config.toml")
    config.to_toml(path)
    assert EEGConformer_config.from_toml(path) == config, "TOML round-trip failed"
    print("  [OK] config round-trip    : dict and TOML")

    # The factory must also accept an already-built config object.
    with torch.no_grad() :
        assert tuple(get_model(config)(x).shape) == (BATCH, N_OUTPUT), "factory-built model output shape wrong"
    print("  [OK] get_model(config)    : builds from a config object")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Main

def main() -> None :
    """
    Run every check and print a final confirmation.
    """

    torch.manual_seed(0)
    x = torch.randn(BATCH, DEPTH, N_CHANS, N_TIMES)

    check_default_architecture(x)
    check_pooling_operators(x)
    check_training_step(x)
    check_geometry_and_round_trip(x)

    print(f"\n{'=' * 70}\nAll EEG Conformer checks passed.")


if __name__ == "__main__" :
    main()
