"""
End-to-end check of the vEEGNet model (:mod:`jade.model.vEEGNet`).

Using dummy tensors (no dataset needed) it:
- builds the model through the standard :func:`~jade.model.get_model` factory and checks the output of its forward,
- checks that the reconstruction has EXACTLY the shape of the input, for a set of awkward geometries (a number of time
  samples that no pooling factor divides, a disabled pooling stage, a spatial kernel shorter than the number of
  channels). This is the main regression check of the model: inverting the pooling with the pooling factor, instead of
  with the shape measured at build time, would return 992 samples for a 1000-sample trial,
- checks both latent mappings ('linear', the one of the original paper, and 'convolutional'),
- checks the reparametrization trick, the ``reconstruct`` and ``generate`` helpers, and a training step (gradient flow
  through both the encoder and the decoder),
- swaps the two encoder pooling stages for every custom :mod:`jade.model.pooling_1d` operator (avg / blur / sinc / dwt /
  max) and confirms the model still builds and reconstructs,
- checks the config dict/TOML round-trip, the registry aliases and the cross-check against a classification loss.

Some pooling operators need optional dependencies (``scipy`` for ``sinc``, ``PyWavelets`` for some ``dwt`` wavelets);
those arms are skipped with a message if the dependency is missing.

Run with:
    python test/custom/vEEGNet.py

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

# Full module imports
import itertools
import tempfile
import os
import warnings

# External libraries
import torch

# Internal libraries
from jade.config import cross_entropy_config, elbo_config, training_config
from jade.config.model import EEGNet_config, vEEGNet_config, get_model_config
from jade.model import get_model, vEEGNet
from jade.model.support_functions import count_trainable_parameters
from jade.training import support_training
from jade.training.custom_loss_function import get_custom_loss_function

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

# EEG geometry of the dummy input (batch, depth, channels, time samples).
BATCH   = 4
DEPTH   = 1
N_CHANS = 22
N_TIMES = 1000

# Dimension of the latent space with the 'linear' mapping (the value used in the original paper).
HIDDEN_SPACE_DIMENSION = 64

# Geometries used by the output-length check.
# Note that 999 and 257 are not divisible by the pooling factors, which is exactly the case the naive inversion breaks
# on, and that a negative factor disables the corresponding pooling stage.
TIME_SAMPLES_CASES   = (1000, 999, 512, 257)
POOLING_FACTOR_CASES = ((4, 8), (5, 7), (4, -1), (-1, 8))
CHANNELS_CASES       = (22, 32)

# Pooling operators tried on the two encoder stages: name -> type-specific params.
# 'dwt' needs a power-of-two decimation factor, which the default factors (4 and 8) already are.
POOLING_CASES = {
    "default" : {},
    "avg"     : {},
    "max"     : {},
    "blur"    : {},
    "sinc"    : {},
    "dwt"     : {"wavelet" : "db4"},
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Helper functions

def build_config(**kwargs) -> vEEGNet_config :
    """
    Build a vEEGNet config for the default dummy geometry, overridden by ``kwargs``.

    ``input_size`` must agree with ``depth_first_layer``/``C``/``T``, so it is derived here instead of being passed by every check.
    """

    n_chans = kwargs.pop("C", N_CHANS)
    n_times = kwargs.pop("T", N_TIMES)

    return vEEGNet_config(
        C          = n_chans,
        T          = n_times,
        input_size = (1, DEPTH, n_chans, n_times),
        **kwargs
    )


def dummy_input(n_chans : int = N_CHANS, n_times : int = N_TIMES) -> torch.Tensor :
    """
    Create a dummy batch of EEG trials.
    """

    return torch.randn(BATCH, DEPTH, n_chans, n_times)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Checks

def check_default_architecture() -> None :
    """
    Build the model with its defaults and check the output of the forward.
    """

    print("Default architecture")

    model = get_model(build_config())
    assert isinstance(model, vEEGNet), f"get_model returned a {type(model)}"

    x = dummy_input()
    output = model(x)

    assert isinstance(output, tuple) and len(output) == 4, f"forward must return a 4-tuple, got {type(output)} of length {len(output)}"
    x_reconstructed, mu, log_var, z = output

    assert x_reconstructed.shape == x.shape, f"reconstruction {tuple(x_reconstructed.shape)} != input {tuple(x.shape)}"
    assert mu.shape == log_var.shape == z.shape == (BATCH, HIDDEN_SPACE_DIMENSION), f"latent tensors have shape {tuple(mu.shape)}"

    print(f"  [OK] forward             : 4-tuple, reconstruction {tuple(x_reconstructed.shape)}, latent {tuple(mu.shape)}")
    print(f"  [OK] parameters          : encoder {count_trainable_parameters(model.encoder)}, "
          f"sampling {count_trainable_parameters(model.sampling)}, decoder {count_trainable_parameters(model.decoder)}")


def check_output_length() -> None :
    """
    Check that the reconstruction has exactly the shape of the input, for every awkward geometry.

    This is the regression check of the shape handling of the decoder: with ``T = 1000`` and the default pooling factors
    the encoder produces 31 time samples, so upsampling by the factors would give back 992 samples instead of 1000.
    """

    print("Output length (the reconstruction must have exactly the shape of the input)")

    n_case = 0
    for n_times, (pooling_factor_1, pooling_factor_2), n_chans in itertools.product(TIME_SAMPLES_CASES, POOLING_FACTOR_CASES, CHANNELS_CASES) :
        # Both a spatial kernel that collapses the channel axis and one that does not.
        for kernel_height in (n_chans, 11) :
            config = build_config(
                C          = n_chans,
                T          = n_times,
                c_kernel_2 = (kernel_height, 1),
                pooling    = dict(pooling_factor_1 = pooling_factor_1, pooling_factor_2 = pooling_factor_2),
            )

            # A negative pooling factor is legal (it disables the stage) but the config warns about it.
            with warnings.catch_warnings() :
                warnings.simplefilter("ignore")
                model = get_model(config)

            x = dummy_input(n_chans, n_times)
            x_reconstructed = model(x)[0]

            assert x_reconstructed.shape == x.shape, (
                f"T = {n_times}, factors = ({pooling_factor_1}, {pooling_factor_2}), C = {n_chans}, kernel = {kernel_height}: "
                f"reconstruction {tuple(x_reconstructed.shape)} != input {tuple(x.shape)}"
            )

            # The first pooling stage of EEGNet receives the time axis untouched, so its recorded target is T itself.
            assert model.upsample_target_shapes[0][-1] == n_times, f"the first upsample target is {model.upsample_target_shapes[0]} instead of T = {n_times}"

            n_case += 1

    print(f"  [OK] {n_case} geometries      : T in {TIME_SAMPLES_CASES}, factors in {POOLING_FACTOR_CASES}, C in {CHANNELS_CASES}")


def check_latent_mappings() -> None :
    """
    Check the two ways of mapping the encoder output to the parameters of the latent distribution.
    """

    print("Latent mappings")

    # 'linear': the latent variable is a vector of hidden_space_dimension elements.
    model = get_model(build_config(parameters_map_type = "linear", hidden_space_dimension = 32))
    x = dummy_input()
    x_reconstructed, mu, log_var, z = model(x)

    assert mu.shape == (BATCH, 32), f"linear mapping gives mu of shape {tuple(mu.shape)}"
    assert model.latent_shape == (32,), f"latent_shape is {model.latent_shape}"
    assert x_reconstructed.shape == x.shape
    print(f"  [OK] linear              : latent {tuple(mu.shape)}, reconstruction {tuple(x_reconstructed.shape)}")

    # 'convolutional': the latent variable keeps the shape of the encoder output.
    model = get_model(build_config(parameters_map_type = "convolutional"))
    x_reconstructed, mu, log_var, z = model(x)

    assert tuple(mu.shape[1:]) == model.encoder_output_shape, f"convolutional mapping gives mu of shape {tuple(mu.shape)} instead of {model.encoder_output_shape}"
    assert model.latent_shape == model.encoder_output_shape, f"latent_shape is {model.latent_shape}"
    assert x_reconstructed.shape == x.shape
    print(f"  [OK] convolutional       : latent {tuple(mu.shape)}, reconstruction {tuple(x_reconstructed.shape)}")


def check_reparametrization() -> None :
    """
    Check that the latent variable is sampled, while its parameters are deterministic.
    """

    print("Reparametrization trick")

    model = get_model(build_config())
    model.eval()
    x = dummy_input()

    mu_1, log_var_1 = model.encode(x)
    mu_2, log_var_2 = model.encode(x)

    assert torch.allclose(mu_1, mu_2), "the mean of the posterior is not deterministic in eval mode"
    assert torch.allclose(log_var_1, log_var_2), "the log variance of the posterior is not deterministic in eval mode"

    z_1 = model(x)[3]
    z_2 = model(x)[3]
    assert not torch.allclose(z_1, z_2), "two forward passes returned the same latent sample: the sampling is not random"

    sigma = torch.exp(0.5 * log_var_1)
    assert torch.isfinite(sigma).all() and (sigma > 0).all(), "the standard deviation of the posterior is not finite and positive"

    print("  [OK] mu / log_var        : deterministic in eval mode")
    print("  [OK] z                   : different at every forward, sigma finite and positive")


def check_reconstruct_and_generate() -> None :
    """
    Check the two public helpers of the generative side.
    """

    print("reconstruct and generate")

    for parameters_map_type in ("linear", "convolutional") :
        model = get_model(build_config(parameters_map_type = parameters_map_type))
        x = dummy_input()

        # reconstruct
        x_reconstructed = model.reconstruct(x)
        assert x_reconstructed.shape == x.shape, f"reconstruct returned {tuple(x_reconstructed.shape)}"
        assert not x_reconstructed.requires_grad, "reconstruct(no_grad = True) returned a tensor attached to the graph"

        x_reconstructed = model.reconstruct(x, no_grad = False)
        assert x_reconstructed.requires_grad, "reconstruct(no_grad = False) returned a tensor detached from the graph"

        # generate, from the prior
        x_generated = model.generate(n_samples = 5)
        assert x_generated.shape == (5, DEPTH, N_CHANS, N_TIMES), f"generate returned {tuple(x_generated.shape)}"
        assert x_generated.device == next(model.parameters()).device, "generate returned a tensor on the wrong device"

        # generate, from an explicit latent variable
        x_generated = model.generate(z = torch.randn(3, *model.latent_shape))
        assert x_generated.shape == (3, DEPTH, N_CHANS, N_TIMES), f"generate(z = ...) returned {tuple(x_generated.shape)}"

        print(f"  [OK] {parameters_map_type:14s}     : reconstruct {tuple(x.shape)}, generate 5 and 3 samples")


def check_training_step() -> None :
    """
    Run one optimization step with the ELBO and check that the gradient reaches every parameter.

    This is what catches a branch of the decoder that is not connected to the graph.
    """

    print("Training step (ELBO)")

    for parameters_map_type in ("linear", "convolutional") :
        model = get_model(build_config(parameters_map_type = parameters_map_type))
        loss_function = get_custom_loss_function(elbo_config())

        x = dummy_input()
        terms = loss_function(model(x), x)

        assert terms.total.dim() == 0, f"the total loss is not a scalar, it has {terms.total.dim()} dimensions"
        assert torch.allclose(terms.total, terms.reconstruction + terms.kl), "the total loss does not match its terms with the default weights"

        terms.total.backward()

        parameters_without_gradient = [name for name, parameter in model.named_parameters()
                                       if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all())]
        assert not parameters_without_gradient, f"these parameters have no (finite) gradient: {parameters_without_gradient}"

        print(f"  [OK] {parameters_map_type:14s}     : loss {float(terms.total):.4f} "
              f"(reconstruction {float(terms.reconstruction):.4f}, KL {float(terms.kl):.4f}), gradient on every parameter")


def check_kl_warmup() -> None :
    """
    Check that the warm-up moves the weight of the KL term from its starting value to 1.
    """

    print("KL warm-up")

    loss_function = get_custom_loss_function(elbo_config(beta = 2.0, kl_warmup_type = "linear", kl_warmup_epochs = 4, kl_warmup_start = 0.0))

    weights = []
    for epoch in range(6) :
        loss_function.set_epoch(epoch)
        weights.append(round(loss_function.kl_weight, 3))

    assert weights == [0.0, 0.25, 0.5, 0.75, 1.0, 1.0], f"the warm-up gives the weights {weights}"

    # With a partial weight the two totals must differ; the objective at full KL weight is the larger one.
    model = get_model(build_config())
    x = dummy_input()
    loss_function.set_epoch(1)
    terms = loss_function(model(x), x)

    assert terms.kl_weight == 0.25, f"kl_weight is {terms.kl_weight}"
    assert torch.allclose(terms.total, terms.reconstruction + 2.0 * 0.25 * terms.kl), "the total loss does not use the warm-up weight"
    assert torch.allclose(terms.total_full_kl, terms.reconstruction + 2.0 * terms.kl), "total_full_kl does not use the final KL weight"

    print(f"  [OK] linear schedule     : {weights}")
    print("  [OK] total / total_full_kl: both match their definition")


def check_pooling_operators() -> None :
    """
    Build the model with every 1D pooling operator on the two encoder stages.
    """

    print("Pooling operators")

    for pooling_type, params in POOLING_CASES.items() :
        config = build_config(pooling = dict(
            pooling_type_1 = pooling_type, pooling_type_2 = pooling_type,
            params_1 = dict(params), params_2 = dict(params),
        ))

        try :
            with warnings.catch_warnings() :
                warnings.simplefilter("ignore")
                model = get_model(config)
                x = dummy_input()
                x_reconstructed = model(x)[0]
        except ImportError as error :
            print(f"  [SKIP] {pooling_type:16s}: missing optional dependency ({error})")
            continue

        assert x_reconstructed.shape == x.shape, f"{pooling_type}: reconstruction {tuple(x_reconstructed.shape)} != input {tuple(x.shape)}"
        print(f"  [OK] {pooling_type:16s}  : reconstruction {tuple(x_reconstructed.shape)}")

    # One stage disabled.
    with warnings.catch_warnings() :
        warnings.simplefilter("ignore")
        model = get_model(build_config(pooling = dict(pooling_factor_1 = -1)))
    x = dummy_input()
    assert model(x)[0].shape == x.shape, "a disabled pooling stage breaks the reconstruction shape"
    print("  [OK] disabled stage       : reconstruction still matches the input")


def check_config_round_trip() -> None :
    """
    Check the serialization of the config, the registry aliases and the encoder config derived from it.
    """

    print("Config")

    config = build_config(parameters_map_type = "convolutional", hidden_space_dimension = 16)

    # dict round-trip
    assert vEEGNet_config.from_dict(config.to_dict()) == config, "the dict round-trip changed the config"

    # TOML round-trip (it also covers the nested [pooling] sub-table)
    path_toml = os.path.join(tempfile.gettempdir(), "jade_test_vEEGNet.toml")
    config.to_toml(path_toml)
    assert vEEGNet_config.from_toml(path_toml) == config, "the TOML round-trip changed the config"
    os.remove(path_toml)
    print("  [OK] round-trip          : dict and TOML")

    # The factory accepts a dict, and the name is case-insensitive (the class is spelled vEEGNet).
    assert isinstance(get_model_config({"name" : "vEEGNet"}), vEEGNet_config), "the mixed-case name is not resolved"
    assert isinstance(get_model_config({"name" : "vae_eegnet"}), vEEGNet_config), "the alias vae_eegnet is not resolved"
    assert isinstance(get_model({"name" : "veegnet"}), vEEGNet), "get_model does not accept a plain dict"

    # The classifier aliases are deliberately NOT registered (there is no classification branch yet).
    try :
        get_model_config({"name" : "veegnet_classifier"})
        raise AssertionError("veegnet_classifier should not be a known model")
    except ValueError :
        pass
    print("  [OK] registry            : aliases resolved, veegnet_classifier still unknown")

    # The encoder config must carry the same values, with the two forced fields.
    encoder_config = config.to_encoder_config()
    assert isinstance(encoder_config, EEGNet_config), f"to_encoder_config returned a {type(encoder_config)}"
    assert encoder_config.name == "eegnet", f"the encoder config has name {encoder_config.name!r}"
    assert encoder_config.flatten_output is False, "the encoder config flattens its output"
    for field_name in ("use_bias", "D", "activation", "prob_dropout", "use_dropout_2d", "C", "T",
                       "depth_first_layer", "filter_1", "c_kernel_1", "c_kernel_2", "filter_2", "c_kernel_3") :
        assert getattr(encoder_config, field_name) == getattr(config, field_name), f"the encoder config disagrees on {field_name}"
    print("  [OK] to_encoder_config   : every shared field matches, name and flatten_output forced")


def check_input_guard_and_cross_check() -> None :
    """
    Check the input-geometry guard and the new cross-config rules.
    """

    print("Guards")

    model = get_model(build_config())

    # A trial with a different number of time samples cannot be reconstructed by a decoder built for another geometry.
    try :
        model(torch.randn(BATCH, DEPTH, N_CHANS, N_TIMES // 2))
        raise AssertionError("an input with the wrong geometry should be rejected")
    except ValueError :
        pass
    print("  [OK] input geometry      : an input with the wrong shape is rejected")

    # A generative model cannot be trained with a classification loss, and the other way around.
    try :
        support_training.cross_check_config(training_config(loss = cross_entropy_config()), build_config())
        raise AssertionError("a classification loss paired with vEEGNet should be rejected")
    except ValueError :
        pass

    try :
        support_training.cross_check_config(training_config(loss = elbo_config()), EEGNet_config())
        raise AssertionError("the ELBO paired with a classifier should be rejected")
    except ValueError :
        pass

    # The valid pairings must still pass.
    support_training.cross_check_config(training_config(loss = elbo_config()), build_config())
    support_training.cross_check_config(training_config(loss = cross_entropy_config()), EEGNet_config())
    print("  [OK] cross_check_config  : the two mismatched pairings are rejected, the valid ones pass")


def check_print_var() -> None :
    """
    Check that the verbose build (parameter counts plus shape tracking) runs.
    """

    print("Verbose build")

    get_model(build_config(print_var = True))
    print("  [OK] print_var           : the shape tracking runs")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Main

def main() -> None :
    """
    Run every check.
    """

    torch.manual_seed(0)

    print("=" * 70)
    print("vEEGNet checks")
    print("=" * 70)

    check_default_architecture()
    check_output_length()
    check_latent_mappings()
    check_reparametrization()
    check_reconstruct_and_generate()
    check_training_step()
    check_kl_warmup()
    check_pooling_operators()
    check_config_round_trip()
    check_input_guard_and_cross_check()
    check_print_var()

    print()
    print("=" * 70)
    print("All vEEGNet checks passed.")
    print("=" * 70)


if __name__ == "__main__" :
    main()
