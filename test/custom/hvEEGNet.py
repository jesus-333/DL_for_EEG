"""
End-to-end check of the hvEEGNet model (:mod:`jade.model.hvEEGNet`).

Using dummy tensors (no dataset needed) it:
- builds the model through the standard :func:`~jade.model.get_model` factory and checks the structure of its output (reconstruction, three latent spaces, two posterior shifts),
- checks that the reconstruction has EXACTLY the shape of the input for a set of awkward geometries (a number of time samples that no pooling factor divides, both pooling stages active, a spatial kernel shorter than the number of channels),
- checks the hierarchical KL term against the general closed form of the divergence between two Gaussians,
- checks the ``reconstruct``, ``reconstruct_ignoring_latent_spaces`` and ``generate`` helpers,
- runs a training step with the ELBO (gradient flow through the encoder, the decoder and every latent map),
- swaps the encoder pooling stages for the :mod:`jade.model.pooling_1d` operators,
- checks the three latent mappings ('convolutional', 'linear', 'lora') and the extra decoder input ``h`` (trainable, zeros, user-given),
- checks the DTW reconstruction terms of the ELBO ('soft_dtw', 'block_dtw'), skipped if ``dtw-loss-functions`` is not installed,
- checks the config TOML round-trip, the registry aliases and the cross-check against a classification loss,
- trains the model for a few epochs with :func:`jade.training.vae.train_model` on synthetic data.

Run with:
    python test/custom/hvEEGNet.py

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

# Full module imports
import os
import tempfile

# External libraries
import numpy as np
import torch

# Internal libraries
from jade.config import cross_entropy_config, elbo_config, training_config
from jade.config.model import hvEEGNet_config, get_model_config
from jade.dataset import EEG_Dataset
from jade.model import get_model, hvEEGNet
from jade.model.support_functions import count_trainable_parameters
from jade.training import support_training, vae
from jade.training.custom_loss_function import get_custom_loss_function, support_loss

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

# Small EEG geometry, to keep the script fast (the default config is the one of the paper: 22 x 1000).
BATCH   = 4
N_CHANS = 8
N_TIMES = 400

# Geometries used by the output-length check: (C, T, pooling factor 1, pooling factor 2, spatial kernel height).
GEOMETRY_CASES = (
    (22, 1000, 1, 10, 22),   # The paper
    (8, 399, 1, 10, 8),      # T not divisible by the pooling factor
    (8, 257, 4, 8, 8),       # Both stages active, T divisible by neither
    (8, 400, 1, 10, 3),      # Spatial kernel shorter than the number of channels
)

# Latent dimensions used with the vector mappings (small, to keep the feedforward layers light).
HIDDEN_SPACE_DIMENSION_LIST = (8, 16, 32)

# Pooling operators tried on the second encoder stage.
POOLING_CASES = ("avg", "max", "blur", "sinc")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Helper functions

def build_config(C : int = N_CHANS, T : int = N_TIMES, **kwargs) -> hvEEGNet_config :
    """
    Build a small hvEEGNet config with a consistent geometry.
    """

    return hvEEGNet_config(C = C, T = T, c_kernel_2 = kwargs.pop("c_kernel_2", (C, 1)), input_size = (1, 1, C, T), **kwargs)


def general_gaussian_kl(mu_q, log_var_q, mu_p, log_var_p) -> torch.Tensor :
    """
    KL(q || p) between two diagonal Gaussians, summed over the non-batch axes and averaged over the batch.
    """

    kl = 0.5 * (log_var_p - log_var_q + (log_var_q.exp() + (mu_q - mu_p) ** 2) / log_var_p.exp() - 1)

    return kl.flatten(1).sum(1).mean()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Checks

def check_output_structure() -> None :
    model = get_model(build_config())
    x = torch.randn(BATCH, 1, N_CHANS, N_TIMES)

    x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list, z_list = model(x)

    assert x_r.shape == x.shape
    assert len(mu_list) == len(log_var_list) == len(z_list) == 3
    assert len(delta_mu_list) == len(delta_log_var_list) == 2
    for level in range(3) :
        assert mu_list[level].shape == log_var_list[level].shape == z_list[level].shape == (BATCH, *model.latent_shapes[level])
    for level in range(2) :
        assert delta_mu_list[level].shape == mu_list[level + 1].shape

    print(f"  [OK] output structure    : latent shapes {model.latent_shapes}, {count_trainable_parameters(model)} trainable parameters")


def check_geometries() -> None :
    for C, T, factor_1, factor_2, kernel_height in GEOMETRY_CASES :
        pooling = {"pooling_type_1" : "default" if factor_1 > 1 else "identity", "pooling_factor_1" : factor_1, "pooling_type_2" : "default", "pooling_factor_2" : factor_2}
        model = get_model(build_config(C, T, c_kernel_2 = (kernel_height, 1), pooling = pooling))
        x = torch.randn(2, 1, C, T)
        assert model(x)[0].shape == x.shape, f"wrong output shape for C = {C}, T = {T}, factors = ({factor_1}, {factor_2})"
        assert model.generate(2).shape == x.shape

    print(f"  [OK] geometries          : output shape == input shape for {len(GEOMETRY_CASES)} geometries")


def check_hierarchical_kl() -> None :
    torch.manual_seed(0)
    shape = (BATCH, 4, 1, 10)

    mu_list            = [torch.randn(shape) for _ in range(3)]
    log_var_list       = [torch.randn(shape) for _ in range(3)]
    delta_mu_list      = [torch.randn(shape) for _ in range(2)]
    delta_log_var_list = [torch.randn(shape) for _ in range(2)]

    kl = support_loss.kl_divergence_hierarchical(mu_list, log_var_list, delta_mu_list, delta_log_var_list)

    # Same quantity, from the general formula: level 0 against N(0, I), the others against their prior.
    expected_kl = general_gaussian_kl(mu_list[0], log_var_list[0], torch.zeros(shape), torch.zeros(shape))
    for level in range(1, 3) :
        mu_q      = mu_list[level] + delta_mu_list[level - 1]
        log_var_q = log_var_list[level] + delta_log_var_list[level - 1]
        expected_kl = expected_kl + general_gaussian_kl(mu_q, log_var_q, mu_list[level], log_var_list[level])

    assert torch.allclose(kl, expected_kl, rtol = 1e-5), f"hierarchical KL {kl.item()} != {expected_kl.item()}"

    # Zero shifts and a standard normal deepest posterior mean a null divergence.
    zeros = [torch.zeros(shape) for _ in range(3)]
    assert support_loss.kl_divergence_hierarchical(zeros, zeros, zeros[:2], zeros[:2]).abs().item() < 1e-6

    print("  [OK] hierarchical KL     : matches the general Gaussian KL, zero for an unshifted posterior")


def check_helpers() -> None :
    model = get_model(build_config())
    model.eval()
    x = torch.randn(BATCH, 1, N_CHANS, N_TIMES)

    assert model.reconstruct(x).shape == x.shape
    assert not model.reconstruct(x).requires_grad

    # With use_mean the latent variables are not sampled, so (in eval mode) the reconstruction is deterministic and the z are the posterior means
    assert torch.equal(model.reconstruct(x, use_mean = True), model.reconstruct(x, use_mean = True))
    assert not torch.equal(model.reconstruct(x), model.reconstruct(x))
    with torch.no_grad() :
        output = model(x, use_mean = True)
    assert torch.equal(output[5][0], output[1][0])                    # z_0 = mu_0
    assert torch.equal(output[5][1], output[1][1] + output[3][0])     # z_1 = prior mean + posterior shift

    for flags in ([True, False, False], [False, True, True], [True, True, True]) :
        assert model.reconstruct_ignoring_latent_spaces(x, flags).shape == x.shape

    try :
        model.reconstruct_ignoring_latent_spaces(x, [True, False])
        raise AssertionError("a list of two flags must be rejected")
    except ValueError :
        pass

    mu, log_var = model.encode(x)
    assert mu.shape == (BATCH, *model.latent_shapes[0])

    assert model.generate(3).shape == (3, 1, N_CHANS, N_TIMES)
    assert model.generate(z = torch.randn(2, *model.latent_shapes[0])).shape == (2, 1, N_CHANS, N_TIMES)

    try :
        model(torch.randn(BATCH, 1, N_CHANS, N_TIMES + 1))
        raise AssertionError("an input with the wrong geometry must be rejected")
    except ValueError :
        pass

    print("  [OK] helpers             : reconstruct (sampled and use_mean), reconstruct_ignoring_latent_spaces, encode, generate, input guard")


def check_training_step() -> None :
    model = get_model(build_config())
    loss_function = get_custom_loss_function(elbo_config())
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    x = torch.randn(BATCH, 1, N_CHANS, N_TIMES)
    loss_terms = loss_function(model(x), x)
    assert torch.isfinite(loss_terms.total)
    assert loss_terms.kl.item() >= 0

    optimizer.zero_grad()
    loss_terms.total.backward()

    # Every parameter (encoder, decoder, prior maps, posterior maps, combinations) must receive a gradient.
    missing = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None]
    assert not missing, f"parameters without gradient: {missing}"
    optimizer.step()

    print(f"  [OK] training step       : total = {loss_terms.total.item():.3f}, kl = {loss_terms.kl.item():.3f}, every parameter has a gradient")


def check_pooling_operators() -> None :
    x = torch.randn(2, 1, N_CHANS, N_TIMES)
    for pooling_type in POOLING_CASES :
        try :
            model = get_model(build_config(pooling = {"pooling_type_1" : "identity", "pooling_factor_1" : 1, "pooling_type_2" : pooling_type, "pooling_factor_2" : 8}))
        except ImportError as error :
            print(f"  [SKIP] pooling {pooling_type} : {error}")
            continue
        assert model(x)[0].shape == x.shape, f"wrong output shape with pooling {pooling_type}"

    print(f"  [OK] pooling operators   : {POOLING_CASES}")


def check_latent_mappings() -> None :
    x = torch.randn(BATCH, 1, N_CHANS, N_TIMES)
    loss_function = get_custom_loss_function(elbo_config())

    for parameters_map_type, use_spectral_norm in (("linear", False), ("lora", False), ("lora", True)) :
        model = get_model(build_config(parameters_map_type = parameters_map_type, hidden_space_dimension_list = HIDDEN_SPACE_DIMENSION_LIST, lora_rank = 4, use_spectral_norm = use_spectral_norm))
        assert model.latent_shapes == tuple((dimension,) for dimension in HIDDEN_SPACE_DIMENSION_LIST)

        output = model(x)
        assert output[0].shape == x.shape
        for level in range(3) : assert output[1][level].shape == output[5][level].shape == (BATCH, HIDDEN_SPACE_DIMENSION_LIST[level])

        # Gradient flow, with the hierarchical KL computed on vectors.
        loss_function(output, x).total.backward()
        missing = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None]
        assert not missing, f"{parameters_map_type}: parameters without gradient: {missing}"

        # Generation goes through the same (trained) projections.
        assert model.generate(2).shape == (2, 1, N_CHANS, N_TIMES)

    print(f"  [OK] latent mappings     : 'linear', 'lora', 'lora' + spectral norm with latent dimensions {HIDDEN_SPACE_DIMENSION_LIST}")


def check_h() -> None :
    x = torch.randn(BATCH, 1, N_CHANS, N_TIMES)

    # Default: no trainable h (zeros, as in the paper).
    model = get_model(build_config())
    assert not hasattr(model, "h")

    # Trainable h: it is a parameter and it receives a gradient.
    model = get_model(build_config(use_h_in_decoder = True))
    assert any(name == "h" for name, _ in model.named_parameters())
    get_custom_loss_function(elbo_config())(model(x), x).total.backward()
    assert model.h.grad is not None and model.h.grad.abs().sum() > 0, "the trainable h received no gradient"

    # A user-given h changes the generation (same z), and a batch axis of 1 is broadcasted.
    model.eval()
    z = torch.randn(3, *model.latent_shapes[0])
    h_shape = (1, *model.encoder_outputs_shapes[2])
    x_default = model.generate(z = z)
    x_seed    = model.generate(z = z, h = 5 * torch.ones(h_shape))
    assert x_seed.shape == x_default.shape and not torch.allclose(x_seed, x_default)

    try :
        model.generate(z = z, h = torch.ones(1, 1, 1, 1))
        raise AssertionError("an h with the wrong shape must be rejected")
    except ValueError :
        pass

    print("  [OK] h                   : zeros by default, trainable with use_h_in_decoder, user-given seed in generate")


def check_dtw_reconstruction() -> None :
    try :
        import dtw_loss_functions  # noqa: F401
    except ImportError :
        print("  [SKIP] DTW reconstruction : dtw-loss-functions is not installed")
        return

    model = get_model(build_config())
    x = torch.randn(BATCH, 1, N_CHANS, N_TIMES)

    for reconstruction_type, block_size in (("soft_dtw", 128), ("block_dtw", 100), ("block_dtw", 128)) :
        model.zero_grad()
        loss_function = get_custom_loss_function(elbo_config(reconstruction_type = reconstruction_type, dtw_block_size = block_size, dtw_average_time_samples = True))
        loss_terms = loss_function(model(x), x)
        assert torch.isfinite(loss_terms.total)
        loss_terms.total.backward()
        assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)

    print("  [OK] DTW reconstruction  : 'soft_dtw' and 'block_dtw' (optimized and naive block path) train hvEEGNet")


def check_config() -> None :
    config = hvEEGNet_config()
    assert config.c_kernel_1 == (1, 128) and config.c_kernel_3 == (1, 32) and config.pooling.pooling_factor_2 == 10

    path_toml = os.path.join(tempfile.mkdtemp(prefix = "jade_test_hvEEGNet_"), "hvEEGNet.toml")
    config.to_toml(path_toml)
    assert hvEEGNet_config.from_toml(path_toml) == config, "the TOML round-trip changed the config"

    for name in ("hveegnet", "hierarchical_veegnet", "hvEEGNet") :
        assert isinstance(get_model_config({"name" : name}), hvEEGNet_config)
    assert isinstance(get_model(build_config()), hvEEGNet)

    try :
        support_training.cross_check_config(training_config(loss = cross_entropy_config()), config)
        raise AssertionError("hvEEGNet paired with a classification loss must be rejected")
    except ValueError :
        pass
    support_training.cross_check_config(training_config(loss = elbo_config()), config)

    print("  [OK] config              : paper defaults, TOML round-trip, registry aliases, cross_check_config")


def check_training_cycle() -> None :
    def generate_dataset(n_samples : int) -> EEG_Dataset :
        data = np.random.randn(n_samples, 1, N_CHANS, N_TIMES).astype(np.float32)
        return EEG_Dataset(data = data, labels = np.zeros(n_samples).astype(np.int64), ch_list = [f"Ch{i + 1}" for i in range(N_CHANS)])

    epochs = 3
    training_config_object = training_config(
        epochs             = epochs,
        batch_size         = 8,
        device             = "cpu",
        path_to_save_model = os.path.join(tempfile.mkdtemp(prefix = "jade_test_hvEEGNet_"), "model_weights"),
        loss               = elbo_config(kl_warmup_type = "linear", kl_warmup_epochs = 2),
    )

    _, metrics = vae.train_model(training_config_object, get_model(build_config()), generate_dataset(16), validation_dataset = generate_dataset(16))
    assert all(np.isfinite(value) for value in metrics["train_loss"]), "non finite training loss"
    assert len(metrics["train_loss"]) == epochs

    print(f"  [OK] training cycle      : {epochs} epochs with jade.training.vae, train loss {[round(value, 3) for value in metrics['train_loss']]}")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Main

if __name__ == "__main__" :
    torch.manual_seed(0)
    np.random.seed(0)

    check_output_structure()
    check_geometries()
    check_hierarchical_kl()
    check_helpers()
    check_training_step()
    check_latent_mappings()
    check_h()
    check_dtw_reconstruction()
    check_pooling_operators()
    check_config()
    check_training_cycle()

    print("\nAll hvEEGNet checks passed")
