"""
End-to-end check of the generative training cycle (:mod:`jade.training.vae`).

- Create a vEEGNet
- Create synthetic data of shape [N x 1 x C x T] with N = n. of training samples, C = n. of channels, T = n. of time points
- Train it for a few epochs with the ELBO and a linear KL warm-up, and check that
  every term of the loss is tracked, that the checkpoints are written, that the loss
  decreases and that the warm-up ramps the weight of the KL term as expected.

Run with:
    python test/custom/vEEGNet_creation_and_training.py

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

# Full module imports
import os
import shutil
import tempfile

# External libraries
import numpy as np
import torch

# Internal libraries
from jade.config import elbo_config, training_config
from jade.config.model import vEEGNet_config
from jade.dataset import EEG_Dataset
from jade.model import get_model
from jade.training import logging_utils, vae

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

# Number of training samples, time points and channels
N = 12
T = 512
C = 8

# Number of epochs, and duration of the KL warm-up (shorter than the training, so the weight reaches 1).
EPOCHS           = 3
KL_WARMUP_EPOCHS = 2

# The terms of the loss that must be tracked for each set.
LOSS_TERMS = ("loss", "recon_loss", "kl_loss", "loss_full_kl")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Synthetic data and dataset creation

def generate_dataset(n_samples : int, n_chans : int, n_times : int) -> EEG_Dataset :
    """
    Create a dataset of synthetic EEG trials.

    The labels are not used by the generative cycle, but the dataset yields them anyway, exactly as it does for a classifier.
    """

    data = np.random.rand(n_samples, 1, n_chans, n_times).astype(np.float32)
    labels = np.zeros(n_samples).astype(np.int64)
    ch_list = [f"Ch{i + 1}" for i in range(n_chans)]

    return EEG_Dataset(data = data, labels = labels, ch_list = ch_list)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Main

def main() -> None :
    """
    Build the model, train it and check the outcome.
    """

    torch.manual_seed(0)
    np.random.seed(0)

    logging_utils.setup_logging()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Model and data

    model_config = vEEGNet_config(
        C          = C,
        T          = T,
        c_kernel_2 = (C, 1),
        input_size = (1, 1, C, T),
    )
    model = get_model(model_config)
    print(f"Model config:\n{model_config}")

    train_dataset      = generate_dataset(N, C, T)
    validation_dataset = generate_dataset(N, C, T)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Training config

    device = "cuda" if torch.cuda.is_available() else "cpu"

    path_to_save_model = os.path.join(tempfile.mkdtemp(prefix = "jade_test_vEEGNet_"), "model_weights")

    training_config_object = training_config(
        epochs             = EPOCHS,
        batch_size         = N // 2,
        device             = device,
        path_to_save_model = path_to_save_model,
        loss               = elbo_config(kl_warmup_type = "linear", kl_warmup_epochs = KL_WARMUP_EPOCHS),
    )
    print(f"\nTraining config:\n{training_config_object}")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Training

    output = vae.train_model(training_config_object, model, train_dataset, validation_dataset = validation_dataset)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Checks

    print()
    print("=" * 70)

    assert isinstance(output, tuple) and len(output) == 2, f"train_model must return a (model, metrics) pair, got {type(output)}"
    trained_model, computed_metrics_during_training = output
    print("  [OK] return value        : the (model, metrics) pair")

    # Every term of the loss, for both sets, plus the weight of the KL term.
    for set_name in ("train", "validation") :
        for term in LOSS_TERMS :
            key = f"{set_name}_{term}"
            assert key in computed_metrics_during_training, f"{key} is missing from the tracked metrics"
            assert len(computed_metrics_during_training[key]) == EPOCHS, f"{key} has {len(computed_metrics_during_training[key])} values instead of {EPOCHS}"
    print(f"  [OK] loss terms          : {LOSS_TERMS} tracked for both sets, {EPOCHS} values each")

    # The reconstruction metrics (measure_metrics_during_training is true by default).
    for key in ("reconstruction_mse_train", "reconstruction_mae_train", "reconstruction_mse_validation", "reconstruction_mae_validation") :
        assert key in computed_metrics_during_training, f"{key} is missing from the tracked metrics"
        assert len(computed_metrics_during_training[key]) == EPOCHS, f"{key} has {len(computed_metrics_during_training[key])} values"
    print("  [OK] reconstruction metrics: MSE and MAE tracked for both sets")

    # The KL warm-up must ramp from kl_warmup_start (0) to 1 over KL_WARMUP_EPOCHS epochs.
    kl_weight = computed_metrics_during_training["kl_weight"]
    expected_kl_weight = [0.0, 0.5, 1.0]
    assert kl_weight == expected_kl_weight, f"the KL weight went {kl_weight} instead of {expected_kl_weight}"
    print(f"  [OK] KL warm-up          : {kl_weight}")

    # The model must actually learn to reconstruct.
    # Note that the OPTIMIZED loss (train_loss) is expected to GROW here: the KL warm-up ramps the weight of the KL
    # term from 0 to 1 over the run, so that value changes meaning at every epoch. This is exactly why the cycle
    # tracks (and selects the best model on) loss_full_kl, and why the reconstruction term is the one checked here.
    train_recon_loss = computed_metrics_during_training["train_recon_loss"]
    assert train_recon_loss[-1] < train_recon_loss[0], f"the reconstruction error did not decrease: {train_recon_loss}"
    print(f"  [OK] training            : reconstruction term from {train_recon_loss[0]:.4f} to {train_recon_loss[-1]:.4f}")

    # The weighted loss grows while the warm-up runs, so the two totals must differ before the weight reaches 1.
    train_loss         = computed_metrics_during_training["train_loss"]
    train_loss_full_kl = computed_metrics_during_training["train_loss_full_kl"]
    assert train_loss[0] < train_loss_full_kl[0], "at epoch 0 the optimized loss should be smaller than the one at full KL weight"
    assert abs(train_loss[-1] - train_loss_full_kl[-1]) < 1e-6, "once the warm-up is over the two totals must coincide"
    print(f"  [OK] warm-up effect      : optimized loss {train_loss[0]:.4f} vs full-KL {train_loss_full_kl[0]:.4f} at epoch 0, equal at the end")

    # The checkpoints must be on disk.
    for checkpoint in ("model_END.pth", "model_BEST.pth") :
        path_checkpoint = os.path.join(path_to_save_model, checkpoint)
        assert os.path.isfile(path_checkpoint), f"{checkpoint} was not saved"
    print("  [OK] checkpoints         : model_END.pth and model_BEST.pth saved")

    # The trained model must still reconstruct and generate.
    x = torch.randn(2, 1, C, T).to(device)
    assert trained_model.reconstruct(x).shape == x.shape, "the trained model does not reconstruct the right shape"
    assert trained_model.generate(n_samples = 2).shape == (2, 1, C, T), "the trained model does not generate the right shape"
    print("  [OK] trained model       : reconstruct and generate still work")

    # Clean up the temporary folder.
    shutil.rmtree(os.path.dirname(path_to_save_model), ignore_errors = True)

    print("=" * 70)
    print("All vEEGNet training checks passed.")
    print("=" * 70)


if __name__ == "__main__" :
    main()
