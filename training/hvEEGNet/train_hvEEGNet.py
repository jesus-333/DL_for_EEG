"""
Train the hvEEGNet model (hierarchical variational autoencoder) with a specified dataset from moabb

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# CLI Arguments

print("\nStarting training script...")

import argparse

parser = argparse.ArgumentParser(description = 'Train hvEEGNet model.')
parser.add_argument('--path_dataset_config'              , type = str  , default = None, help = 'Path to the dataset toml config file. If not provided, the default path (see default settings section) will be used.')
parser.add_argument('--path_model_config'                , type = str  , default = None, help = 'Path to the model toml config file. If not provided, the default path (see default settings section) will be used.')
parser.add_argument('--path_training_config'             , type = str  , default = None, help = 'Path to the training toml config file. If not provided, the default path (see default settings section) will be used.')
parser.add_argument('--scale_factor'                     , type = float, default = 1e6 , help = 'Factor multiplied to the EEG data after loading. The default (1e6) converts the data from volt (the unit of MNE / moabb) to microvolt, as in the original hvEEGNet code. Use 1 to keep the data unchanged.')
parser.add_argument('--percentage_validation'            , type = float, default = 0.1 , help = 'Fraction of the training set used for validation (0.1 in the hvEEGNet paper). If 0, the test set is used as validation set (as in the EEGNet training script).')

args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

# External libraries
import toml
import torch

# Internal libraries
from jade.config import dataset_config, hvEEGNet_config, training_config
from jade.dataset import moabb_dataset
from jade.model import hvEEGNet
from jade.training import vae, logging_utils

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Default settings. Used only if the corresponding argument is not provided when launching the script.
# You can use these paths if you want to execute directly this script with the "python" command, without passing any argument.
# The default configs reproduce the training conditions of the hvEEGNet paper (Cisotto et al., 2024) : BCI IV-2a (one subject), original train/test split, batch size 30, 80 epochs, AdamW (lr 0.01), ExponentialLR (gamma 0.999), ELBO with SoftDTW.

# Path to config files.
default_path_config_dataset  = './training/config/starting_config/dataset/dataset_hvEEGNet.toml'
default_path_config_model    = './training/config/starting_config/model/hvEEGNet.toml'
default_path_config_training = './training/config/starting_config/train/training_hvEEGNet.toml'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load config

# Check if the user provided a path to the config files, otherwise use the default paths
path_config_dataset  = args.path_dataset_config if args.path_dataset_config is not None else default_path_config_dataset
path_config_model    = args.path_model_config if args.path_model_config is not None else default_path_config_model
path_config_training = args.path_training_config if args.path_training_config is not None else default_path_config_training

print("Debug Python")
print(f"path_config_dataset  = {path_config_dataset}")
print(f"path_config_model    = {path_config_model}")
print(f"path_config_training = {path_config_training}")

# Load configs
dataset_config_dict  = toml.load(path_config_dataset)
model_config_dict    = toml.load(path_config_model)
training_config_dict = toml.load(path_config_training)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dataset creation

# Create the dataset_config object from the dictionary
dataset_config_obj = dataset_config.from_dict(dataset_config_dict)
print(f"Dataset config:\n{dataset_config_obj}")

# Create the EEG_Dataset object from the dataset_config object
# Remember that data saved inside the EEG_dataset is a tensor of shape [N x 1 x C x T], where N is the number of trials, C is the number of channels and T is the number of time samples.
# Note that the function get dataset will try to split the dataset based on the variable `use_original_split` inside the dataset_config.
# If that variable is set to True, the dataset will be splitted based on sessions names (e.g. "session_T" and "session_E" for BNCI2014_001).
# If the variable is set to False, the data of all sessions will be put together and the percentage indicated by the variable `split_train_test` will be used for training and the rest for testing.
# Note that not all datasets have a predefined split. For those datasets, the variable `use_original_split` will do nothing and the split through `split_train_test` will be used.
# The labels are loaded too, but hvEEGNet does not use them (it is trained to reconstruct its input).
eeg_dataset_train, eeg_dataset_test = moabb_dataset.get_dataset(dataset_config_obj)

print("Dataset created")

# Convert the data from volt (unit used by MNE / moabb) to microvolt, as in the original hvEEGNet code.
# This is necessary with the SoftDTW: with signals of the order of 1e-5 the soft-min (gamma = 1) is dominated by the number of alignment paths and not by the signal,
# so the reconstruction loss becomes a large negative number that the decoder minimizes without reconstructing the input.
eeg_dataset_train.data *= args.scale_factor
eeg_dataset_test.data  *= args.scale_factor
print(f"Data multiplied by {args.scale_factor} (std of the train data : {eeg_dataset_train.data.std().item():.4g})")
print(f"Data shape (TRAIN)   : {eeg_dataset_train.data.shape}.")
print(f"Data shape (TEST)    : {eeg_dataset_test.data.shape}.")

# Split the training set in train and validation (90/10 in the paper).
# The split is random but reproducible : it uses the seed of the training config (set by 0_prepare_config.py, so it changes for every run, as the 20 runs per subject of the paper, each with a different train/validation split).
# If the training config has no seed, the seed of the dataset config is used.
# If percentage_validation is 0, the test set is used as validation set (as in the EEGNet training script).
if not 0 <= args.percentage_validation < 1 :
    raise ValueError(f"percentage_validation must be in [0, 1). Current value : {args.percentage_validation}")

if args.percentage_validation > 0 :
    split_seed   = training_config_dict.get("seed", dataset_config_obj.seed)
    n_validation = int(len(eeg_dataset_train) * args.percentage_validation) # Floor, so 288 trials give 260 train and 28 validation, as in the paper
    n_train      = len(eeg_dataset_train) - n_validation
    eeg_dataset_train, eeg_dataset_validation = torch.utils.data.random_split(eeg_dataset_train, [n_train, n_validation], generator = torch.Generator().manual_seed(split_seed))
    print(f"Training set split in train ({n_train} trials) and validation ({n_validation} trials)")
else :
    eeg_dataset_validation = eeg_dataset_test
    print("Test set used as validation set")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Model creation

# Get dataset parameters
C = eeg_dataset_test.data.shape[2] # Number of EEG channels
T = eeg_dataset_test.data.shape[3] # Number of time samples

# Update config with custom values
# Note that, contrary to EEGNet, the input_size drives the whole decoder geometry of hvEEGNet, so it must match C and T.
# The spatial kernel spans all the channels, as in the original hvEEGNet.
# The temporal kernels (c_kernel_1, c_kernel_3) and the pooling factors are NOT adapted to the sampling frequency: the default values are the ones of the paper (250 Hz). Edit the model config if your data have a different sampling frequency.
model_config_dict["input_size"] = (1, 1, C, T)
model_config_dict["C"]          = C
model_config_dict["T"]          = T
model_config_dict["c_kernel_2"] = (C, 1)

# Create model config object
model_config_obj = hvEEGNet_config.from_dict(model_config_dict)
print(f"Model config:\n{model_config_obj}")

# Create model
model = hvEEGNet(model_config_obj)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Training

# Logger setup
logging_utils.setup_logging()

if torch.cuda.is_available() :
    device = "cuda"
    print("\nCUDA backend in use")
elif torch.backends.mps.is_available():
    device = "mps"
    print("\nmps backend (apple metal) in use")
else:
    device = "cpu"
    print("\nNo backend in use. Device set to cpu")

# Update config with custom values
training_config_dict["device"] = device

# Create training config object
training_config_object = training_config.from_dict(training_config_dict)
print(f"\nTraining config:\n{training_config_object}")

# Train the model
# The training is tracked with wandb only if the training config contains the [wandb] section.
# Note that the test set is used as validation set only if percentage_validation is 0 (see the dataset creation section).
if training_config_object.wandb is not None :
    model = vae.wandb_train_model(training_config_object, model, eeg_dataset_train, eeg_dataset_validation, model_config_obj, dataset_config_obj)
else :
    model = vae.train_model(training_config_object, model, eeg_dataset_train, eeg_dataset_validation)
