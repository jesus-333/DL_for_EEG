"""
Train the EEGNet model with a specified dataset from moaab

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# CLI Arguments

print("\nStarting training script...")

import argparse

parser = argparse.ArgumentParser(description = 'Train EEGNet model.')
parser.add_argument('--path_dataset_config'              , type = str  , default = None, help = 'Path to the dataset toml config file. If not provided, an error will be thrown.')
parser.add_argument('--path_model_config'                , type = str  , default = None, help = 'Path to the model toml config file. If not provided, an error will be thrown.')
parser.add_argument('--path_training_config'             , type = str  , default = None, help = 'Path to the training toml config file. If not provided, an error will be thrown.')

args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

# External libraries
import numpy as np
import toml
import torch

# Internal libraries
from jade.config import dataset_config, EEGNet_config, training_config
from jade.dataset import EEG_Dataset, moabb_dataset
from jade.model import EEGNet_Classifier
from jade.training import classifier, logging_utils

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Default settings. Used only if the corresponding argument is not provided when launching the script.
# You can use this folder if you want to execute directly this script with the "python" command, without passing any argument.

# Path to config files.
defualt_path_config_dataset  = './training/config/starting_config/dataset/dataset.toml'
defualt_path_config_model    = './training/config/starting_config/model/EEGNet.toml'
defualt_path_config_training = './training/config/starting_config/train/training_clf.toml'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load config

# Check if the user provided a path to the config files, otherwise use the default paths
path_config_dataset  = args.path_dataset_config if args.path_dataset_config is not None else defualt_path_config_dataset
path_config_model    = args.path_model_config if args.path_model_config is not None else defualt_path_config_model
path_config_training = args.path_training_config if args.path_training_config is not None else defualt_path_config_training

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
eeg_dataset_train = moabb_dataset.get_dataset(dataset_config_obj, dataset_type = 'train')
eeg_dataset_test  = moabb_dataset.get_dataset(dataset_config_obj, dataset_type = 'test')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Model creation

# Get dataset parameters
n_classes = len(np.unique(eeg_dataset_train.labels))
C = eeg_dataset_train.data.shape[2] # Number of EEG channels
T = eeg_dataset_train.data.shape[3] # Number of time samples

# Update config with custom values
model_config_dict["input_size"] = (1, 1, C, T)
model_config_dict["n_classes"]  = n_classes
model_config_dict["C"]          = C
model_config_dict["T"]          = T

# Create model config object
model_config_obj = EEGNet_config.from_dict(model_config_dict)
print(f"Model config:\n{model_config_obj}")

# Create model
model = EEGNet_Classifier(model_config_obj)

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
traingin_config_object = training_config.from_dict(training_config_dict)
print(f"\nTraining config:\n{traingin_config_object}")

# Train the model
model = classifier.wandb_train_model(traingin_config_object, model, eeg_dataset_train, eeg_dataset_test, model_config_obj, dataset_config_obj)
