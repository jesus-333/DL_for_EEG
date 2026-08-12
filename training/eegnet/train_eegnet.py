"""
Train the EEGNet model with a specified dataset from moaab

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# CLI Arguments and check

import argparse

parser = argparse.ArgumentParser(description = 'Train dement model in a centralized way.')
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
defualt_path_config_dataset  = './config/default_config/dataset/dataset.toml'
defualt_path_config_model    = './config/default_config/model/EEGNet.toml'
defualt_path_config_training = './config/demnet_wandb/training.toml'

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
dataset_config_obj = dataset_config.dataset_config.from_dict(dataset_config_dict)
print(f"Dataset config:\n{dataset_config_obj}")

# Create the EEG_Dataset object from the dataset_config object
eeg_dataset_train = moabb_dataset.get_dataset(dataset_config_obj, dataset_type = 'train')
eeg_dataset_test  = moabb_dataset.get_dataset(dataset_config_obj, dataset_type = 'test')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Model creation

# Create model config object
model_config_obj = EEGNet_config.from_dict(model_config_dict)
print(f"Model config:\n{model_config_obj}")

# Create model
model = EEGNet_Classifier(model_config_obj)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Synthetic data and dataset_creation

def generate_dataset(N, C, T, n_classes) :
    # Create synthetic data and labels
    data = np.random.rand(N, 1, C, T).astype(np.float32)
    labels = np.random.randint(0, n_classes, size=(N,)).astype(np.int64)
    ch_list = [f"Ch{i + 1}" for i in range(C)]

    # Dataset creation
    dataset = EEG_Dataset(data = data, labels = labels, ch_list = ch_list)

    return dataset

# Create synthetic datasets (train and validation)
train_dataset      = generate_dataset(N, C, T, n_classes)
validation_dataset = generate_dataset(N, C, T, n_classes)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Logger setup

logging_utils.setup_logging()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Training

if torch.cuda.is_available() :
    device = "cuda"
    print("\nCUDA backend in use")
elif torch.backends.mps.is_available():
    device = "mps"
    print("\nmps backend (apple metal) in use")
else:
    device = "cpu"
    print("\nNo backend in use. Device set to cpu")

# Load training config
training_config_dict = toml.load(path_training_config)

# Update config with custom values
training_config_dict["epochs"]      = 1
training_config_dict["batch_size"]  = N // 2
training_config_dict["device"]      = device

# Create training config object
traingin_config_object = training_config.from_dict(training_config_dict)
print(f"\nTraining config:\n{traingin_config_object}")

# Train the model
model = classifier.train_model(traingin_config_object, model, train_dataset, validation_dataset = validation_dataset, wandb_model_artifact = None)
