"""
- Create an EEGNet
- Create synthetic data of shape [N x 1 x C x T] with N = n. of training samples, C = n. of channels, T = n. of time points
- Test training function (1 epoch)

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

# External libraries
import numpy as np
import toml
import torch

# Internal libraries
from jade.config import dataset_config, EEGNet_config, training_config
from jade.dataset import EEG_Dataset
from jade.model import EEGNet_Classifier
from jade.training import classifier, logging_utils

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

path_model_config = "./custom_config/model/EEGNet.toml"
path_training_config = "./custom_config/train/training_clf.toml"

# Number of training samples, time points and channels
N = 10
T = 1000
C = 22

n_classes = 4

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Model creation

# Load config from file
model_config_dict = toml.load(path_model_config)

# Update config with custom values
model_config_dict["input_size"] = (1, 1, C, T)
model_config_dict["n_classes"]  = n_classes
model_config_dict["C"]          = C
model_config_dict["T"]          = T

# Create model config object
model_config = EEGNet_config.from_dict(model_config_dict)
print(f"Model config:\n{model_config}")

# Create model
model = EEGNet_Classifier(model_config)

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
