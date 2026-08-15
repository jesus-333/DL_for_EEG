"""
This script is used to train the EEGNet model on different pooling layer on all the subjects of a moabb dataset

The main difference with `train_EEGNet.py` is that this script will launch several training runs, based on the variable specify below.
This is used to iterate over the different subjects and EEGNet configurations.

`train_EEGNet.py` is used to train a single model with a single configuration.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# CLI Arguments

print("\nStarting training script (EEGNet d2a pooling) ...")

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

pooling_layer_list = ['default', 'blur', 'dwt', 'max', 'sinc'] # default is the original EEGNet average pooling layer

# Options are BNCI2014_001 (Dataset 2a BCI Competition IV), Schirrmeister2017 (High Gamma Dataset), Weibo2014, Cho2017
dataset_name = "Cho2017"

# Used when kernels are resized for dataset with sampling frequency different from 128 Hz.
# If True, the kernel size will be forced to be odd. If False, the kernel size will be rounded to the nearest integer.
keep_kernel_odd = True

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

# Logger setup
logging_utils.setup_logging()

# Device selection
if torch.cuda.is_available() :
    device = "cuda"
    print("\nCUDA backend in use")
elif torch.backends.mps.is_available():
    device = "mps"
    print("\nmps backend (apple metal) in use")
else:
    device = "cpu"
    print("\nNo backend in use. Device set to cpu")

# Override some config values (iteration independent)

training_config_dict["device"] = device

dataset_config_dict['dataset_name'] = dataset_name

if dataset_config_dict['dataset_name'] == "BNCI2014_001" :
    subjects_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    training_config_dict['wandb']['project_name'] = "EEGNet_pooling_d2a"
elif dataset_config_dict['dataset_name'] == "Schirrmeister2017" :
    subjects_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    training_config_dict['wandb']['project_name'] = "EEGNet_pooling_HGD"
elif dataset_config_dict['dataset_name'] == "Weibo2014" :
    subjects_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    training_config_dict['wandb']['project_name'] = "EEGNet_pooling_Weibo"
elif dataset_config_dict['dataset_name'] == "Cho2017" :
    subjects_list = np.arange(1, 53) # Cho2017 has 52 subjects. Arange create a list from start (included) to end (excluded)
    training_config_dict['wandb']['project_name'] = "EEGNet_pooling_Cho2017"
else :
    raise ValueError(f"Invalid dataset name {dataset_config_dict['dataset_name']}")

def compute_EEGNet_kernel_size(original_kernel_size : int, current_signal_fs : int, keep_kernel_odd : bool = True) -> int :
    """
    The size of the convolutional time kernel in EEGNet where designed with the dataset 2a of BCI Competition IV in mind.
    More precisely, they were designed to work with a resampled version of the dataset 2a, where the original signal was downsampled to 128 Hz.

    This function computes the new kernel size to use for a different sampling frequency, based on the original kernel size and the original sampling frequency (128 Hz).
    """

    if original_kernel_size != 64 and original_kernel_size != 16 :
        # Note that the two original kernel sizes are 64 and 16, which correspond to the two convolutional time layers of EEGNet.
        raise ValueError(f"Invalid original kernel size {original_kernel_size}. Must be 64 or 16.")
    else :
        # Compute the new kernel size based on the original kernel size and the current signal sampling frequency
        original_signal_fs = 128 # Hz
        new_kernel_size = round(original_kernel_size * current_signal_fs / original_signal_fs)
        
        # (OPTIONAL) Make sure that the new kernel size is odd
        if keep_kernel_odd and new_kernel_size % 2 == 0 : new_kernel_size += 1

        return new_kernel_size

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

for i in range(len(subjects_list)) :
    # Get subject and update the dataset config with the current subject
    subject = subjects_list[i]
    dataset_config_dict["subjects"] = [subject]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Dataset creation

    # Create the dataset_config object from the dictionary
    dataset_config_obj = dataset_config.from_dict(dataset_config_dict)

    # Create the EEG_Dataset object from the dataset_config object
    # Remember that data saved inside the EEG_dataset is a tensor of shape [N x 1 x C x T], where N is the number of trials, C is the number of channels and T is the number of time samples.
    # Note that the function get dataset will try to split the dataset based on the variable `use_original_split` inside the dataset_config.
    # If that variable is set to True, the dataset will be splitted based on sessions names (e.g. "session_T" and "session_E" for BNCI2014_001).
    # If the variable is set to False, the data of all sessions will be put together and the percentage indicated by the variable `split_train_test` will be used for training and the rest for testing.
    # Note that not all datasets have a predefined split. For those datasets, the variable `use_original_split` will do nothing and the split through `split_train_test` will be used.
    eeg_dataset_train, eeg_dataset_test = moabb_dataset.get_dataset(dataset_config_obj)
    
    print(f"Dataset created for subject {subject}.")
    print(f"Data shape (TRAIN)   : {eeg_dataset_train.data.shape}.")
    print(f"Labels shape (TRAIN) : {eeg_dataset_train.labels.shape}.")
    print(f"Data shape (TEST)    : {eeg_dataset_test.data.shape}.")
    print(f"Labels shape (TEST)  : {eeg_dataset_test.labels.shape}.")

    for j in range(len(pooling_layer_list)) :
        # Get pooling layer
        pooling_layer = pooling_layer_list[j]

        # Update pooling type in model config
        # Note that even for the non-default pooling layers, we will use their default hyperparameters.
        model_config_dict['pooling']['pooling_type_1'] = pooling_layer
        model_config_dict['pooling']['pooling_type_2'] = pooling_layer

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
        
        # Update kernel size if required. Note that the check for 1 is used to avoid resizing when the normalize frequency is used.
        # Also by default, if sampling frequency is not provided when the dataset is created it will be set to 1.
        if eeg_dataset_train.fs != 128 and eeg_dataset_train.fs != 1 :
            # Compute the new kernel sizes for the two convolutional time layers of EEGNet, based on the current signal sampling frequency and the original kernel sizes (64 and 16).
            c_kernel_1 = compute_EEGNet_kernel_size(64, eeg_dataset_train.fs, keep_kernel_odd = keep_kernel_odd)
            c_kernel_3 = compute_EEGNet_kernel_size(16, eeg_dataset_train.fs, keep_kernel_odd = keep_kernel_odd)
            
            # Update model config
            model_config_dict["c_kernel_1"] = [1, c_kernel_1]
            model_config_dict["c_kernel_3"] = [1, c_kernel_3]

        # Create model config object
        model_config_obj = EEGNet_config.from_dict(model_config_dict)
        print(f"Model config:\n{model_config_obj}")

        # Create model
        model = EEGNet_Classifier(model_config_obj)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Training
        
        # Update training config with custom values (iteration dependent)
        training_config_dict['seed'] = np.random.randint(0, 100000000) # Random seed for each training run. Note that the seed is also used as run_id for wandb. Keeping the same seed means that all the trainings will be logged on the same wandb run
        training_config_dict['wandb']['name_training_run'] = f"S{subject}_pooling_{pooling_layer}"

        # Create training config object
        traingin_config_object = training_config.from_dict(training_config_dict)
        print(f"\nTraining config:\n{traingin_config_object}")

        # Train the model
        print(f"\nSTART TRAINING EEGNet for subject {subject} with pooling layer {pooling_layer}...")
        model = classifier.wandb_train_model(traingin_config_object, model, eeg_dataset_train, eeg_dataset_test, model_config_obj, dataset_config_obj)
        print(f"\nEND TRAINING EEGNet for subject {subject} with pooling layer {pooling_layer}.")
