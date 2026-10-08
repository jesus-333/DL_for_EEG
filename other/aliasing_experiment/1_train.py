"""
Train the EEGNet model for the aliasing experiment.

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
import os
import sys
import toml
import torch

# Jade libraries
from jade.config import dataset_config, EEGNet_config, training_config
from jade.dataset import EEG_Dataset, moabb_dataset
from jade.training import classifier_epoch, logging_utils, support_training, metrics

# Experiment imports
from other.aliasing_experiment.z_model import EEGNet_aliasing
import other.aliasing_experiment.z_support_analysis as support_analysis

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

seed = torch.random.seed()
# seed = 42 # Uncomment this line to set a fixed seed for reproducibility

# (OPTIONAL) Name of the experiment. If specified, it will change the name of the folder where the results are saved.
# See the variable path_results below
name_experiment = None
name_experiment = "d2a_subj_3"

# Uncomment the following lines to use a timestamp as the name of the experiment (useful for multiple runs).
# import time
# name_experiment = f"{time.strftime('%Y_%m_%d_%H_%M_%S')}" if name_experiment is None else f"{name_experiment}_{time.strftime('%Y_%m_%d_%H_%M_%S')}"

# The script saves the optimizer state_dict every epoch. If you want to delete those files after training, set this variable to True.
delete_optimizer_files = True 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Path to config files.
defualt_path_config_dataset  = './other/aliasing_experiment/config/dataset/dataset.toml'
defualt_path_config_model    = './other/aliasing_experiment/config/model/EEGNet.toml'
defualt_path_config_training = './other/aliasing_experiment/config/train/training_clf.toml'

# Path to save results (models, metrics, etc.)
path_results = f'./other/aliasing_experiment/results/no_name_experiment/{seed}/' if name_experiment is None else f'./other/aliasing_experiment/results/{name_experiment}/{seed}/'
path_results_metrics = os.path.join(path_results, "metrics/")
os.makedirs(path_results_metrics, exist_ok = True)

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

# Set the random seed in the config dictionaries (for reproducibility)
dataset_config_dict["seed"]  = seed
training_config_dict["seed"] = seed

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
eeg_dataset_train, eeg_dataset_test = moabb_dataset.get_dataset(dataset_config_obj)

print("Dataset created")
print(f"Data shape (TRAIN)   : {eeg_dataset_train.data.shape}.")
print(f"Labels shape (TRAIN) : {eeg_dataset_train.labels.shape}.")
print(f"Data shape (TEST)    : {eeg_dataset_test.data.shape}.")
print(f"Labels shape (TEST)  : {eeg_dataset_test.labels.shape}.")

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
model = EEGNet_aliasing(model_config_obj)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Training preparation

if torch.cuda.is_available() :
    device = "cuda"
    print("\nCUDA backend in use")
elif torch.backends.mps.is_available():
    device = "mps"
    print("\nmps backend (apple metal) in use")
else:
    device = "cpu"
    print("\nNo backend in use. Device set to cpu")

# Update training config with some specific stuff necessary for this experiment (e.g. device, path to save model, etc.)
training_config_dict["device"] = device # Device automatically detected
training_config_dict["path_to_save_model"] = os.path.join(path_results, "model_weights/") # Path to save model with the seed (for reproducibility)
training_config_dict["epoch_to_save_model"] = 1 # Save a checkpoint every epoch (i.e. model_n.pth). Used later for analysis

# Create training config object
training_config_object = training_config.from_dict(training_config_dict)
print(f"\nTraining config:\n{training_config_object}")

# Create the dataloaders for training, validation and testing.
train_loader = support_training.get_dataloader(eeg_dataset_train, training_config_object.batch_size, training_config_object.dataloader)
validation_loader = None
test_loader  = support_training.get_dataloader(eeg_dataset_test , training_config_object.batch_size, training_config_object.dataloader)

# Move the model to the device (GPU or CPU).
model.to(device)

# Create the loss function and move it to the device (the move matters when the loss carries class weights).
loss_function = support_training.get_loss_function(training_config_object.loss)
loss_function.to(device)

# Create the optimizer.
optimizer = support_training.get_optimizer(training_config_object.optimizer, model)

# (OPTIONAL) Create the learning-rate scheduler (used only when a scheduler config is provided).
if training_config_object.lr_scheduler is not None :
    lr_scheduler = support_training.get_lr_scheduler(training_config_object.lr_scheduler, optimizer)
else :
    lr_scheduler = None

# Create the folder where the checkpoints are stored (if it does not exist already).
os.makedirs(training_config_object.path_to_save_model, exist_ok = True)

# Variable to track the best (lowest) validation loss.
best_loss_val = sys.maxsize

# Two dictionaries with different purposes:
# - computed_metrics_during_training stores, for each metric, the list of its per-epoch values, and it is returned at the end.
# - log_dict stores the metrics of a single epoch and it is used by wandb. It is not reset explicitly at each epoch: the same keys are overwritten, which has the same effect.
log_dict = {}
computed_metrics_during_training = dict()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Backup of the config dictionaries (for reproducibility)
path_backup_config_folder = os.path.join(path_results, "config_backup")
os.makedirs(path_backup_config_folder, exist_ok = True)
with open(os.path.join(path_backup_config_folder, "dataset_config.toml"), 'w') as f  : toml.dump(dataset_config_dict, f)
with open(os.path.join(path_backup_config_folder, "model_config.toml"), 'w') as f    : toml.dump(model_config_dict, f)
with open(os.path.join(path_backup_config_folder, "training_config.toml"), 'w') as f : toml.dump(training_config_dict, f)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Training loop

for epoch in range(training_config_object.epochs) :
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # (MANDATORY) Training epoch

    # Advance one epoch on the training set (forward + backward pass).
    train_loss = classifier_epoch.train_epoch_function(model, train_loader, loss_function, optimizer, device, log_dict, training_config_object.print_var)

    # Save a checkpoint every epoch_to_save_model epochs (i.e. model_n.pth).
    # Note that when epoch == n the model has been trained for n + 1 epochs.
    if (epoch + 1) % training_config_object.epoch_to_save_model == 0 :
        torch.save(model.state_dict(), os.path.join(training_config_object.path_to_save_model, f"model_{epoch + 1}.pth"))
        torch.save(optimizer.state_dict(), os.path.join(training_config_object.path_to_save_model, f"optimizer_{epoch + 1}.pth"))

    # (OPTIONAL) Save an alternating A/B backup every epoch (overwritten each time; handy for very large models).
    if training_config_object.backup_model_every_epoch :
        support_training.backup_every_epoch(training_config_object.path_to_save_model, epoch, model, optimizer, lr_scheduler)

    # Accumulate the training loss (create the list on the first epoch, append afterwards).
    if epoch == 0 :
        computed_metrics_during_training["train_loss"] = [train_loss]
    else :
        computed_metrics_during_training["train_loss"].append(train_loss)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # (OPTIONAL) Validation epoch

    validation_loss = None
    if validation_loader is not None :
        validation_loss = classifier_epoch.validation_epoch_function(model, validation_loader, loss_function, device, log_dict)

        # Save the new BEST model when a new minimum validation loss is reached.
        if validation_loss < best_loss_val :
            best_loss_val = validation_loss
            new_best_found = True
            torch.save(model.state_dict(), os.path.join(training_config_object.path_to_save_model, "model_BEST.pth"))
            torch.save(optimizer.state_dict(), os.path.join(training_config_object.path_to_save_model, "optimizer_BEST.pth"))
        else :
            new_best_found = False

        # Accumulate the validation loss.
        if epoch == 0 :
            computed_metrics_during_training["validation_loss"] = [validation_loss]
        else :
            computed_metrics_during_training["validation_loss"].append(validation_loss)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # (OPTIONAL) Classification metrics

    train_metrics_dict = None
    validation_metrics_dict = None
    if training_config_object.measure_metrics_during_training :
        # Compute the metrics on the training set (and validation set, if any).
        train_metrics_dict = metrics.compute_metrics(model, train_loader, device)
        if validation_loader is not None :
            validation_metrics_dict = metrics.compute_metrics(model, validation_loader, device)

            if new_best_found :
                # Save the metrics of the new BEST model (TODO)

                # Save the current accuracy (used for log)
                best_accuracy_val = validation_metrics_dict["accuracy"]

        # Accumulate each metric (create the list on the first epoch, append afterwards).
        for metric in train_metrics_dict :
            if epoch == 0 :
                computed_metrics_during_training[f"{metric}_train"] = [train_metrics_dict[metric]]
                if validation_loader is not None :
                    computed_metrics_during_training[f"{metric}_validation"] = [validation_metrics_dict[metric]]
            else :
                computed_metrics_during_training[f"{metric}_train"].append(train_metrics_dict[metric])
                if validation_loader is not None :
                    computed_metrics_during_training[f"{metric}_validation"].append(validation_metrics_dict[metric])

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Learning-rate logging and scheduler step

    # (OPTIONAL) Update the learning rate.
    if lr_scheduler is not None : lr_scheduler.step()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Print epoch summary

    if training_config_object.print_var :
        print(f"\nEpoch {epoch + 1}/{training_config_object.epochs} summary:")
        print(f"\tTrain loss             = {train_loss}")
        if validation_loader is not None :
            print(f"\tValidation loss        = {validation_loss}")
            print(f"\tValidation loss (BEST) = {best_loss_val}")
        if lr_scheduler is not None :
            print(f"\tLearning rate          = {optimizer.param_groups[0]['lr']}")
        if training_config_object.measure_metrics_during_training :
            print(f"\tAccuracy (TRAIN)       = {train_metrics_dict['accuracy']}")
            if validation_loader is not None :
                print(f"\tAccuracy (VALID)       = {validation_metrics_dict['accuracy']}")
                print(f"\tAccuracy (VALID/BEST)  = {best_accuracy_val}")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# End of training: save the final model

model_file_path_END = os.path.join(training_config_object.path_to_save_model, "model_END.pth")
torch.save(model.state_dict(), model_file_path_END)

if delete_optimizer_files : support_analysis.remove_all_files_in_folder_with_name(training_config_object.path_to_save_model, "optimizer_")

