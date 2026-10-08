"""
Used the weights save at each epoch during training compute the metrics required.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import os
import toml
import torch

from jade.dataset import EEG_Dataset, moabb_dataset

from jade.config import dataset_config, EEGNet_config, training_config
from other.aliasing_experiment.z_model import EEGNet_aliasing
import other.aliasing_experiment.z_support_analysis as support_analysis


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

experiment_name = "d2a_subj_3"

path_results = f'./other/aliasing_experiment/results/{experiment_name}/'

# For each repetition create a tensor of shape (epochs, samples, depth channels)
save_tensors_per_repetition = True

# For each experiment create a tensor of shape (repetitions, epochs, samples, depth channels)
save_single_tensor_all_repetitions = True

# Note that at least one of the two options above must be set to True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if not save_tensors_per_repetition and not save_single_tensor_all_repetitions : raise ValueError("At least one of the two options must be set to True: save_tensors_per_repetition, save_single_tensor_all_repetitions")

if torch.cuda.is_available() :
    device = "cuda"
    print("\nCUDA backend in use")
elif torch.backends.mps.is_available():
    device = "mps"
    print("\nmps backend (apple metal) in use")
else:
    device = "cpu"
    print("\nNo backend in use. Device set to cpu")

list_experiment_repetition = support_analysis.get_all_folders_in_folder(path_results)

energy_ratio_train_1_all_repetitions = []
energy_ratio_test_1_all_repetitions  = []
energy_ratio_train_2_all_repetitions = []
energy_ratio_test_2_all_repetitions  = []

for i in range(len(list_experiment_repetition)) :
    print(f"Processing repetition {i + 1} / {len(list_experiment_repetition)}")

    path_results_repetition = list_experiment_repetition[i]
    path_results_metrics = os.path.join(path_results_repetition, "metrics/")
    path_results_config = os.path.join(path_results_repetition, "config_backup/")

    # Get dataset
    dataset_config_dict = toml.load(os.path.join(path_results_config, "dataset_config.toml"))
    dataset_config_obj = dataset_config.from_dict(dataset_config_dict)
    eeg_dataset_train, eeg_dataset_test = moabb_dataset.get_dataset(dataset_config_obj)

    # Get model
    model_config_dict = toml.load(os.path.join(path_results_config, "model_config.toml"))
    model_confg_obj = EEGNet_config.from_dict(model_config_dict)
    model = EEGNet_aliasing(model_confg_obj)
    model.to(device)
    
    # Get training config
    training_config_dict = toml.load(os.path.join(path_results_config, "training_config.toml"))
    training_config_obj = training_config.from_dict(training_config_dict)

    # Create dataloaders
    dataloader_train = torch.utils.data.DataLoader(eeg_dataset_train, batch_size = training_config_obj.batch_size, shuffle = True)
    dataloader_test  = torch.utils.data.DataLoader(eeg_dataset_test , batch_size = training_config_obj.batch_size, shuffle = True)

    energy_ratio_train_1_repetition = []
    energy_ratio_test_1_repetition  = []
    energy_ratio_train_2_repetition = []
    energy_ratio_test_2_repetition  = []
    
    step_print = int(training_config_obj.epochs / 10)
    for epoch in range(training_config_obj.epochs) :
        if (epoch + 1) % step_print == 0 : print(f"\tProcessing epoch {epoch + 1} / {training_config_obj.epochs}")

        with torch.no_grad() :
            # Load model weights
            path_model_weights = os.path.join(training_config_obj.path_to_save_model, f"model_{epoch + 1}.pth")
            model.load_state_dict(torch.load(path_model_weights))
            model.eval()

            # Compute aliased energy ratio before the first pooling layer
            aliased_energy_ratio_train_1, aliased_energy_ratio_train_2 = support_analysis.compute_aliased_energy_ratio_dataloader(model, dataloader_train, device, eeg_dataset_train.fs)
            aliased_energy_ratio_test_1, aliased_energy_ratio_test_2 = support_analysis.compute_aliased_energy_ratio_dataloader(model, dataloader_test, device, eeg_dataset_test.fs)

            # Save the metrics for this epoch
            energy_ratio_train_1_repetition.append(aliased_energy_ratio_train_1.squeeze().cpu())
            energy_ratio_test_1_repetition.append(aliased_energy_ratio_test_1.squeeze().cpu())
            energy_ratio_train_2_repetition.append(aliased_energy_ratio_train_2.squeeze().cpu())
            energy_ratio_test_2_repetition.append(aliased_energy_ratio_test_2.squeeze().cpu())

    # Save the metrics for this repetition
    energy_ratio_train_1_all_repetitions.append(torch.stack(energy_ratio_train_1_repetition))
    energy_ratio_test_1_all_repetitions.append(torch.stack(energy_ratio_test_1_repetition))
    energy_ratio_train_2_all_repetitions.append(torch.stack(energy_ratio_train_2_repetition))
    energy_ratio_test_2_all_repetitions.append(torch.stack(energy_ratio_test_2_repetition))

    # Save the metrics for this repetition to disk
    if save_tensors_per_repetition :
        torch.save(torch.stack(energy_ratio_train_1_repetition), os.path.join(path_results_metrics, "energy_ratio_train_1.pt"))
        torch.save(torch.stack(energy_ratio_test_1_repetition) , os.path.join(path_results_metrics, "energy_ratio_test_1.pt"))
        torch.save(torch.stack(energy_ratio_train_2_repetition), os.path.join(path_results_metrics, "energy_ratio_train_2.pt"))
        torch.save(torch.stack(energy_ratio_test_2_repetition) , os.path.join(path_results_metrics, "energy_ratio_test_2.pt"))

# Convert to tensor
energy_ratio_train_1_all_repetitions = torch.stack(energy_ratio_train_1_all_repetitions)
energy_ratio_test_1_all_repetitions  = torch.stack(energy_ratio_test_1_all_repetitions)
energy_ratio_train_2_all_repetitions = torch.stack(energy_ratio_train_2_all_repetitions)
energy_ratio_test_2_all_repetitions  = torch.stack(energy_ratio_test_2_all_repetitions)

# Save the metrics to disk
if save_single_tensor_all_repetitions :
    torch.save(energy_ratio_train_1_all_repetitions, os.path.join(path_results, "energy_ratio_train_1_all_repetitions.pt"))
    torch.save(energy_ratio_test_1_all_repetitions , os.path.join(path_results, "energy_ratio_test_1_all_repetitions.pt"))
    torch.save(energy_ratio_train_2_all_repetitions, os.path.join(path_results, "energy_ratio_train_2_all_repetitions.pt"))
    torch.save(energy_ratio_test_2_all_repetitions , os.path.join(path_results, "energy_ratio_test_2_all_repetitions.pt"))


