"""
Collection of functions used to computed additional information during training that will later used for analysis

-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import numpy as np
import os
import torch

from other.aliasing_experiment.z_model import EEGNet_aliasing

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Metrics-computation functions

def compute_aliased_energy_ratio_dataloader(model : EEGNet_aliasing, dataloader : torch.utils.data.DataLoader, device : torch.device, fs : float = 1) -> [torch.Tensor, torch.Tensor] :
    """
    Compute the aliased energy for the signal in the dataloader for the EEGNet model.

    The ratio is computed for both pooling layer.
    For each layer a tensor of shape (N, C) is returned, where N is the number of trials and C is the number of EEG channels.
    """

    list_ratio_1, list_ratio_2 = [], []

    for x, _ in dataloader :
        # Move the data to the device
        x = x.to(device)

        # Compute the aliased energy ratio for the first pooling layer
        ratio_1 = model.compute_aliased_energy_block_1(x, fs)
        list_ratio_1.append(ratio_1)
        
        # Compute the aliased energy ratio for the second pooling layer
        ratio_2 = model.compute_aliased_energy_block_2(x, fs)
        list_ratio_2.append(ratio_2)

    # Convert the list of tensors to a single tensor
    aliased_energy_ratio_1 = torch.cat(list_ratio_1, dim = 0)
    aliased_energy_ratio_2 = torch.cat(list_ratio_2, dim = 0)

    return aliased_energy_ratio_1, aliased_energy_ratio_2

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Metrics reading functions

def read_aliased_energy_ratio_single_repetition(path_results_repetitions, block : int) -> np.ndarray :
    """
    Read the aliased energy ratio from the metrics file for a given block.

    To run this fucntion you need to run the script 2_computed_metrics_experiment.py first, whith `save_tensors_per_repetition` set to True, so that the metric for that specific repetition is saved to disk.

    Parameters
    ----------
    path_results_metrics : str
        Path to the metrics folder.
    block : int
        EEGNet "block". 1 give the energy ratio for for the first pooling layer, 2 for the second pooling layer.

    Returns
    -------
    np.ndarray
        Array of aliased energy ratios for the specified block across epochs. The final shape is (epochs, N, D), where
        - epochs is the number of training epochs (clearly),
        - N is the number of samples in the dataset
        - D is the number of depth channels in the EEGNet model for that layer (n.b. this should not be confused with the number of EEG channels).
    """

    if block != 1 and block != 2: raise ValueError(f"Block {block} is not supported. Only block 1 and 2 are supported.")
    
    # Read energy ratio for train data
    path_metrics_train = os.path.join(path_results_repetitions, f"metrics/energy_ratio_train_{block}.pt")
    energy_ratio_train = torch.load(path_metrics_train).squeeze().cpu().detach().numpy()

    # Read energy ratio for test data
    path_metrics_test = os.path.join(path_results_repetitions, f"metrics/energy_ratio_test_{block}.pt")
    energy_ratio_test = torch.load(path_metrics_test).squeeze().cpu().detach().numpy()

    return energy_ratio_train, energy_ratio_test

def read_aliased_energy_ratio_experiment(path_results_experiment, block : int) -> np.ndarray :
    """
    Very similar to `read_aliased_energy_ratio_single_repetition`, but this function reads the metrics for all repetitions of the experiment, and returns a tensor of shape (R, epochs, N, D), where R is the number of repetitions.

    Note that this script assumes that the results are stored with a specific folder structure, i.e.
    ```
    path_results_experiment/
        repetition_1/
            model_weights/
        repetition_2/
            model_weights/
        ...
        repetition_R/
            model_weights/
    ```
    
    The script`1_train.py` produce exactly this folder structure.
    Note that the folder inside does not need to be called `repetition_X`, it can be any name, as long as each folder contains the `model_weights` folder with the weights of the model for that repetition.

    N.b. This function also expects that each repetition use the same number of epochs, the same number of depth channels and the same number of samples in the dataset. If this is not the case, the function will raise an error.

    """

    if block != 1 and block != 2: raise ValueError(f"Block {block} is not supported. Only block 1 and 2 are supported.")

    tmp_list_energy_ratio_train, tmp_list_energy_ratio_test = [], []

    # Get all repetitions folders
    list_repetitions_folders = get_all_folders_in_folder(path_results_experiment)

    for repetition_folder in list_repetitions_folders :
        # N.b. this function already return the energy ration in numpy format
        energy_ratio_train, energy_ratio_test = read_aliased_energy_ratio_single_repetition(repetition_folder, block)

        # Save the energy ratio for this repetition to the list
        tmp_list_energy_ratio_train.append(energy_ratio_train)
        tmp_list_energy_ratio_test.append(energy_ratio_test)

    # Convert the list to a numpy array for easier manipulation
    return np.array(tmp_list_energy_ratio_train), np.array(tmp_list_energy_ratio_test)

def get_idx_sorting_depth_channels_by_energy_ratio(energy_ratio_array : np.ndarray) -> np.ndarray :
    """
    The function receive in input an array of shape (epochs, N, D), with
    - epochs is the number of training epochs (clearly)
    - N is the number of samples in the dataset
    - D is the number of depth channels in the EEGNet model for that layer (n.b. this should not be confused with the number of EEG channels).

    The element (i, j, k) of the array is the aliased energy ratio for the i-th epoch, j-th sample and k-th depth channel.

    The output is an array of shape (D,) with the indices that would sort the depth channels by their mean aliased energy ratio at the last epoch.
    The first element of the output array is the index of the depth channel with the lowest mean aliased energy ratio at the last epoch, and the last element is the index of the depth channel with the highest mean aliased energy ratio at the last epoch.
    """

    # Check that the input array has the correct shape
    if len(energy_ratio_array.shape) != 3 : raise ValueError(f"Input array must have shape (epochs, N, D), but got {energy_ratio_array.shape}")

    # Get the last epoch
    last_epoch_energy_ratio = energy_ratio_array[-1, :, :]

    # Compute the mean aliased energy ratio for each depth channel across all samples
    mean_energy_ratio_per_depth_channel = np.mean(last_epoch_energy_ratio, axis = 0)

    # Get the indices that would sort the depth channels by their mean aliased energy ratio
    sorted_indices = np.argsort(mean_energy_ratio_per_depth_channel)

    return sorted_indices


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# General utility functions

def get_all_files_in_folder(folder_path: str, file_extension: str) -> list:
    """
    Get all files in a folder with a specific extension.

    Parameters
    ----------
    folder_path : str
        Path to the folder.
    file_extension : str
        File extension to filter by (e.g., '.txt').

    Returns
    -------
    list
        List of file paths with the specified extension.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided path '{folder_path}' is not a valid directory.")

    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(file_extension)]

def get_all_folders_in_folder(folder_path: str) -> list :
    """
    Get all folders in a folder.

    Parameters
    ----------
    folder_path : str
        Path to the folder.

    Returns
    -------
    list
        List of folder paths.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided path '{folder_path}' is not a valid directory.")

    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

def remove_all_files_in_folder_with_name(folder_path: str, name_substring: str) -> None:
    """
    Remove all files in a folder that contain a specific substring in their name.

    Parameters
    ----------
    folder_path : str
        Path to the folder.
    name_substring : str
        Substring to filter files by.

    Returns
    -------
    None
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided path '{folder_path}' is not a valid directory.")

    for f in os.listdir(folder_path):
        if name_substring in f:
            file_path = os.path.join(folder_path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Time series 

def smooth_time_series(data : np.array, smooth_type : str = 'moving_average', window_size : int = 5) -> list:
    """
    Smooth a time series data using the specified smoothing method.
    """

    if smooth_type == 'moving_average':
        return np.convolve(data, np.ones(window_size) / window_size, mode = 'valid')
    elif smooth_type == 'gaussian':
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(data, sigma = window_size)
    else:
        raise ValueError(f"Unsupported smoothing type: {smooth_type}. Use 'moving_average' or 'gaussian'.")



