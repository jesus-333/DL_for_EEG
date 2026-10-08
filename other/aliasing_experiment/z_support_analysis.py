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



