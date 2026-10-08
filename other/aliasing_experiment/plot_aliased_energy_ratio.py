"""
Plot the aliased energy ratio for a single experiment.

-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import matplotlib.pyplot as plt
import numpy as np
import os
import toml
import torch

import other.aliasing_experiment.support_analysis as support_analysis_plot

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

experiment_name = "subj_3/15634437420104226787/"

path_results = f'./other/aliasing_experiment/results/{experiment_name}/'
path_results_metrics = os.path.join(path_results, "metrics/")
path_results_config = os.path.join(path_results, "config_backup/")
path_results_plots = os.path.join(path_results, "plots/")

plot_config = dict(
    # Common settings for all plots
    show_plot = True,
    save_plot = True,
    # Plot 1 settings
    figsize_1 = (14, 8),
    cmap = 'viridis',
    vmin = 0.0,
    vmax = 1.0,
    # vmin = None,
    # vmax = None,
    # Plot 2 settings
    figsize_2 = (14, 8),
    smoothing_window = 50,
    smoothing_method = 'moving_average',  # Options: 'moving_average', 'gaussian', None
    show_legend = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Read data

training_config = toml.load(os.path.join(path_results_config, "training_config.toml"))

def read_aliased_energy_ratio(path_results_metrics, block : int, epochs : int) -> np.ndarray:
    """
    Read the aliased energy ratio from the metrics file for a given block.

    Parameters
    ----------
    path_results_metrics : str
        Path to the metrics folder.
    block : int
        EEGNet "block". 1 give the energy ratio for for the first pooling layer, 2 for the second pooling layer.
    epochs : int
        Number of epochs to read.

    Returns
    -------
    np.ndarray
        Array of aliased energy ratios for the specified block across epochs. The final shape is (epochs, N, D), where
        - epochs is the number of epochs (clearly),
        - N is the number of samples in the dataset
        - D is the number of depth channels in the EEGNet model for that layer (n.b. this should not be confused with the number of EEG channels).
    """

    if block != 1 and block != 2: raise ValueError(f"Block {block} is not supported. Only block 1 and 2 are supported.")
    if epochs < 0: raise ValueError(f"Epoch {epochs} is not supported. Only positive epochs are supported.")

    tmp_list_energy_ratio = []

    for i in range(epochs):
        file_path = os.path.join(path_results_metrics, f"aliased_energy_ratio_{block}_epoch_{i + 1}.pt")
        if not os.path.isfile(file_path) :
            raise FileNotFoundError(f"Metrics file for block {block} and epoch {i} not found at {file_path}.")
        
        tmp_energy_ratio = torch.load(file_path)
        tmp_list_energy_ratio.append(tmp_energy_ratio.squeeze().cpu().detach().numpy())

    # Convert the list to a numpy array for easier manipulation
    return np.array(tmp_list_energy_ratio)

# Read the aliased energy ratio for both blocks
# This array will have shape (epochs, N, D), where N is the number of samples and C is the number of channels
aliased_energy_ratio_block_1 = read_aliased_energy_ratio(path_results_metrics, block = 1, epochs = training_config['epochs'])
aliased_energy_ratio_block_2 = read_aliased_energy_ratio(path_results_metrics, block = 2, epochs = training_config['epochs'])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot 1 : Average aliased energy ratio per sample vs epoch

def create_plot_1(aliased_energy_ratio_array, plot_config, title = None) :

    # Average across depth channels (D) and get an array of shape (epochs, N)
    array_to_plot = np.mean(aliased_energy_ratio_array, axis = 2)
    
    # Create the plot
    fig, ax = plt.subplots(figsize = plot_config['figsize_1'])

    # Due to the big size it will plot as an image
    cax = ax.imshow(array_to_plot.T, aspect = 'auto',
                    cmap = plot_config['cmap'],
                    vmin = plot_config['vmin'], vmax = plot_config['vmax']
                    )

    # Add colorbar
    cbar = fig.colorbar(cax)
    cbar.set_label('Aliased Energy Ratio', rotation = 270, labelpad = 10)
    
    # Beautify the plot
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Samples')
    if title is not None : ax.set_title(title)

    fig.tight_layout()

    return fig, ax

fig_1_block_1, ax_1_block_1 = create_plot_1(aliased_energy_ratio_block_1, plot_config, title = "Aliased Energy Ratio for Block 1")
fig_1_block_2, ax_1_block_2 = create_plot_1(aliased_energy_ratio_block_2, plot_config, title = "Aliased Energy Ratio for Block 2")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot 2 : Average aliased energy ratio per depth channel vs epoch

plot_config['epochs'] = training_config['epochs']

def create_plot_2(aliased_energy_ratio_array, plot_config, title = None) :
    
    # Average across samples (N) and get an array of shape (epochs, D)
    array_to_plot = np.mean(aliased_energy_ratio_array, axis = 1)
    
    # Create the plot
    fig, ax = plt.subplots(figsize = plot_config['figsize_2'])

    # Since the number of depth channels is usually small, we can plot it as a line plot
    for d in range(array_to_plot.shape[1]) :
        data_to_plot = array_to_plot[:, d]
        
        if plot_config['smoothing_method'] is not None :
            data_to_plot = support_analysis_plot.smooth_time_series(data_to_plot, smooth_type = plot_config['smoothing_method'], window_size = plot_config['smoothing_window'])
            epochs_to_plot = np.linspace(0, plot_config['epochs'], len(data_to_plot))
        else :
            epochs_to_plot = np.arange(plot_config['epochs'])

        ax.plot(epochs_to_plot, data_to_plot, label = f'Depth Channel {d + 1}')

    # Beautify the plot
    if plot_config['show_legend'] : ax.legend()
    ax.grid(True)
    ax.set_xlabel('Epochs')
    ax.set_xlim(1, plot_config['epochs'])
    ax.set_ylabel('Average Aliased Energy Ratio')
    if title is not None : ax.set_title(title)

    fig.tight_layout()

    return fig, ax


fig_2_block_1, ax_2_block_1 = create_plot_2(aliased_energy_ratio_block_1, plot_config, title = "Average Aliased Energy Ratio per Depth Channel for Block 1")
fig_2_block_2, ax_2_block_2 = create_plot_2(aliased_energy_ratio_block_2, plot_config, title = "Average Aliased Energy Ratio per Depth Channel for Block 2")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# (OPTIONAL) Show the plots
if plot_config['show_plot'] :
    plt.show()












