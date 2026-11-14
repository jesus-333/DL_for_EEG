"""
Plot the aliased energy ratio for a single experiment.

To run this script you need to run the script 2_computed_metrics_experiment.py first, whith `save_single_tensor_all_repetitions` set to True, so that the metric with all the data of all repetitions is saved to disk.

Note that the plot would be the same of `3_plot_aliased_energy_ratio_repetition.py` but average across repetitions.

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

import other.aliasing_experiment.z_support_analysis as support_analysis

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

experiment_name = "d2a_subj_2/"

path_results = f'./other/aliasing_experiment/results/{experiment_name}/'
path_results_plots = os.path.join(path_results, "plots/")

sort_depth_channels_by_energy_ratio = True
sort_depth_channels_by_energy_ratio_block = 1 # Choose which block to use for sorting the depth channels. 1 or 2.

plot_config = dict(
    # Common settings for all plots
    show_plot = True,
    save_plot = True,
    epochs = 500, # Update based on the experiment
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
    add_std = False,
    show_legend = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Read data

# Read the aliased energy ratio for both blocks
# This array will have shape (R, epochs, N, D), where R is the numbero of repetition, epochs is the number of training epochs, N is the number of samples and C is the number of channels
aliased_energy_ratio_block_1_train, aliased_energy_ratio_block_1_test = support_analysis.read_aliased_energy_ratio_experiment(path_results, block = 1)
aliased_energy_ratio_block_2_train, aliased_energy_ratio_block_2_test = support_analysis.read_aliased_energy_ratio_experiment(path_results, block = 2)

if sort_depth_channels_by_energy_ratio :
    for i in range(aliased_energy_ratio_block_1_train.shape[0]) :

        # Get the indices that would sort the depth channels based on the average aliased energy on last epoch
        if sort_depth_channels_by_energy_ratio_block == 1 :
            idx_sorted_block = support_analysis.get_idx_sorting_depth_channels_by_energy_ratio(aliased_energy_ratio_block_1_train[i])
        elif sort_depth_channels_by_energy_ratio_block == 2 :
            idx_sorted_block = support_analysis.get_idx_sorting_depth_channels_by_energy_ratio(aliased_energy_ratio_block_2_train[i])
        else :
            raise ValueError(f"Block {sort_depth_channels_by_energy_ratio_block} is not supported. Only block 1 and 2 are supported.")

        # Sort the depth channels based on the indices
        aliased_energy_ratio_block_1_train[i] = aliased_energy_ratio_block_1_train[i][:, :, idx_sorted_block]
        aliased_energy_ratio_block_1_test[i] = aliased_energy_ratio_block_1_test[i][:, :, idx_sorted_block]
        aliased_energy_ratio_block_2_train[i] = aliased_energy_ratio_block_2_train[i][:, :, idx_sorted_block]
        aliased_energy_ratio_block_2_test[i] = aliased_energy_ratio_block_2_test[i][:, :, idx_sorted_block]

        # Note that the same indices are used to sort both blocks and both train and test sets.
        # The rationale to use them in both block is to analyze if channels that have a lower aliased energy ratio in block 1 also have a lower aliased energy ratio in block 2, and vice versa.
        # For the train/test set the rationale follow a simialr philosophy, i.e. if a channel has a lower aliased energy ratio in the train set, it should also have a lower aliased energy ratio in the test set. Basically, we check if the model generalize well across the train/test set.

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot 1 : Average aliased energy ratio per sample vs epoch

def create_plot_1(aliased_energy_ratio_array, plot_config, title = None) :

    # Average across depth channels (D) and get an array of shape (epochs, N)
    array_to_plot = np.mean(aliased_energy_ratio_array, axis = (0, 3))  # Average across repetitions and depth channels
    
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

fig_1_block_1_train, ax_1_block_1_train = create_plot_1(aliased_energy_ratio_block_1_train, plot_config, title = "Aliased Energy Ratio for Block 1 (TRAIN)")
fig_1_block_2_train, ax_1_block_2_train = create_plot_1(aliased_energy_ratio_block_2_train, plot_config, title = "Aliased Energy Ratio for Block 2 (TRAIN)")
fig_1_block_1_test, ax_1_block_1_test = create_plot_1(aliased_energy_ratio_block_1_test, plot_config, title = "Aliased Energy Ratio for Block 1 (TEST)")
fig_1_block_2_test, ax_1_block_2_test = create_plot_1(aliased_energy_ratio_block_2_test, plot_config, title = "Aliased Energy Ratio for Block 2 (TEST)")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot 2 : Average aliased energy ratio per depth channel vs epoch

def create_plot_2(aliased_energy_ratio_array, plot_config, title = None) :
    
    # Average across samples (N) and get an array of shape (epochs, D)
    array_to_plot = np.mean(aliased_energy_ratio_array, axis = (0, 1)) # Average across repetitions and samples

    # (OPTIONAL) Computed the standard deviation across samples (N) and get an array of shape (epochs, D)
    if plot_config['add_std'] : array_std_to_plot = np.std(aliased_energy_ratio_array, axis = (0, 1))
    
    # Create the plot
    fig, ax = plt.subplots(figsize = plot_config['figsize_2'])

    # Since the number of depth channels is usually small, we can plot it as a line plot
    for d in range(array_to_plot.shape[1]) :
        data_to_plot = array_to_plot[:, d]

        if plot_config['add_std'] : std_to_plot = array_std_to_plot[:, d]
        
        if plot_config['smoothing_method'] is not None :
            data_to_plot = support_analysis.smooth_time_series(data_to_plot, smooth_type = plot_config['smoothing_method'], window_size = plot_config['smoothing_window'])
            epochs_to_plot = np.linspace(0, plot_config['epochs'], len(data_to_plot))
            if plot_config['add_std'] : std_to_plot = support_analysis.smooth_time_series(std_to_plot, smooth_type = plot_config['smoothing_method'], window_size = plot_config['smoothing_window'])
        else :
            epochs_to_plot = np.arange(plot_config['epochs'])

        ax.plot(epochs_to_plot, data_to_plot, label = f'Depth Channel {d + 1}')
        if plot_config['add_std'] : ax.fill_between(epochs_to_plot, data_to_plot - std_to_plot, data_to_plot + std_to_plot, alpha = 0.2)

    # Beautify the plot
    if plot_config['show_legend'] : ax.legend()
    ax.grid(True)
    ax.set_xlabel('Epochs')
    ax.set_xlim(1, plot_config['epochs'])
    ax.set_ylabel('Average Aliased Energy Ratio')
    if title is not None : ax.set_title(title)

    fig.tight_layout()

    return fig, ax

fig_2_block_1_train, ax_2_block_1_train = create_plot_2(aliased_energy_ratio_block_1_train, plot_config, title = "Average Aliased Energy Ratio per Depth Channel for Block 1 (TRAIN)")
fig_2_block_2_train, ax_2_block_2_train = create_plot_2(aliased_energy_ratio_block_2_train, plot_config, title = "Average Aliased Energy Ratio per Depth Channel for Block 2 (TRAIN)")
fig_2_block_1_test, ax_2_block_1_test = create_plot_2(aliased_energy_ratio_block_1_test, plot_config, title = "Average Aliased Energy Ratio per Depth Channel for Block 1 (TEST)")
fig_2_block_2_test, ax_2_block_2_test = create_plot_2(aliased_energy_ratio_block_2_test, plot_config, title = "Average Aliased Energy Ratio per Depth Channel for Block 2 (TEST)")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# (OPTIONAL) Show the plots
if plot_config['show_plot'] :
    plt.show()












