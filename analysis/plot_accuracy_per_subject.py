"""

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import json
import matplotlib.pyplot as plt
import numpy as np
import os

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

# Available files
# d2a
# HGD
# Ofner2017
# Tavakolan2017
# Wu2020
dataset_name = "Tavakolan2017"
path_to_metrics_file = f"./analysis/metrics_paper/EEGNet_pooling_{dataset_name}.json"

# metrics_to_plot_list = ['accuracy_validation']
metric_to_plot = 'accuracy_validation'
pooling_types_to_plot_list = ['default', 'blur', 'dwt', 'sinc', 'max']
pooling_types_to_plot_list = ['default', 'blur', 'dwt', 'sinc', ]

# If used must be a positivi integer. It keeps the training run with only this specific number of epochs
filter_length = 500

# Smoothing settings. If smoothing is applied, the metric will be smoothed with a moving average of this length.
# To disable smoothing, set it to 1 or less
window_length = 10

# Note that one plot per subject will be generated. The plots will be saved in the same folder as the metrics file
plot_settings = dict(
    figsize = (12, 8),
    fontsize = 14,
    add_std = False,
    add_title = True,
    save_plot = True,
    path_to_save_plot = f"./analysis/metrics_paper/plots/{dataset_name}/",
    format_list = ['png'],
    show_plot = False
)

pooling_type_colors = dict(
    default = 'red',
    blur    = 'orange',
    dwt     = 'green',
    sinc    = 'blue',
    max     = 'purple'
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load data and prepare the structure to store it

n_subj_per_dataset = dict(
    d2a = 9,
    HGD = 14,
    Ofner2017 = 15,
    Tavakolan2017 = 12,
    Wu2020 = 6,
)

# Load data
data_ALL = json.load(open(path_to_metrics_file, 'r'))

# Create the structure to store the data for each subject and pooling type
# It is a dictionary where the key are the subject names and the value are other dictionaries.
# The inner dictionaries have as key the pooling type and as value a list with the metric for each training run executed with that pooling type
data_per_subj = {}
data_to_plot_mean, data_to_plot_std = {}, {}
for i in range(n_subj_per_dataset[dataset_name]) :
    # Create the subject name
    subj_string = f"S{i + 1}"
    
    # Create the inner dictionaries for that subject
    data_per_subj[subj_string]      = {}
    data_to_plot_mean[subj_string]  = {}
    data_to_plot_std[subj_string]   = {}

    # Create the inner dictionaries for each pooling type
    for pooling_type in pooling_types_to_plot_list :
        data_per_subj[subj_string][pooling_type]     = []
        data_to_plot_mean[subj_string][pooling_type] = []
        data_to_plot_std[subj_string][pooling_type]  = []

# Get all data for that metric
# This is another dictionary where the key are the name of the training runs and the value are list with the metric, logged at each epoch
metric_data = data_ALL[metric_to_plot]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save the data for each subject and pooling type

# Iterate over the training run
for training_run_name in metric_data :
    # Get run information
    tmp_name_split = training_run_name.split('_')
    subject = tmp_name_split[0]
    pooling_type = tmp_name_split[2]
    
    # Save data only if the training run was executed with the specific pooling type
    if pooling_type in pooling_types_to_plot_list :
        # Get the metric for the training run
        metric_vs_epoch = metric_data[training_run_name]

        # (OPTIONAL) Exclude the training run with a length different from the one specified
        if len(metric_vs_epoch) != filter_length and filter_length > 0 :
            print(f"Skip run {training_run_name}. Length is {len(metric_vs_epoch)}. Required length is {filter_length}")
            continue
        
        # Save the data
        data_per_subj[subject][pooling_type].append(metric_vs_epoch)

# For each subject and pooling type, compute the mean and std of the metric across all training runs
for subject in data_per_subj :
    for pooling_type in data_per_subj[subject] :
        # Get the data for that subject and pooling type
        data = data_per_subj[subject][pooling_type]

        # (OPTIONAL) Smooth the data with a moving average
        if window_length > 1 :
            data = [np.convolve(metric, np.ones(window_length) / window_length, mode = 'valid') for metric in data]
        
        # Compute the mean and std across all training runs
        if len(data) > 0 :
            data_to_plot_mean[subject][pooling_type] = np.mean(data, axis=0)
            data_to_plot_std[subject][pooling_type]  = np.std(data, axis=0)
        else :
            print(f"Warning: No data for subject {subject} and pooling type {pooling_type}.")
        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plotting

for subject in data_to_plot_mean :
    # Create the figure
    fig, ax = plt.subplots(figsize = plot_settings['figsize'])

    # Plot the data for each pooling type
    for pooling_type in pooling_types_to_plot_list :
        # Get the data for that subject and pooling type
        mean_data = data_to_plot_mean[subject][pooling_type]
        std_data  = data_to_plot_std[subject][pooling_type]
        
        # Get the number of epochs for that subject and pooling type
        n_epochs = np.arange(1, len(mean_data) + 1)

        # Skip if there is no data to plot
        if len(mean_data) == 0 :
            print(f"Warning: No data to plot for subject {subject} and pooling type {pooling_type}.")
            continue

        # Plot the mean data
        ax.plot(n_epochs, mean_data, label = pooling_type, color = pooling_type_colors[pooling_type])

        # (OPTIONAL) add the std as a shaded area
        if plot_settings['add_std'] :
            ax.fill_between(n_epochs, mean_data - std_data, mean_data + std_data,
                            alpha = 0.2, color = pooling_type_colors[pooling_type]
                            )

        # Beautify the plot
        if plot_settings['add_title'] : ax.set_title(f"{metric_to_plot} for subject {subject}", fontsize = plot_settings['fontsize'])
        ax.set_xlabel("Epochs", fontsize = plot_settings['fontsize'])
        ax.set_ylabel(metric_to_plot, fontsize = plot_settings['fontsize'])
        ax.legend(fontsize = plot_settings['fontsize'])
        ax.grid(True)
        ax.set_xlim([n_epochs[0], n_epochs[-1]])
        fig.tight_layout()

        # (OPTIONAL) Save the plot
        if plot_settings['save_plot'] :
            # Create the folder if it does not exist
            os.makedirs(plot_settings['path_to_save_plot'], exist_ok = True)

            # Save the plot in all formats specified
            for format in plot_settings['format_list'] :
                path_to_save_plot = os.path.join(plot_settings['path_to_save_plot'], f"{subject}_{metric_to_plot}.{format}")
                fig.savefig(path_to_save_plot, format = format)
                print(f"Saved plot for subject {subject} and metric {metric_to_plot} in {path_to_save_plot}")

        # (OPTIONAL) Show the plot
        if plot_settings['show_plot'] : fig.show()
        else : plt.close()
