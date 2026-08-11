"""
- Download one of the dataset inside moabb
- Apply the various pooling to a random sample
- Plot the sample before and after pooling, both in frequency and time domain, to show the pooling effect

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

# External libraries
import matplotlib.pyplot as plt
import numpy as np
import toml
import torch
import os

# Internal libraries
from jade.dataset import moabb_dataset, support_plot_freq
from jade.config import pooling_config, dataset_config
from jade.model import pooling_1d, support_functions

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Paths
folder_paths_pooling_config = "./custom_config/model/pooling/"
dataset_config_path = "./custom_config/dataset/dataset.toml"

# Pooling parameters
decimation_factor = 4

# Generic dataset parameters
dataset_name = "BNCI2014_001"
subjects = [3, 5]
resample = False
resample_freq = None

idx_ch = 'C3'
plot_config = dict(
    figsize      = (12, 8),
    fontsize     = 16,
    linewidth    = 1.3,
    x_lim        = None, # Set to None to automatically adjust based on the data
    y_lim        = None,
    add_title    = True,
    title_prefix = None,
    save_plot    = False,
    plot_format  = ['png'],
    plot_path    = 'plot/',
    time_in_ms   = False,
    use_log_scale_freq = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dataset

# Get config from toml file
dataset_config_dict = toml.load(dataset_config_path)

if dataset_name is not None  : dataset_config_dict['dataset_name']  = dataset_name
if subjects is not None      : dataset_config_dict['subjects']      = subjects
if resample is not None      : dataset_config_dict['resample']      = resample
if resample_freq is not None : dataset_config_dict['resample_freq'] = resample

# No filtering applied
dataset_config_dict['filter'] = None

# Create the dataset_config object from the dictionary
dataset_config = dataset_config.from_dict(dataset_config_dict)

# Get dataset
eeg_dataset = moabb_dataset.get_dataset(dataset_config, dataset_type = 'train')

# Get random sample
idx_random = np.random.randint(0, len(eeg_dataset))
x_eeg, _ = eeg_dataset[idx_random]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if torch.cuda.is_available() :
    device = "cuda"
    print("\nCUDA backend in use")
elif torch.backends.mps.is_available():
    device = "mps"
    print("\nmps backend (apple metal) in use")
else:
    device = "cpu"
    print("\nNo backend in use. Device set to cpu")

# Get all the pooling config files in the folder
list_paths_pooling_config = [os.path.join(folder_paths_pooling_config, f) for f in os.listdir(folder_paths_pooling_config) if f.endswith(".toml")]
list_pooling_layer_names  = [os.path.splitext(os.path.basename(f))[0] for f in list_paths_pooling_config]

list_x_eeg_after_pooling = {} # Yeah I know that is not a list

for i in range(len(list_paths_pooling_config)) :
    # Load config from file
    pooling_config_dict = toml.load(list_paths_pooling_config[i])

    # Apply custom config
    pooling_config_dict['decimation_factor'] = decimation_factor

    # Create pooling config object
    pooling_config_obj = pooling_config.from_dict(pooling_config_dict)

    # Createa pooling layer
    pooling_layer = support_functions.get_multichannel_pooling(pooling_config_obj).to(device)
    
    # Apply pooling to the sample
    x_eeg_after_pooling = pooling_layer(x_eeg.unsqueeze(0).to(device)) # Add batch dimension

    # Saved the result
    list_x_eeg_after_pooling[list_pooling_layer_names[i]] = x_eeg_after_pooling.squeeze().cpu().numpy() # Remove batch dimension and move to cpu

# Compute AvgPooling2d with kernel size equal to decimation factor 
# pooling_layer_avg = torch.nn.AvgPool2d(kernel_size = (1, decimation_factor), stride = (1, decimation_factor), padding = 0)
# x_eeg_after_pooling_avg = pooling_layer_avg(x_eeg.unsqueeze(0).to(device)) # Add batch dimension
# list_x_eeg_after_pooling['AvgPool_TORCH'] = x_eeg_after_pooling_avg.squeeze().cpu().numpy() # Remove batch dimension and move
# list_pooling_layer_names.append('AvgPool_TORCH') # Add the AvgPool_TORCH to the list of pooling layer names

# Remove extra dimension from the original signal (i.e. get a signal of shape [n_channels, n_samples])
x_eeg = x_eeg.squeeze().cpu().numpy()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot (TIME domain)

# Create the figure and axis
fig_time, ax_time = plt.subplots(1, 1, figsize = plot_config['figsize'])

# Get time vector and x_lim for the plot
if dataset_name == "BNCI2014_001" :
    # The dataset is already cropped to 4 seconds
    t = np.linspace(2, 6, x_eeg_after_pooling.shape[-1])
else :
    t = np.arange(x_eeg.shape[-1]) / (eeg_dataset.fs / decimation_factor)

# Get idx channel
if type(idx_ch) is str : idx_ch = eeg_dataset.ch_list.index(idx_ch)

for i in range(len(list_pooling_layer_names)) :
    # Get the pooling layer name
    pooling_layer_name = list_pooling_layer_names[i]

    # Get the signal to plot
    x_eeg_to_plot = list_x_eeg_after_pooling[pooling_layer_name][idx_ch, :]
    
    # Plot signal
    ax_time.plot(t, x_eeg_to_plot, label = pooling_layer_name, linewidth = plot_config['linewidth'])

plot_config['x_lim'] = [t[0], t[-1]]  # Update x_lim for the plot based on the time vector

# Add original signal to the plot
# t = np.linspace(t[0], t[-1], x_eeg.shape[-1]) # Update time vector to match the original signal length
# ax_time.plot(t, x_eeg[idx_ch, :], label = "Original", linewidth = plot_config['linewidth'], color = 'black', linestyle = '--')

# Beautify the plot
ax_time.set_xlabel("Time [s]", fontsize = plot_config['fontsize'])
ax_time.set_xlim(plot_config['x_lim'])
ax_time.set_ylabel(r"Amplitude [$\mu V$]", fontsize = plot_config['fontsize'])
ax_time.grid(True)
ax_time.legend(fontsize = plot_config['fontsize'])

fig_time.tight_layout()
fig_time.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot (FREQUENCY domain)

# Create the figure and axis
fig_freq, ax_freq = plt.subplots(1, 1, figsize = plot_config['figsize'])

for i in range(len(list_pooling_layer_names)) :
    # Get the pooling layer name
    pooling_layer_name = list_pooling_layer_names[i]
    
    # Get the signal to plot
    x_eeg_to_plot = list_x_eeg_after_pooling[pooling_layer_name][idx_ch, :]

    # Convert signals to frequency domain
    freqs, psd = support_plot_freq.compute_frequency_spectrum(x_eeg_to_plot, eeg_dataset.fs / decimation_factor)

    # Plot signal
    ax_freq.plot(freqs, psd, label = pooling_layer_name, linewidth = plot_config['linewidth'])


# Update x_lim for the plot based on the frequency vector
plot_config['x_lim'] = [freqs[0], freqs[-1]]

# Add original signal to the plot
freqs, psd = support_plot_freq.compute_frequency_spectrum(x_eeg[idx_ch, :], eeg_dataset.fs)
ax_freq.plot(freqs, psd, label = "Original", linewidth = plot_config['linewidth'], color = 'black', linestyle = '--')

# Beautify the plot
ax_freq.set_xlabel("Frequency [Hz]", fontsize = plot_config['fontsize'])
ax_freq.set_ylabel(r"Power Spectral Density [$\mu V^2$/Hz]", fontsize = plot_config['fontsize'])
ax_freq.set_xlim(plot_config['x_lim'])
ax_freq.grid(True)
ax_freq.legend(fontsize = plot_config['fontsize'])
if plot_config['use_log_scale_freq'] : ax_freq.set_yscale('log')

fig_freq.tight_layout()
fig_freq.show()
