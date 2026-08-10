"""
- Download a dataset inside the moabb colection
- Create an istance of the `EEG_Dataset` class (see the file `dataset_time.py` inside jade source code) 
- Visualize a random sample from the dataset

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

# Exteral librarires
import numpy as np
import matplotlib.pyplot as plt
import toml

# Internal impports
from jade.config import dataset as dataset_cfg
from jade.dataset import moabb_dataset, dataset_time

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

dataset_config_path = "./custom_config/dataset/dataset.toml"

# Generic dataset parameters
dataset_name = "BNCI2014_001"
subjects = [3, 5]
resample = False
resample_freq = None

# Filter parameters
# See the `filter_config` class in `src/jade/config/filter.py` for details on the filter parameters.
use_filter = False
filter_type = "bandpass"
l_freq = 8.0
h_freq = 32.0

# Plot settings
# See the `check_plot_config` function in `src/jade/dataset/dataset_time.py` for details on the plot parameters.
idx_ch = 'C3'
plot_config = dict(
    figsize      = (12, 8),
    fontsize     = 16,
    linewidth    = 1.3,
    color        = 'black',
    x_lim        = None,
    y_lim        = None,
    add_title    = True,
    title_prefix = None,
    save_plot    = False,
    plot_format  = ['png'],
    plot_path    = 'plot/',
    time_in_ms   = False,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load config and create config object

# Note you can create the dataset_config object in 3 way :
# 1) Using the from_dict method. In this case you have to pass a dictionary with the config parameters
# 2) Using the from_toml method. In this case you have to pass the path to the toml file
# 3) Using directly the "constructor" of the dataset_config class. In this case you have to pass the parameters directly. E.g. dataset_config = dataset_cfg.dataset_config(dataset_name=dataset_name, subjects=subjects_list ...)

# Here we will use the first method, but you can choose the one you prefer.

# Load the config from the toml file
dataset_config_dict = toml.load(dataset_config_path)

# Update the config dictionary with the parameters we want to change
if dataset_name is not None  : dataset_config_dict['dataset_name']  = dataset_name
if subjects is not None      : dataset_config_dict['subjects']      = subjects
if resample is not None      : dataset_config_dict['resample']      = resample
if resample_freq is not None : dataset_config_dict['resample_freq'] = resample

# (OPTIONAL) Update the filter parameters if we want to use a custom filter
if use_filter :
    filter_dict = dict(
        filter_type = filter_type,
        l_freq      = l_freq,
        h_freq      = h_freq
    )
else : 
    filter_dict = None
dataset_config_dict['filter'] = filter_dict

# Create the dataset_config object from the dictionary
dataset_config = dataset_cfg.dataset_config.from_dict(dataset_config_dict)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Download the dataset and create an instance of the EEG_Dataset class

# Note that the moabb_dataset module include two functions to "handle" dataset creation
# 1) `download_dataset()` that required only the dataset name and (optionally) subjects list and download folder.
#    This function will download the dataset from moabb and return it as a moabb object.
# 2) `get_dataset()` that required a dataset_config object and a dataset_type (train or test, default to train if not specified).
#    This function will download the dataset (if not already downloaded), using internally `download_dataset()` and then directly converting from the moabb object to an instance of the EEG_Dataset class. 
#    This function will return an instance of the EEG_Dataset class.

# Note that to apply preprocessing you need to use the latter function, because the preprocessing parameters are included in the dataset_config object.

eeg_dataset_train = moabb_dataset.get_dataset(dataset_config, dataset_type = 'train')
eeg_dataset_test  = moabb_dataset.get_dataset(dataset_config, dataset_type = 'test')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot a random sample from the dataset

def plot_random_sample(eeg_dataset : dataset_time.EEG_Dataset, plot_type : str, idx_ch : str, plot_config : dict = None) :
    """
    Plot a random sample from the dataset.

    Parameters
    ----------
    eeg_dataset : EEG_Dataset
        An instance of the EEG_Dataset class.
    """

    # Get a random sample from the dataset
    random_idx = np.random.randint(0, len(eeg_dataset))
    
    # Create plot and return it
    if plot_type == 'time' :
        fig, ax = eeg_dataset.plot_trial_time(random_idx, idx_ch, plot_config)
    elif plot_type == 'freq' :
        fig, ax = eeg_dataset.plot_trial_freq(random_idx, idx_ch, plot_config)
    elif plot_type == 'both' :
        fig, ax = eeg_dataset.plot_trial_time_and_freq(random_idx, idx_ch, plot_config)
    return fig, ax

fig_time, ax_time = plot_random_sample(eeg_dataset_train, 'time', idx_ch, plot_config = plot_config)
fig_freq, ax_freq = plot_random_sample(eeg_dataset_train, 'freq', idx_ch, plot_config = plot_config)
fig_both, ax_both = plot_random_sample(eeg_dataset_train, 'both', idx_ch, plot_config = plot_config)

plt.show()


