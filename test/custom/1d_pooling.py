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
from jade.config import pooling_config
from jade.model import pooling_1d, support_functions

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

folder_paths_pooling_config = "./custom_config/model/pooling/"

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

# Get all the pooling config files in the folder
list_paths_pooling_config = [os.path.join(folder_paths_pooling_config, f) for f in os.listdir(folder_paths_pooling_config) if f.endswith(".toml")]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

for i in range(len(list_paths_pooling_config)) :
    # Load config from file
    pooling_config_dict = toml.load(list_paths_pooling_config[i])

    # Create pooling config object
    pooling_config_obj = pooling_config.from_dict(pooling_config_dict)

    pooling_layer = support_functions.get_multichannel_pooling(pooling_config_obj)




