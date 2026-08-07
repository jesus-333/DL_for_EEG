"""
Implementation the command-line interface (cli)

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

import argparse

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def download_dataset_moabb() :
    """
    Download a dataset from moabb
    """
    
    from jade.cli.download_daset import download_dataset_moabb

    # ***************************************
    # Command-line arguments parsing
    
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(prog = "nii-to-hdf5", description = "Convert all nii files in a folder to a single hdf5 files.")
    parser.add_argument("--dataset_name", required = True, help = "Name of the dataset to download. It must be one of the dataset available through moabb.")
    parser.add_argument("--subject", type = int, default = None, help = "The subject to download. If not specified, all the subjects will be downloaded.")
    parser.add_argument("--dataset_type", default = None, help = "The type of the dataset to download, i.e train or test. If not specified, both will be downloaded.")
    parser.add_argument("--dataset_folder", default = None, help = "Specify where store the downloaded dataset. If not specified, the dataset will be downloaded in default moabb folder (`~/mne_data/`). Note that moabb does not have an internal download registry, so if you change folder between different executions, only the last location will be used to check previously downloaded datasets.")
    args = parser.parse_args()

    # ***************************************

    # Call the function that implements the cli command
    download_dataset_moabb(args)
