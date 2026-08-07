"""
Implementation of the script used to download moabb dataset

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

from __future__ import annotations

import os

from jade.dataset import moabb_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def download_dataset_moabb(args) -> None:
    """
    Download one of the dataset available through moabb. A complete list of dataset is available at `https://moabb.neurotechx.com/docs/dataset_summary.html`
    Note that the purpose of this function is just to download it so you avoid to wait later when you will use it.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments parsed by the argparse library. The expected arguments are:
        - dataset_name : name of the dataset to download. It must be one of the dataset available through moabb.
        - subject : int. The subject to download. If not specified, all the subjects will be downloaded.
        - dataset_type : str. The type of the dataset to download, i.e. train or test. If not specified, both will be downloaded.
        - dataset_folder : str, optional. Specify where store the downloaded dataset. If not specified, the dataset will be downloaded in default moabb folder (`~/mne_data/`). Note that moabb does not have an internal download registry, so if you change folder between different executions, only the last location will be used to check previously downloaded datasets.
    """
    # ***************************************
    # Check the arguments

    # ***************************************
    # Conversion
