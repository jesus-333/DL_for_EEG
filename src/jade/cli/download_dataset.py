"""
Implementation of the cli command used to download a moabb dataset.

This is the *implementation* side of the ``download-dataset-moabb`` command: it receives the already-parsed ``argparse`` arguments (the parsing itself lives in :func:`jade.cli.cli.download_dataset_moabb`) and forwards them to :func:`jade.dataset.moabb_dataset.download_dataset`.
No new behavior is added here; the command only exposes the existing dataset-module function on the command line.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Internal imports
from jade.dataset import download_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def download_dataset_moabb(args) -> None :
    """
    Download one of the datasets available through moabb. A complete list of datasets is available at ``https://moabb.neurotechx.com/docs/dataset_summary.html``.

    This is a thin wrapper around :func:`~jade.dataset.moabb_dataset.download_dataset`: its purpose is just to fetch the raw recordings ahead of time, so that you do not have to wait for the download later when you actually use the dataset.
    Note that moabb downloads the full raw recordings (every session), so there is no train/test split at download time.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments parsed in :func:`jade.cli.cli.download_dataset_moabb`. The expected fields are:

        - ``dataset_name`` : str. Name of the dataset to download. It must be one of the datasets available through moabb.
        - ``subject`` : int or None. The subject to download. If ``None`` all the subjects are downloaded.
        - ``dataset_folder`` : str or None. Where to store the downloaded dataset. If ``None`` the default moabb folder (``~/mne_data/``) is used.
    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Download the dataset

    # Forward the parsed arguments to the dataset-module function that does the actual work.
    # download_dataset accepts a single int (or None) for subjects, which matches the --subject argument.
    download_dataset(
        dataset_name   = args.dataset_name,
        subjects       = args.subject,
        dataset_folder = args.dataset_folder,
    )
