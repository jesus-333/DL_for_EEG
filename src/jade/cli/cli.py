"""
Entry points of the command-line interface (cli).

This module is the single place where the ``jade`` cli commands are declared. Every function here only builds its ``argparse`` parser and then hands the parsed arguments over to the matching implementation living in a thematic file of this subpackage (e.g. :mod:`jade.cli.download_dataset`).
By convention the entry-point function and its implementation share the same name.
The commands are wired to console scripts through the ``[project.scripts]`` table of ``pyproject.toml``.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Full module imports
import argparse

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# download-dataset-moabb

def download_dataset_moabb() -> None :
    """
    Entry point of the ``download-dataset-moabb`` command.

    It parses the command-line arguments and forwards them to :func:`jade.cli.download_dataset.download_dataset_moabb`, which downloads the requested moabb dataset through :func:`jade.dataset.moabb_dataset.download_dataset`.
    """

    from jade.cli.download_dataset import download_dataset_moabb

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Command-line arguments parsing

    parser = argparse.ArgumentParser(
        prog        = "download-dataset-moabb",
        description = "Download one of the datasets provided by moabb (see https://moabb.neurotechx.com/docs/dataset_summary.html).",
    )

    parser.add_argument(
        "--dataset_name",
        type     = str,
        required = True,
        help     = "Name of the dataset to download. It must be one of the datasets available through moabb (case-insensitive).",
    )
    parser.add_argument(
        "--subject",
        type    = int,
        default = None,
        help    = "The subject to download. If not specified, all the subjects will be downloaded. (int, default: None)",
    )
    parser.add_argument(
        "--dataset_folder",
        type    = str,
        default = None,
        help    = "Where to store the downloaded dataset. If not specified, the dataset is downloaded in the default moabb folder (~/mne_data/). "
                  "Note that moabb does not have an internal download registry, so if you change folder between different executions, only the last location will be used to check previously downloaded datasets. (str, default: None)",
    )

    args = parser.parse_args()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Run the command

    download_dataset_moabb(args)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# create-default-config

def default_config() -> None :
    """
    Entry point of the ``create-default-config`` command.

    It parses the command-line arguments and forwards them to :func:`jade.cli.default_config.default_config`, which writes a ``default_config`` folder of commented TOML skeleton configs (train / model / dataset) at the requested location.
    """

    from jade.cli.default_config import default_config

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Command-line arguments parsing

    parser = argparse.ArgumentParser(
        prog        = "create-default-config",
        description = "Create a 'default_config' folder with commented TOML skeleton configs that can be copied and edited.",
    )

    parser.add_argument(
        "--path",
        type    = str,
        default = ".",
        help    = "Directory where the 'default_config' folder is created. (str, default: current directory)",
    )
    parser.add_argument(
        "--type",
        type    = str,
        choices = ["train", "model", "dataset", "all"],
        default = "all",
        help    = "Which configs to create: 'train', 'model', 'dataset' or 'all'. (str, default: all)",
    )

    args = parser.parse_args()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Run the command

    default_config(args)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# download-project-runs-metrics

def download_project_runs_metrics() -> None :
    """
    Entry point of the ``download-project-runs-metrics`` command.

    It parses the command-line arguments and forwards them to :func:`jade.cli.wandb.download_project_runs_metrics`, which downloads the metrics of every run in a wandb project (through the wandb public API :class:`wandb.Api`) and saves them to a JSON file.
    """

    from jade.cli.wandb import download_project_runs_metrics

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Command-line arguments parsing

    parser = argparse.ArgumentParser(
        prog        = "download-project-runs-metrics",
        description = "Download the metrics of all the runs in a wandb project and save them to a JSON file.",
    )

    parser.add_argument(
        "--entity",
        type     = str,
        required = True,
        help     = "Name of the wandb entity (i.e. the user or team name) that owns the project. (str, required)",
    )
    parser.add_argument(
        "--project",
        type     = str,
        required = True,
        help     = "Name of the wandb project whose runs metrics are downloaded. (str, required)",
    )
    parser.add_argument(
        "--path_save",
        type    = str,
        default = None,
        help    = "Path of the JSON file where the metrics are saved. If not specified, the metrics are saved in the current working directory in the file wandb_metrics.json. (str, default: None)",
    )
    parser.add_argument(
        "--timeout",
        type    = int,
        default = 60,
        help    = "Timeout (in seconds) of the wandb public API GraphQL requests. Projects with many runs can exceed the wandb default (19 s) and raise a read timeout, so a larger value is used here. Increase it further if you still hit timeouts. (int, default: 60)",
    )

    args = parser.parse_args()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Run the command

    download_project_runs_metrics(args)
