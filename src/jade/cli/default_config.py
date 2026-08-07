"""
Implementation of the cli command used to create the default configuration files.

This is the *implementation* side of the ``create-default-config`` command: it receives the already-parsed ``argparse`` arguments (the parsing itself lives in :func:`jade.cli.cli.default_config`) and copies the commented TOML skeleton configs shipped as package data.

The skeleton files live under ``jade/config/default_config`` (one folder per config type: ``train``, ``model``, ``dataset``); this command simply copies them to the requested location, so the TOML content is kept in the config package and not hardcoded in any Python file.
The produced configs are a *skeleton* (every possible parameter, with reasonable values and comments), meant to be edited, not used as-is.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Full module imports
import importlib.resources
import os
import shutil

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module constants

# Name of the folder that holds (and that is created with) the default configs.
DEFAULT_CONFIG_FOLDER_NAME = "default_config"

# Map the --type choice to the subfolder of the packaged skeleton tree ('all' handled separately).
TYPE_TO_SUBFOLDER = {
    "train"   : "train",
    "model"   : "model",
    "dataset" : "dataset",
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def default_config(args) -> None :
    """
    Create a ``default_config`` folder with the commented TOML skeleton configs.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments parsed in :func:`jade.cli.cli.default_config`. The expected fields are:

        - ``path`` : str. Directory where the ``default_config`` folder is created (the current directory if not given).
        - ``type`` : str. Which configs to create, one of ``'train'``, ``'model'``, ``'dataset'`` or ``'all'``.
    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Locate the packaged skeleton tree

    # The TOML files are shipped as package data inside jade/config/default_config.
    # importlib.resources resolves the folder for both an installed and an editable/source install.
    packaged_default_config = importlib.resources.files("jade.config").joinpath(DEFAULT_CONFIG_FOLDER_NAME)
    source_root = str(packaged_default_config)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Resolve the destination and copy

    # The whole output lives under <path>/default_config.
    destination_root = os.path.join(args.path, DEFAULT_CONFIG_FOLDER_NAME)

    if args.type == "all" :
        # Copy the whole skeleton tree (train + model + dataset).
        shutil.copytree(source_root, destination_root, dirs_exist_ok = True)
    else :
        # Copy only the subfolder matching the requested type.
        subfolder = TYPE_TO_SUBFOLDER[args.type]
        shutil.copytree(
            os.path.join(source_root, subfolder),
            os.path.join(destination_root, subfolder),
            dirs_exist_ok = True,
        )

    # A short, user-facing confirmation (this is a cli command, not library-internal code).
    print("Default config ({}) created at {}".format(args.type, os.path.abspath(destination_root)))
