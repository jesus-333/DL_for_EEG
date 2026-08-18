"""
Implementation of the cli commands that interact with wandb.

This is the *implementation* side of the wandb-related cli commands: it receives the already-parsed ``argparse`` arguments (the parsing itself lives in :mod:`jade.cli.cli`) and talks to the wandb public API (:class:`wandb.Api`).

Unlike the other thematic cli files, the small helpers this command needs (reading a run history and stripping the model parameters/gradients that :func:`~jade.training.classifier.wandb_train_model` logs through ``wandb.watch``) do not yet live elsewhere in the package, so they are reconstructed here as module-level helpers.
If more wandb functionality is added later they can be promoted to a dedicated ``jade`` module and the cli reduced back to a thin wrapper.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Full module imports
import json

# Third party imports (wandb is an optional extra of the package; pandas is a core dependency).
import pandas as pd
import wandb

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Helper functions

def get_run_information(run) -> dict :
    """
    Collect from a single wandb run the information needed to export its metrics.

    The run history is downloaded with ``run.scan_history()`` (the full, unsampled history: every logged step) and materialized into a :class:`pandas.DataFrame`.
    Since different metrics can be logged at different steps, the resulting DataFrame is sparse (missing values become ``NaN``), which is handled downstream in :func:`download_project_runs_metrics`.

    Parameters
    ----------
    run : wandb.apis.public.Run
        A run object returned by :meth:`wandb.Api.runs`.

    Returns
    -------
    run_information : dict
        A dictionary with the fields:

        - ``name`` : str. The human-readable name of the run.
        - ``id`` : str. The unique id of the run.
        - ``history`` : pandas.DataFrame. The full run history (one column per logged key, one row per logged step).
    """

    # Materialize the full (unsampled) history into a DataFrame. scan_history yields one dict per logged step, so pandas builds the union of all keys, filling the gaps with NaN.
    history = pd.DataFrame(run.scan_history())

    return {
        "name"    : run.name,
        "id"      : run.id,
        "history" : history,
    }

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def remove_parameters_from_history(run_history : pd.DataFrame) -> pd.DataFrame :
    """
    Remove from a run history the columns logged by ``wandb.watch`` (the model parameters and gradients).

    When a run is tracked with ``wandb.watch(model, log = "all", ...)`` (as :func:`~jade.training.classifier.wandb_train_model` does), wandb logs the per-layer parameters and gradients under the ``parameters/`` and ``gradients/`` column prefixes.
    Those are model internals, not training metrics, so they are dropped here before the metrics are exported.

    Parameters
    ----------
    run_history : pandas.DataFrame
        The run history as returned by :func:`get_run_information` (the ``history`` field).

    Returns
    -------
    metrics_history : pandas.DataFrame
        The same history without the ``parameters/`` and ``gradients/`` columns.
    """

    # Collect the columns that correspond to the model parameters/gradients logged by wandb.watch.
    columns_to_drop = [column for column in run_history.columns if column.startswith("parameters/") or column.startswith("gradients/")]

    # Drop them (a no-op when the run was not tracked with wandb.watch).
    return run_history.drop(columns = columns_to_drop)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Main function

def download_project_runs_metrics(args) -> None :
    """
    Download the metrics of all the runs in a wandb project and save them to a JSON file.

    Every run of the ``entity/project`` is fetched through the wandb public API, its full history is read (via :func:`get_run_information`) and the model parameters/gradients are stripped from it (via :func:`remove_parameters_from_history`).
    The output is a JSON file mapping each metric name to a dictionary that maps every run (identified by ``"{run_name}_{run_id}"``) to the list of the values that metric took over the run.
    The wandb internal columns (``_step``, ``_runtime``, ``_timestamp``, ...) are skipped, and the ``NaN`` left by the sparse history are dropped per metric so that each list is the clean curve of that metric.

    Example output ::

        {
            "train_loss": {
                "run_name_run_id": [1.0, 0.9, 0.8],
                "run_name_run_id": [1.1, 1.0, 0.9]
            },
            "accuracy": {
                "run_name_run_id": [0.1, 0.2, 0.3],
                "run_name_run_id": [0.15, 0.25, 0.35]
            }
        }

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments parsed in :func:`jade.cli.cli.download_project_runs_metrics`. The expected fields are:

        - ``entity`` : str. Name of the wandb entity (i.e. the user or team name) that owns the project.
        - ``project`` : str. Name of the wandb project whose runs metrics are downloaded.
        - ``path_save`` : str or None. Path of the JSON file where the metrics are saved. If ``None`` the file ``wandb_metrics.json`` is created in the current working directory.
    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Check the arguments

    # entity and project are required (the argparse parser already enforces this, these guards also protect direct programmatic calls).
    if args.entity is None  : raise ValueError("The --entity argument is required. Please specify the name of the wandb entity (i.e. the user or team name).")
    if args.project is None : raise ValueError("The --project argument is required. Please specify the name of the wandb project.")

    # Resolve the output path, falling back to wandb_metrics.json in the current working directory.
    if args.path_save is None :
        print("The --path_save argument is not specified. The metrics will be saved in the current working directory in the file wandb_metrics.json.")
        path_save = "./wandb_metrics.json"
    else :
        path_save = args.path_save

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Download the metrics

    try :
        # Get all the runs of the specified project.
        runs = wandb.Api().runs(f"{args.entity}/{args.project}")

        metrics_dict = dict()

        for run in runs :
            # Read the run information (name, id and full history) and remove the model parameters/gradients from the history.
            run_info   = get_run_information(run)
            metrics_df = remove_parameters_from_history(run_info["history"])

            # Build the run identifier used as key (run name plus id, so runs sharing a name stay distinct).
            name_to_use = f"{run_info['name']}_{run_info['id']}"

            # Save the metrics of the current run in the metrics dict.
            for metric in metrics_df.columns :
                # Skip the columns created by wandb (_step, _runtime, _timestamp, ...).
                if metric.startswith("_") : continue

                # If the metric is not already in the metrics dict, create a new entry for it.
                if metric not in metrics_dict : metrics_dict[metric] = dict()

                # Save the metric values for the current run, dropping the NaN left by the sparse full history so the list is the clean metric curve.
                metrics_dict[metric][name_to_use] = metrics_df[metric].dropna().tolist()

        # Save the metrics dict to a JSON file.
        with open(path_save, "w") as file_json : json.dump(metrics_dict, file_json, indent = 4)

    except wandb.errors.CommError as e :
        print(f"An error occurred while trying to get the runs from the specified project. Please check the specified arguments and try again. Error message: {str(e)}")
    except Exception as e :
        print(f"An unexpected error occurred while trying to get the runs from the specified project. Please check the specified arguments and try again. Error message: {str(e)}")
