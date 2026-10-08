"""
wandb helpers shared by every training cycle.

`Weights & Biases <https://wandb.ai>`_ is an optional way to track a training run.
Everything it needs (opening the run, logging the configs and the per-epoch metrics, building and uploading the model artifact) is model-agnostic, so it lives here instead of being duplicated inside each training cycle.
Thanks to this module neither :mod:`jade.training.classifier` nor :mod:`jade.training.vae` imports ``wandb`` directly.

The two public entry points of a tracked run, :func:`jade.training.classifier.wandb_train_model` and :func:`jade.training.vae.wandb_train_model`, are thin wrappers around :func:`run_training_with_wandb`, which is the generic version: it receives the training function to call as its first argument.

Note that ``wandb`` is an optional dependency (the ``extra`` group of the package), so it is imported defensively: importing this module never fails, and a clear :class:`ImportError` is raised only when a tracked run is actually requested (see :func:`require_wandb`).
Note also that the name of this module is ``wandb_utils`` and not ``wandb`` on purpose: ``import wandb`` below refers to the installed package (python 3 always uses absolute imports), but a module named ``wandb`` inside the package would make every read of ``jade.training.wandb`` ambiguous.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Full module imports
import logging

# Specific imports
from contextlib import contextmanager

# Optional third party import (wandb is an optional extra of the package).
try :
    import wandb
except ImportError :
    wandb = None

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module logger and constants

logger = logging.getLogger(__name__)

# True when wandb is installed, i.e. when a tracked run is possible.
WANDB_IS_AVAILABLE = wandb is not None

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Availability

def require_wandb() -> None :
    """
    Ensure wandb is installed.

    Raises
    ------
    ImportError
        If wandb is not installed.
    """

    if wandb is None :
        raise ImportError('wandb logging was requested but wandb is not installed. Install it with "pip install wandb" (or the "extra" optional dependency of this package).')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Run setup

def build_config_to_log(training_config, model_config = None, dataset_config = None) -> dict :
    """
    Build the dictionary of configs attached to a wandb run.

    Only the training config is mandatory. The model and dataset configs are logged when provided, for reproducibility only: they do not affect the training.

    Parameters
    ----------
    training_config : training_config
        The training configuration.
    model_config : model_config, optional
        The model configuration, by default None.
    dataset_config : dataset_config, optional
        The dataset configuration, by default None.

    Returns
    -------
    dict
        The configs, each serialized with its own ``to_dict``.
    """

    config_to_log = {}

    # The whole training config is logged
    config_to_log["training_config"] = training_config.to_dict()

    # The model and dataset configs are logged too when provided (for reproducibility only, they do not affect training).
    if model_config is not None :
        config_to_log["model_config"] = model_config.to_dict()
    if dataset_config is not None :
        config_to_log["dataset_config"] = dataset_config.to_dict()

    return config_to_log


def build_run_id(training_config) -> str :
    """
    Build the id of a wandb run.

    Parameters
    ----------
    training_config : training_config
        The training configuration. Both its ``seed`` and the ``name_training_run`` of its wandb config are used.

    Returns
    -------
    str or None
        The run id, or None when either the seed or the run name is missing (in that case wandb generates its own id).
    """

    wandb_config = training_config.wandb

    # Use the seed (if any) and the name of the training run as the wandb run id, so a run can be resumed with the same id.
    if training_config.seed is not None and wandb_config.name_training_run is not None :
        return f"{str(training_config.seed)}_{wandb_config.name_training_run}"

    return None


def create_model_artifact(wandb_config, config_to_log : dict) :
    """
    Create the wandb artifact used to store the weights of the model.

    Parameters
    ----------
    wandb_config : wandb_training_config
        The wandb configuration. Both ``model_artifact_name`` and ``log_model_artifact`` are read.
    config_to_log : dict
        The configs of the run, attached to the artifact as metadata.

    Returns
    -------
    wandb.Artifact or None
        The artifact, or None when ``wandb_config.log_model_artifact`` is False.
    """

    if not wandb_config.log_model_artifact :
        return None

    model_artifact_name = wandb_config.model_artifact_name + "_trained"

    return wandb.Artifact(
        model_artifact_name,
        type        = "model",
        description = f"Trained {wandb_config.model_artifact_name} model",
        metadata    = config_to_log,
    )


@contextmanager
def wandb_run(training_config, model_config = None, dataset_config = None) :
    """
    Open a wandb run (and its model artifact) for a training cycle.

    It is a context manager: it yields the run and the artifact, and it uploads the artifact when the block is left without an error.

    Parameters
    ----------
    training_config : training_config
        The training configuration. Its ``wandb`` field must be set.
    model_config : model_config, optional
        The model configuration, logged for reproducibility, by default None.
    dataset_config : dataset_config, optional
        The dataset configuration, logged for reproducibility, by default None.

    Yields
    ------
    run : wandb.sdk.wandb_run.Run
        The wandb run.
    wandb_model_artifact : wandb.Artifact or None
        The artifact where the weights are stored (None when the artifact logging is disabled).

    Raises
    ------
    ImportError
        If wandb is not installed.
    ValueError
        If ``training_config.wandb`` is not set.
    """

    # wandb must be installed and the wandb config must be set.
    require_wandb()
    if training_config.wandb is None :
        raise ValueError("training_config.wandb must be set (a wandb_training_config) to track a run with wandb")

    wandb_config  = training_config.wandb
    config_to_log = build_config_to_log(training_config, model_config, dataset_config)

    with wandb.init(
        project  = wandb_config.project_name,
        entity   = wandb_config.entity,
        name     = wandb_config.name_training_run,
        job_type = "train",
        config   = config_to_log,
        notes    = wandb_config.notes,
        id       = build_run_id(training_config),
        resume   = "allow",
    ) as run :

        # Setup the artifact used to store the model.
        wandb_model_artifact = create_model_artifact(wandb_config, config_to_log)

        yield run, wandb_model_artifact

        # Log the model artifact. Note that this is reached only when the training block ends without an error.
        if wandb_model_artifact is not None :
            run.log_artifact(wandb_model_artifact)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Logging during the training

def watch_model(model, loss_function, wandb_config) -> None :
    """
    Ask wandb to track the parameters and the gradients of a model.

    Parameters
    ----------
    model : torch.nn.Module
        The model being trained.
    loss_function : torch.nn.Module
        The loss used to train it (wandb calls it the criterion).
    wandb_config : wandb_training_config
        The wandb configuration. Only ``log_freq`` is read.
    """

    wandb.watch(model, criterion = loss_function, log = "all", log_freq = wandb_config.log_freq, log_graph = True)


def add_file_to_artifact(wandb_model_artifact, file_path : str) -> None :
    """
    Add a file to the model artifact and upload it.

    It is a no-op when there is no artifact, so the caller does not have to check it.

    Parameters
    ----------
    wandb_model_artifact : wandb.Artifact or None
        The artifact where the weights are stored.
    file_path : str
        Path of the file to add.
    """

    if wandb_model_artifact is None :
        return

    wandb_model_artifact.add_file(file_path)
    wandb.save(file_path)


def log_epoch(log_dict : dict) -> None :
    """
    Log the metrics of an epoch on wandb.

    It exists so a training cycle never has to import wandb itself.

    Parameters
    ----------
    log_dict : dict
        The metrics of the epoch.
    """

    wandb.log(log_dict)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Generic tracked training

def run_training_with_wandb(train_model_function, training_config, model, train_dataset, validation_dataset = None, model_config = None, dataset_config = None) :
    """
    Run a training cycle inside a wandb run.

    This is the generic body shared by every ``wandb_train_model``: the only thing that changes between a classifier and a generative model is *which* training function is called, so it is received as the first argument.
    The public entry points (:func:`jade.training.classifier.wandb_train_model` and :func:`jade.training.vae.wandb_train_model`) are thin wrappers that pass their own ``train_model``.

    Parameters
    ----------
    train_model_function : callable
        The function that trains the already-built model. It is called as ``train_model_function(training_config, model, train_dataset, validation_dataset, wandb_model_artifact)`` and it must return the ``(model, computed_metrics_during_training)`` pair.
    training_config : training_config
        The training configuration. Its ``wandb`` field must be set.
    model : torch.nn.Module
        The model to train.
    train_dataset : torch.utils.data.Dataset
        Dataset used for training.
    validation_dataset : torch.utils.data.Dataset, optional
        Dataset used for validation, by default None.
    model_config : model_config, optional
        The model configuration, logged for reproducibility, by default None.
    dataset_config : dataset_config, optional
        The dataset configuration, logged for reproducibility, by default None.

    Returns
    -------
    model : torch.nn.Module
        The trained model.
    computed_metrics_during_training : dict
        The per-epoch metrics returned by ``train_model_function``.
    """

    with wandb_run(training_config, model_config, dataset_config) as (run, wandb_model_artifact) :

        # Train the model. The model is already built, so the already-built-model function is called directly.
        model, computed_metrics_during_training = train_model_function(training_config, model, train_dataset, validation_dataset, wandb_model_artifact)

    return model, computed_metrics_during_training
