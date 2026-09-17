"""
Full training cycle for a classifier model.

This module implements the complete classification training loop.
There are two entry points that build the model from its config, and two lower-level functions that train an already-built model (the ``wandb`` variants mirror the plain ones, adding a wandb run around them):

* :func:`train` is the standard entry point: it builds the model from a ``model_config`` (via :func:`~jade.model.get_model`), cross-checks the two configs, and then delegates to :func:`train_model`.
* :func:`train_model` runs the whole cycle on an already-built model (dataloaders, optimizer/scheduler/loss creation, epoch loop, checkpointing, optional metrics and optional wandb logging) and returns the trained model together with the metrics computed during training.
* :func:`wandb_train` mirrors :func:`train` but wraps the run with wandb: it cross-checks, builds the model and delegates to :func:`wandb_train_model`.
* :func:`wandb_train_model` mirrors :func:`train_model` (it receives an already-built model): it opens the wandb run (and a model artifact) and then calls :func:`train_model` inside it.

Every function is driven by config *instances*: a :class:`~jade.config.training.training_config` (and, for the model-building entry points, a :class:`~jade.config.model.generic.model_config`).
Consistently with the rest of the codebase these functions accept only the config dataclasses, not plain dicts (build the config with ``training_config.from_dict(...)`` first if you have a dict).
The single-epoch logic lives in :mod:`jade.training.classifier_epoch`, the factories and helpers in :mod:`jade.training.support_training`, and the metrics in :mod:`jade.training.metrics`.

Only the classification (finetuning) objective is implemented for now; a reconstruction/pretraining cycle may be added later in a sibling module.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Full module imports
import logging
import os
import sys

# Third party imports
import torch

# Internal imports
from jade.config.training import training_config as training_config_class
from jade.config.model.generic import model_config as model_config_class
from jade.config.dataset import dataset_config as dataset_config_class
from jade.model import get_model
from jade.training import classifier_epoch, metrics, support_training, wandb_utils

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module logger

logger = logging.getLogger(__name__)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Helper

# The check is shared with the other training cycles (see jade.training.vae), so it lives in support_training.
# The module-private name is kept as an alias for backward compatibility.
_require_config_instance = support_training.require_config_instance

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Full training cycle

def train(training_config, model_config, train_dataset, validation_dataset = None, wandb_model_artifact = None) :
    """
    Train a classifier model built from its configuration.

    This is the standard entry point of a training run: the model is created internally from ``model_config`` (through :func:`~jade.model.get_model`), the two configs are checked for consistency (:func:`~jade.training.support_training.cross_check_config`), and the actual training is delegated to :func:`train_model`.
    Call :func:`train_model` directly instead when you already have a built model you want to train.

    Parameters
    ----------
    training_config : training_config
        The training configuration, an instance of :class:`~jade.config.training.training_config`.
    model_config : model_config
        The model configuration, an instance of a :class:`~jade.config.model.generic.model_config` subclass (e.g. :class:`~jade.config.model.EEGNet_cfg.EEGNet_config`).
        The model to train is built from it.
    train_dataset : torch.utils.data.Dataset
        Dataset used for training. Each item must be a ``(data, label)`` pair.
    validation_dataset : torch.utils.data.Dataset, optional
        Dataset used for validation, by default None. If None no validation is performed (and no ``model_BEST.pth`` is saved).
    wandb_model_artifact : wandb.Artifact, optional
        The wandb model artifact the checkpoints are added to, by default None. It is normally created and passed by :func:`wandb_train_model`.

    Returns
    -------
    model : torch.nn.Module
        The trained model.
    computed_metrics_during_training : dict
        Dictionary mapping each tracked quantity (``train_loss``, ``validation_loss``, and the per-set metrics when enabled) to the list of its per-epoch values.

    Raises
    ------
    TypeError
        If ``training_config`` or ``model_config`` is not the corresponding config dataclass instance.
    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Config checks

    # Only config dataclass instances are accepted (no dict promotion), consistently with the rest of the codebase.
    _require_config_instance(training_config, training_config_class, "training_config")
    _require_config_instance(model_config, model_config_class, "model_config")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Cross checks

    # Verify the training and model configs are mutually consistent (e.g. loss vs last-layer activation) before building anything, so a mismatch fails fast with a clear message.
    support_training.cross_check_config(training_config, model_config)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Model

    # Build the model from its config: the model name carried by the config selects the class.
    model = get_model(model_config)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Training

    # Delegate the actual training cycle to train_model, forwarding every argument.
    return train_model(training_config, model, train_dataset, validation_dataset, wandb_model_artifact)

def train_model(training_config, model, train_dataset, validation_dataset = None, wandb_model_artifact = None) :
    """
    Train a classifier model using the given configuration and dataset(s).

    Parameters
    ----------
    training_config : training_config
        The training configuration, an instance of :class:`~jade.config.training.training_config`.
        It holds every hyper-parameter and option of the run, including the nested optimizer/scheduler/loss/dataloader/wandb configs.
    model : torch.nn.Module
        The model to train. For the metrics computation it should expose a ``classify`` method (as :class:`~jade.model.EEGNet.EEGNet_Classifier` does).
    train_dataset : torch.utils.data.Dataset
        Dataset used for training. Each item must be a ``(data, label)`` pair.
    validation_dataset : torch.utils.data.Dataset, optional
        Dataset used for validation, by default None. If None no validation is performed (and no ``model_BEST.pth`` is saved).
    wandb_model_artifact : wandb.Artifact, optional
        The wandb model artifact the checkpoints are added to, by default None.
        It is normally created and passed by :func:`wandb_train_model`; call that function instead of this one if you want to track the run with wandb.

    Returns
    -------
    model : torch.nn.Module
        The trained model.
    computed_metrics_during_training : dict
        Dictionary mapping each tracked quantity (``train_loss``, ``validation_loss``, and the per-set metrics when enabled) to the list of its per-epoch values.

    Raises
    ------
    TypeError
        If ``training_config`` is not a :class:`~jade.config.training.training_config` instance.
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Config check and setup

    # Only a training_config dataclass instance is accepted (no dict promotion).
    _require_config_instance(training_config, training_config_class, "training_config")

    # Convenience aliases used throughout the function.
    device    = training_config.device
    use_wandb = training_config.wandb is not None

    # If wandb logging is requested, make sure wandb is actually installed.
    if use_wandb : wandb_utils.require_wandb()

    # (OPTIONAL) Seed numpy/torch for reproducibility.
    if training_config.seed is not None : support_training.set_seed(training_config.seed)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Dataloaders
    
    # Training set
    train_loader = support_training.get_dataloader(train_dataset, training_config.batch_size, training_config.dataloader)

    # Validation set
    if validation_dataset is not None :
        validation_loader = support_training.get_dataloader(validation_dataset, training_config.batch_size, training_config.dataloader)
    else :
        validation_loader = None

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Loss function, optimizer and scheduler

    # Move the model to the training device.
    model.to(device)

    # (OPTIONAL) Load past model weights to continue a previous training.
    if training_config.path_past_weights is not None :
        model_state_dict = torch.load(training_config.path_past_weights, map_location = device)
        model.load_state_dict(model_state_dict)
        if training_config.print_var : logger.info("Model weights loaded from %s", training_config.path_past_weights)

    # Create the loss function and move it to the device (the move matters when the loss carries class weights).
    loss_function = support_training.get_loss_function(training_config.loss)
    loss_function.to(device)

    # Create the optimizer.
    optimizer = support_training.get_optimizer(training_config.optimizer, model)

    # (OPTIONAL) Load a past optimizer state to continue a previous training.
    if training_config.path_past_optimizer is not None :
        optimizer_state_dict = torch.load(training_config.path_past_optimizer, map_location = device)
        optimizer.load_state_dict(optimizer_state_dict)
        if training_config.print_var : logger.info("Optimizer state dict loaded from %s", training_config.path_past_optimizer)

    # (OPTIONAL) Create the learning-rate scheduler (used only when a scheduler config is provided).
    if training_config.lr_scheduler is not None :
        lr_scheduler = support_training.get_lr_scheduler(training_config.lr_scheduler, optimizer)
    else :
        lr_scheduler = None

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Training preparation

    # Create the folder where the checkpoints are stored (if it does not exist already).
    os.makedirs(training_config.path_to_save_model, exist_ok = True)

    # (OPTIONAL) Ask wandb to track the model parameters/gradients.
    if use_wandb :
        wandb_utils.watch_model(model, loss_function, training_config.wandb)

    # Variable to track the best (lowest) validation loss.
    best_loss_val = sys.maxsize

    # Two dictionaries with different purposes:
    # - computed_metrics_during_training stores, for each metric, the list of its per-epoch values, and it is returned at the end.
    # - log_dict stores the metrics of a single epoch and it is used by wandb. It is not reset explicitly at each epoch: the same keys are overwritten, which has the same effect.
    log_dict = {}
    computed_metrics_during_training = dict()

    if training_config.print_var : logger.info("Start training")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Epoch loop

    for epoch in range(training_config.epochs) :
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # (MANDATORY) Training epoch

        # Advance one epoch on the training set (forward + backward pass).
        train_loss = classifier_epoch.train_epoch_function(model, train_loader, loss_function, optimizer, device, log_dict, training_config.print_var)

        # Save a checkpoint every epoch_to_save_model epochs (i.e. model_n.pth).
        # Note that when epoch == n the model has been trained for n + 1 epochs.
        if (epoch + 1) % training_config.epoch_to_save_model == 0 :
            torch.save(model.state_dict(), os.path.join(training_config.path_to_save_model, f"model_{epoch + 1}.pth"))
            torch.save(optimizer.state_dict(), os.path.join(training_config.path_to_save_model, f"optimizer_{epoch + 1}.pth"))

        # (OPTIONAL) Save an alternating A/B backup every epoch (overwritten each time; handy for very large models).
        if training_config.backup_model_every_epoch :
            support_training.backup_every_epoch(training_config.path_to_save_model, epoch, model, optimizer, lr_scheduler)

        # Accumulate the training loss (create the list on the first epoch, append afterwards).
        if epoch == 0 :
            computed_metrics_during_training["train_loss"] = [train_loss]
        else :
            computed_metrics_during_training["train_loss"].append(train_loss)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # (OPTIONAL) Validation epoch

        validation_loss = None
        if validation_loader is not None :
            validation_loss = classifier_epoch.validation_epoch_function(model, validation_loader, loss_function, device, log_dict)

            # Save the new BEST model when a new minimum validation loss is reached.
            if validation_loss < best_loss_val :
                best_loss_val = validation_loss
                new_best_found = True
                torch.save(model.state_dict(), os.path.join(training_config.path_to_save_model, "model_BEST.pth"))
                torch.save(optimizer.state_dict(), os.path.join(training_config.path_to_save_model, "optimizer_BEST.pth"))
            else :
                new_best_found = False

            # Accumulate the validation loss.
            if epoch == 0 :
                computed_metrics_during_training["validation_loss"] = [validation_loss]
            else :
                computed_metrics_during_training["validation_loss"].append(validation_loss)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # (OPTIONAL) Classification metrics

        train_metrics_dict = None
        validation_metrics_dict = None
        if training_config.measure_metrics_during_training :
            # Compute the metrics on the training set (and validation set, if any).
            train_metrics_dict = metrics.compute_metrics(model, train_loader, device)
            if validation_loader is not None :
                validation_metrics_dict = metrics.compute_metrics(model, validation_loader, device)

                if new_best_found :
                    # Save the metrics of the new BEST model (TODO)

                    # Save the current accuracy (used for log)
                    best_accuracy_val = validation_metrics_dict["accuracy"]

            # Accumulate each metric (create the list on the first epoch, append afterwards).
            for metric in train_metrics_dict :
                if epoch == 0 :
                    computed_metrics_during_training[f"{metric}_train"] = [train_metrics_dict[metric]]
                    if validation_loader is not None :
                        computed_metrics_during_training[f"{metric}_validation"] = [validation_metrics_dict[metric]]
                else :
                    computed_metrics_during_training[f"{metric}_train"].append(train_metrics_dict[metric])
                    if validation_loader is not None :
                        computed_metrics_during_training[f"{metric}_validation"].append(validation_metrics_dict[metric])


        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Learning-rate logging and scheduler step

        # Save the current learning rate (before stepping) so it is logged for this epoch.
        if use_wandb : log_dict["learning_rate"] = optimizer.param_groups[0]["lr"]

        # (OPTIONAL) Update the learning rate.
        if lr_scheduler is not None : lr_scheduler.step()

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # (OPTIONAL) Log the epoch summary (the spacing keeps the output aligned/readable)

        if training_config.print_var :
            logger.info("Epoch: %d", epoch)
            logger.info("\t Train loss             = %s", train_loss)
            if validation_loader is not None :
                logger.info("\t Validation loss        = %s", validation_loss)
                logger.info("\t Validation loss (BEST) = %s", best_loss_val)
            if lr_scheduler is not None :
                logger.info("\t Learning rate          = %s", optimizer.param_groups[0]["lr"])
            if training_config.measure_metrics_during_training :
                logger.info("\t Accuracy (TRAIN)       = %s", train_metrics_dict["accuracy"])
                if validation_loader is not None :
                    logger.info("\t Accuracy (VALID)       = %s", validation_metrics_dict["accuracy"])
                    logger.info("\t Accuracy (VALID/BEST)  = %s", best_accuracy_val)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # (OPTIONAL) Log the epoch on wandb

        if use_wandb :
            # Update the log with the epoch losses.
            log_dict["train_loss"] = train_loss
            if validation_loader is not None :
                log_dict["validation_loss"] = validation_loss

            # Save the metrics in the log.
            if training_config.measure_metrics_during_training :
                support_training.update_log_dict_metrics(train_metrics_dict, log_dict, "train")
                if validation_loader is not None :
                    support_training.update_log_dict_metrics(validation_metrics_dict, log_dict, "validation")

            # Add the periodic checkpoint to the artifact.
            if (epoch + 1) % training_config.epoch_to_save_model == 0 :
                model_file_path = os.path.join(training_config.path_to_save_model, f"model_{epoch + 1}.pth")
                wandb_utils.add_file_to_artifact(wandb_model_artifact, model_file_path)

            # Log everything for this epoch.
            wandb_utils.log_epoch(log_dict)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # End of training: save the final model

    model_file_path_END = os.path.join(training_config.path_to_save_model, "model_END.pth")
    torch.save(model.state_dict(), model_file_path_END)

    # (OPTIONAL) Upload the final (and best) model to the wandb artifact.
    if use_wandb and wandb_model_artifact is not None :
        wandb_utils.add_file_to_artifact(wandb_model_artifact, model_file_path_END)

        # If validation was performed, upload the best model too.
        if validation_loader is not None :
            model_file_path_BEST = os.path.join(training_config.path_to_save_model, "model_BEST.pth")
            wandb_utils.add_file_to_artifact(wandb_model_artifact, model_file_path_BEST)

    return model, computed_metrics_during_training

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# wandb wrappers

def wandb_train(training_config, model_config, train_dataset, validation_dataset = None, dataset_config = None) :
    """
    Train a classifier built from its configuration while tracking the run on wandb.

    This is the wandb counterpart of :func:`train`: it cross-checks the two configs, builds the model from ``model_config`` (through :func:`~jade.model.get_model`), and delegates to :func:`wandb_train_model` (which opens the wandb run).
    Call :func:`wandb_train_model` directly instead when you already have a built model you want to train.

    Parameters
    ----------
    training_config : training_config
        The training configuration, an instance of :class:`~jade.config.training.training_config`. Its ``wandb`` field (a :class:`~jade.config.wandb_training.wandb_training_config`) must be set, otherwise a ``ValueError`` is raised.
    model_config : model_config
        The model configuration, an instance of a :class:`~jade.config.model.generic.model_config` subclass. The model to train is built from it; it is also logged on wandb for reproducibility.
    train_dataset : torch.utils.data.Dataset
        Dataset used for training. Each item must be a ``(data, label)`` pair.
    validation_dataset : torch.utils.data.Dataset, optional
        Dataset used for validation, by default None.
    dataset_config : dataset_config, optional
        The dataset configuration, by default None. It is not used during training; it is only logged on wandb (together with the training and model configs) for reproducibility.

    Returns
    -------
    model : torch.nn.Module
        The trained model.
    computed_metrics_during_training : dict
        The per-epoch metrics returned by :func:`train_model`.

    Raises
    ------
    TypeError
        If ``training_config``/``model_config`` (or ``dataset_config`` when given) is not the corresponding config dataclass instance.
    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Config checks

    # Only config dataclass instances are accepted (no dict promotion).
    _require_config_instance(training_config, training_config_class, "training_config")
    _require_config_instance(model_config, model_config_class, "model_config")
    if dataset_config is not None :
        _require_config_instance(dataset_config, dataset_config_class, "dataset_config")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Cross checks and model

    # Verify the training and model configs are mutually consistent before building anything.
    support_training.cross_check_config(training_config, model_config)

    # Build the model from its config (the model name carried by the config selects the class).
    model = get_model(model_config)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Training

    # Delegate to wandb_train_model, forwarding model_config and dataset_config so both are backed up on wandb.
    return wandb_train_model(
        training_config, model, train_dataset, validation_dataset,
        model_config = model_config, dataset_config = dataset_config,
    )

def wandb_train_model(training_config, model, train_dataset, validation_dataset = None, model_config = None, dataset_config = None) :
    """
    Train an already-built classifier model while tracking the run on wandb.

    This is the wandb counterpart of :func:`train_model`: it opens a wandb run (and, optionally, a model :class:`wandb.Artifact`), then calls :func:`train_model` inside it.
    For every detail about the training itself, refer to :func:`train_model`.

    Parameters
    ----------
    training_config : training_config
        The training configuration, an instance of :class:`~jade.config.training.training_config`. Its ``wandb`` field (a :class:`~jade.config.wandb_training.wandb_training_config`) must be set, otherwise a ``ValueError`` is raised.
    model : torch.nn.Module
        The model to train.
    train_dataset : torch.utils.data.Dataset
        Dataset used for training. Each item must be a ``(data, label)`` pair.
    validation_dataset : torch.utils.data.Dataset, optional
        Dataset used for validation, by default None.
    model_config : model_config, optional
        The model configuration, by default None. It is not used during training (this function receives an already-built ``model``); it is only logged on wandb (together with the training config) for reproducibility.
    dataset_config : dataset_config, optional
        The dataset configuration, by default None. Same logic as ``model_config``: it is not used during training, only logged on wandb for reproducibility.

    Returns
    -------
    model : torch.nn.Module
        The trained model.
    computed_metrics_during_training : dict
        The per-epoch metrics returned by :func:`train_model`.

    Raises
    ------
    TypeError
        If ``training_config`` (or ``model_config``/``dataset_config`` when given) is not the corresponding config dataclass instance.
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Config check

    # Only config dataclass instances are accepted (no dict promotion).
    _require_config_instance(training_config, training_config_class, "training_config")
    if model_config is not None :
        _require_config_instance(model_config, model_config_class, "model_config")
    if dataset_config is not None :
        _require_config_instance(dataset_config, dataset_config_class, "dataset_config")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Open the wandb run and train

    # Everything about the wandb run is model-agnostic, so it lives in jade.training.wandb_utils and it is shared with the other training cycles.
    # The only classifier-specific part is which training function runs inside the run, which is why train_model is passed as first argument.
    return wandb_utils.run_training_with_wandb(
        train_model, training_config, model, train_dataset, validation_dataset,
        model_config   = model_config,
        dataset_config = dataset_config,
    )
