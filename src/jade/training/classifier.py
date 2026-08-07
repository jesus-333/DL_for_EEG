"""
Full training cycle for a classifier model.

This module implements the complete classification training loop:

* :func:`train` runs the whole cycle (dataloaders, optimizer/scheduler/loss creation, epoch loop, checkpointing, optional metrics and optional wandb logging) and returns the trained model together with the metrics computed during training.
* :func:`wandb_train` is a thin wrapper that opens a wandb run (and a model artifact) and then calls :func:`train` inside it.

Both functions are driven by a :class:`~jade.config.training.training_config` instance (a plain dict is accepted too and promoted automatically).
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

# Optional third party import (wandb is an optional extra of the package).
try :
    import wandb
except ImportError :
    wandb = None

# Third party imports
import torch

# Internal imports
from jade.config.training import training_config as training_config_class
from jade.training import classifier_epoch, metrics, support_training

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module logger

logger = logging.getLogger(__name__)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Full training cycle

def train(training_config, model_config, train_dataset, validation_dataset = None, wandb_model_artifact = None) :
    """

    """
    
    # To implement
    model = None

    return train_model(training_config, model, train_dataset, validation_dataset = None, wandb_model_artifact = None)

def train_model(training_config, model, train_dataset, validation_dataset = None, wandb_model_artifact = None) :
    """
    Train a classifier model using the given configuration and dataset(s).

    Parameters
    ----------
    training_config : training_config or dict
        The training configuration. An instance of :class:`~jade.config.training.training_config` (a dict is accepted and promoted automatically).
        It holds every hyper-parameter and option of the run, including the nested optimizer/scheduler/loss/dataloader/wandb configs.
    model : torch.nn.Module
        The model to train. For the metrics computation it should expose a ``classify`` method (as :class:`~jade.model.EEGNet.EEGNet_Classifier` does).
    train_dataset : torch.utils.data.Dataset
        Dataset used for training. Each item must be a ``(data, label)`` pair.
    validation_dataset : torch.utils.data.Dataset, optional
        Dataset used for validation, by default None. If None no validation is performed (and no ``model_BEST.pth`` is saved).
    wandb_model_artifact : wandb.Artifact, optional
        The wandb model artifact the checkpoints are added to, by default None.
        It is normally created and passed by :func:`wandb_train`; call that function instead of this one if you want to track the run with wandb.

    Returns
    -------
    model : torch.nn.Module
        The trained model.
    computed_metrics_during_training : dict
        Dictionary mapping each tracked quantity (``train_loss``, ``validation_loss``, and the per-set metrics when enabled) to the list of its per-epoch values.
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Config check and setup

    # Accept a plain dict and promote it to a proper training_config.
    if isinstance(training_config, dict) :
        training_config = training_config_class.from_dict(training_config)

    # Convenience aliases used throughout the function.
    device    = training_config.device
    use_wandb = training_config.wandb is not None

    # If wandb logging is requested, make sure wandb is actually installed.
    if use_wandb and wandb is None :
        raise ImportError('wandb logging was requested but wandb is not installed. Install it with "pip install wandb" (or the "extra" optional dependency of this package).')

    # (OPTIONAL) Seed numpy/torch for reproducibility.
    if training_config.seed is not None :
        support_training.set_seed(training_config.seed)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Dataloaders

    train_loader = support_training.get_dataloader(train_dataset, training_config.batch_size, training_config.dataloader)
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
        wandb.watch(model, criterion = loss_function, log = "all", log_freq = training_config.wandb.log_freq, log_graph = True)

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
            torch.save(model.state_dict(), os.path.join(training_config.path_to_save_model, "model_{}.pth".format(epoch + 1)))
            torch.save(optimizer.state_dict(), os.path.join(training_config.path_to_save_model, "optimizer_{}.pth".format(epoch + 1)))

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
                torch.save(model.state_dict(), os.path.join(training_config.path_to_save_model, "model_BEST.pth"))
                torch.save(optimizer.state_dict(), os.path.join(training_config.path_to_save_model, "optimizer_BEST.pth"))

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

            # Accumulate each metric (create the list on the first epoch, append afterwards).
            for metric in train_metrics_dict :
                if epoch == 0 :
                    computed_metrics_during_training["{}_train".format(metric)] = [train_metrics_dict[metric]]
                    if validation_loader is not None :
                        computed_metrics_during_training["{}_validation".format(metric)] = [validation_metrics_dict[metric]]
                else :
                    computed_metrics_during_training["{}_train".format(metric)].append(train_metrics_dict[metric])
                    if validation_loader is not None :
                        computed_metrics_during_training["{}_validation".format(metric)].append(validation_metrics_dict[metric])

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Learning-rate logging and scheduler step

        # Save the current learning rate (before stepping) so it is logged for this epoch.
        if use_wandb :
            log_dict["learning_rate"] = optimizer.param_groups[0]["lr"]

        # (OPTIONAL) Update the learning rate.
        if lr_scheduler is not None :
            lr_scheduler.step()

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # (OPTIONAL) Log the epoch summary (the spacing keeps the output aligned/readable)

        if training_config.print_var :
            logger.info("Epoch: %d", epoch)
            logger.info("\t Train loss        = %s", train_loss)
            if validation_loader is not None :
                logger.info("\t Validation loss   = %s", validation_loss)
            if lr_scheduler is not None :
                logger.info("\t Learning rate     = %s", optimizer.param_groups[0]["lr"])
            if training_config.measure_metrics_during_training :
                logger.info("\t Accuracy (TRAIN)  = %s", train_metrics_dict["accuracy"])
                if validation_loader is not None :
                    logger.info("\t Accuracy (VALID)  = %s", validation_metrics_dict["accuracy"])

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
            if (epoch + 1) % training_config.epoch_to_save_model == 0 and training_config.wandb.log_model_artifact and wandb_model_artifact is not None :
                model_file_path = os.path.join(training_config.path_to_save_model, "model_{}.pth".format(epoch + 1))
                wandb_model_artifact.add_file(model_file_path)
                wandb.save(model_file_path)

            # Log everything for this epoch.
            wandb.log(log_dict)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # End of training: save the final model

    model_file_path_END = os.path.join(training_config.path_to_save_model, "model_END.pth")
    torch.save(model.state_dict(), model_file_path_END)

    # (OPTIONAL) Upload the final (and best) model to the wandb artifact.
    if use_wandb and training_config.wandb.log_model_artifact and wandb_model_artifact is not None :
        wandb_model_artifact.add_file(model_file_path_END)
        wandb.save(model_file_path_END)

        # If validation was performed, upload the best model too.
        if validation_loader is not None :
            model_file_path_BEST = os.path.join(training_config.path_to_save_model, "model_BEST.pth")
            wandb_model_artifact.add_file(model_file_path_BEST)
            wandb.save(model_file_path_BEST)

    return model, computed_metrics_during_training

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# wandb wrapper

def wandb_train(training_config, model, train_dataset, validation_dataset = None, model_config = None) :
    """
    Train a classifier model while tracking the run on wandb.

    This function opens a wandb run (and, optionally, a model :class:`wandb.Artifact`), then calls :func:`train` inside it.
    For every detail about the training itself, refer to :func:`train`.

    Parameters
    ----------
    training_config : training_config or dict
        The training configuration. Its ``wandb`` field (a :class:`~jade.config.wandb_training.wandb_training_config`) must be set, otherwise a ``ValueError`` is raised.
    model : torch.nn.Module
        The model to train.
    train_dataset : torch.utils.data.Dataset
        Dataset used for training.
    validation_dataset : torch.utils.data.Dataset, optional
        Dataset used for validation, by default None.
    model_config : dict, optional
        Model configuration, by default None. It is not used during training, it is only logged on wandb (together with the training config) for reproducibility.

    Returns
    -------
    model : torch.nn.Module
        The trained model.
    computed_metrics_during_training : dict
        The per-epoch metrics returned by :func:`train`.
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Config check

    # Accept a plain dict and promote it to a proper training_config.
    if isinstance(training_config, dict) :
        training_config = training_config_class.from_dict(training_config)

    # wandb must be installed and the wandb config must be set.
    if wandb is None :
        raise ImportError('wandb is not installed. Install it with "pip install wandb" (or the "extra" optional dependency of this package).')
    if training_config.wandb is None :
        raise ValueError("training_config.wandb must be set (a wandb_training_config) to use wandb_train")

    wandb_config = training_config.wandb

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Build the config logged on wandb

    # The whole training config is logged; the model config is logged too when provided.
    config_to_log = dict(training_config = training_config.to_dict())
    if model_config is not None :
        config_to_log["model_config"] = model_config

    # Use the seed (if any) as the wandb run id, so a run can be resumed with the same id.
    run_id = str(training_config.seed) if training_config.seed is not None else None

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Open the wandb run and train

    with wandb.init(
        project  = wandb_config.project_name,
        entity   = wandb_config.entity,
        name     = wandb_config.name_training_run,
        job_type = "train",
        config   = config_to_log,
        notes    = wandb_config.notes,
        id       = run_id,
        resume   = "allow",
    ) as run :

        # Setup the artifact used to store the model.
        model_artifact_name = wandb_config.model_artifact_name + "_trained"
        if wandb_config.log_model_artifact :
            wandb_model_artifact = wandb.Artifact(
                model_artifact_name,
                type        = "model",
                description = "Trained {} model".format(wandb_config.model_artifact_name),
                metadata    = config_to_log,
            )
        else :
            wandb_model_artifact = None

        # Train the model.
        model, computed_metrics_during_training = train(training_config, model, train_dataset, validation_dataset, wandb_model_artifact)

        # Log the model artifact.
        if wandb_config.log_model_artifact :
            run.log_artifact(wandb_model_artifact)

    return model, computed_metrics_during_training
