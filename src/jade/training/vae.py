"""
Full training cycle for a generative (VAE) model.

This module is the counterpart of :mod:`jade.training.classifier` for a model that reconstructs its input instead of classifying it, e.g. :class:`~jade.model.vEEGNet.vEEGNet`.
It offers the same four entry points, with the same signatures and the same return value, so moving a run from a classifier to a variational autoencoder only means importing another module:

* :func:`train` is the standard entry point: it builds the model from a ``model_config`` (via :func:`~jade.model.get_model`), cross-checks the two configs, and then delegates to :func:`train_model`.
* :func:`train_model` runs the whole cycle on an already-built model (dataloaders, optimizer/scheduler/loss creation, epoch loop, checkpointing, optional metrics and optional wandb logging) and returns the trained model together with the metrics computed during training.
* :func:`wandb_train` mirrors :func:`train` but wraps the run with wandb: it cross-checks, builds the model and delegates to :func:`wandb_train_model`.
* :func:`wandb_train_model` mirrors :func:`train_model` (it receives an already-built model): it opens the wandb run (and a model artifact) and then calls :func:`train_model` inside it.

The loss must be a generative one (the ELBO, see :class:`~jade.training.custom_loss_function.elbo.elbo_loss`), which :func:`~jade.training.support_training.cross_check_config` enforces.
Three things differ from the classification cycle:

* the loss of a VAE is made of several terms, so each of them is tracked separately (the single-epoch logic lives in :mod:`jade.training.vae_epoch`);
* the metrics are reconstruction metrics (:func:`~jade.training.metrics.compute_reconstruction_metrics`) and not classification ones, which need a label the model never predicts;
* the loss may have a warm-up, so the cycle tells it which epoch it is in.

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
from jade.training import metrics, support_training, vae_epoch, wandb_utils

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module logger

logger = logging.getLogger(__name__)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Helper

# The check is shared with the other training cycles (see jade.training.classifier), so it lives in support_training.
_require_config_instance = support_training.require_config_instance


def _accumulate(computed_metrics_during_training : dict, key : str, value) -> None :
    """
    Append a per-epoch value to the list of its metric, creating the list the first time.

    Parameters
    ----------
    computed_metrics_during_training : dict
        The dictionary mapping each tracked quantity to the list of its per-epoch values.
    key : str
        The name of the quantity.
    value : float
        The value of the current epoch.
    """

    computed_metrics_during_training.setdefault(key, []).append(value)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Full training cycle

def train(training_config, model_config, train_dataset, validation_dataset = None, wandb_model_artifact = None) :
    """
    Train a generative model built from its configuration.

    This is the standard entry point of a training run: the model is created internally from ``model_config`` (through :func:`~jade.model.get_model`), the two configs are checked for consistency (:func:`~jade.training.support_training.cross_check_config`), and the actual training is delegated to :func:`train_model`.
    Call :func:`train_model` directly instead when you already have a built model you want to train.

    Parameters
    ----------
    training_config : training_config
        The training configuration, an instance of :class:`~jade.config.training.training_config`. Its ``loss`` must be a generative loss (e.g. :class:`~jade.config.loss_functions.elbo.elbo_config`).
    model_config : model_config
        The model configuration, an instance of a :class:`~jade.config.model.generic.model_config` subclass describing a generative model (e.g. :class:`~jade.config.model.vEEGNet_cfg.vEEGNet_config`).
    train_dataset : torch.utils.data.Dataset
        Dataset used for training. Each item is either a ``(data, label)`` pair or the data alone; the label is never used.
    validation_dataset : torch.utils.data.Dataset, optional
        Dataset used for validation, by default None.
    wandb_model_artifact : wandb.Artifact, optional
        The wandb model artifact the checkpoints are added to, by default None. It is normally created and passed by :func:`wandb_train_model`.

    Returns
    -------
    model : torch.nn.Module
        The trained model.
    computed_metrics_during_training : dict
        The per-epoch metrics returned by :func:`train_model`.

    Raises
    ------
    TypeError
        If ``training_config`` or ``model_config`` is not the corresponding config dataclass instance.
    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Config checks

    # Only config dataclass instances are accepted (no dict promotion).
    _require_config_instance(training_config, training_config_class, "training_config")
    _require_config_instance(model_config, model_config_class, "model_config")

    # Verify the training and model configs are mutually consistent before building anything.
    support_training.cross_check_config(training_config, model_config)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Model and training

    # Build the model from its config (the model name carried by the config selects the class).
    model = get_model(model_config)

    return train_model(training_config, model, train_dataset, validation_dataset, wandb_model_artifact)


def train_model(training_config, model, train_dataset, validation_dataset = None, wandb_model_artifact = None) :
    """
    Train an already-built generative model.

    Parameters
    ----------
    training_config : training_config
        The training configuration, an instance of :class:`~jade.config.training.training_config`.
        It holds every hyper-parameter and option of the run, including the nested optimizer/scheduler/loss/dataloader/wandb configs.
    model : torch.nn.Module
        The model to train. Its forward must return the reconstruction and the parameters of the latent distribution.
    train_dataset : torch.utils.data.Dataset
        Dataset used for training.
    validation_dataset : torch.utils.data.Dataset, optional
        Dataset used for validation, by default None.
    wandb_model_artifact : wandb.Artifact, optional
        The wandb model artifact the checkpoints are added to, by default None.
        It is normally created and passed by :func:`wandb_train_model`; call that function instead of this one if you want to track the run with wandb.

    Returns
    -------
    model : torch.nn.Module
        The trained model.
    computed_metrics_during_training : dict
        Dictionary mapping each tracked quantity to the list of its per-epoch values.
        Every term of the loss is tracked (``train_loss``, ``train_recon_loss``, ``train_kl_loss``, ``train_loss_full_kl``, their ``validation_`` counterparts and ``kl_weight``), plus the reconstruction metrics when they are enabled.

    Raises
    ------
    TypeError
        If ``training_config`` is not a :class:`~jade.config.training.training_config` instance.

    Notes
    -----
    The best model is selected on ``loss_full_kl`` and not on the optimized loss. The two are identical unless the loss has a warm-up, in which case the optimized value changes meaning at every epoch (the KL term is cheaper at the beginning), so comparing it across epochs would systematically favour the first ones.
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

    # Create the loss function and move it to the device.
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

    # Variables to track the best (lowest) validation loss and the metrics that go with it.
    # They are initialized before the loop, so they are always defined, whatever happens at the first epoch.
    best_loss_val = sys.maxsize
    best_reconstruction_mse_val = None
    new_best_found = False

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
        # (OPTIONAL) Tell the loss which epoch we are in

        # A loss with a warm-up/annealing schedule (e.g. the ELBO with its KL warm-up) updates its weights here, once per epoch.
        # The check keeps this cycle usable with a loss that has no schedule at all.
        if hasattr(loss_function, "set_epoch") : loss_function.set_epoch(epoch)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # (MANDATORY) Training epoch

        # Advance one epoch on the training set (forward + backward pass).
        train_terms = vae_epoch.train_epoch_function(model, train_loader, loss_function, optimizer, device, log_dict, training_config.print_var)

        # Save a checkpoint every epoch_to_save_model epochs (i.e. model_n.pth).
        # Note that when epoch == n the model has been trained for n + 1 epochs.
        if (epoch + 1) % training_config.epoch_to_save_model == 0 :
            torch.save(model.state_dict(), os.path.join(training_config.path_to_save_model, f"model_{epoch + 1}.pth"))
            torch.save(optimizer.state_dict(), os.path.join(training_config.path_to_save_model, f"optimizer_{epoch + 1}.pth"))

        # (OPTIONAL) Save an alternating A/B backup every epoch (overwritten each time; handy for very large models).
        if training_config.backup_model_every_epoch :
            support_training.backup_every_epoch(training_config.path_to_save_model, epoch, model, optimizer, lr_scheduler)

        # Accumulate every term of the training loss.
        for term_name, term_value in train_terms.items() :
            if term_name == "kl_weight" : _accumulate(computed_metrics_during_training, "kl_weight", term_value)
            else                        : _accumulate(computed_metrics_during_training, f"train_{term_name}", term_value)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # (OPTIONAL) Validation epoch

        validation_terms = None
        if validation_loader is not None :
            validation_terms = vae_epoch.validation_epoch_function(model, validation_loader, loss_function, device, log_dict)

            # Save the new BEST model when a new minimum validation loss is reached.
            # Note that the comparison uses loss_full_kl (see the note in the docstring).
            if validation_terms["loss_full_kl"] < best_loss_val :
                best_loss_val = validation_terms["loss_full_kl"]
                new_best_found = True
                torch.save(model.state_dict(), os.path.join(training_config.path_to_save_model, "model_BEST.pth"))
                torch.save(optimizer.state_dict(), os.path.join(training_config.path_to_save_model, "optimizer_BEST.pth"))
            else :
                new_best_found = False

            # Accumulate every term of the validation loss (kl_weight is shared with the training epoch, so it is not stored twice).
            for term_name, term_value in validation_terms.items() :
                if term_name != "kl_weight" : _accumulate(computed_metrics_during_training, f"validation_{term_name}", term_value)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # (OPTIONAL) Reconstruction metrics

        train_metrics_dict = None
        validation_metrics_dict = None
        if training_config.measure_metrics_during_training :
            # Compute the metrics on the training set (and validation set, if any).
            train_metrics_dict = metrics.compute_reconstruction_metrics(model, train_loader, device)
            if validation_loader is not None :
                validation_metrics_dict = metrics.compute_reconstruction_metrics(model, validation_loader, device)

                if new_best_found :
                    # Save the current reconstruction error (used for log).
                    best_reconstruction_mse_val = validation_metrics_dict["reconstruction_mse"]

            # Accumulate each metric.
            for metric in train_metrics_dict :
                _accumulate(computed_metrics_during_training, f"{metric}_train", train_metrics_dict[metric])
                if validation_loader is not None :
                    _accumulate(computed_metrics_during_training, f"{metric}_validation", validation_metrics_dict[metric])

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
            logger.info("\t Train loss             = %s", train_terms["loss"])
            logger.info("\t   reconstruction       = %s", train_terms["recon_loss"])
            logger.info("\t   KL (weight %.3f)     = %s", train_terms["kl_weight"], train_terms["kl_loss"])
            if validation_loader is not None :
                logger.info("\t Validation loss        = %s", validation_terms["loss"])
                # The best model is selected on the objective computed with the FINAL weight of the KL term, so both values are logged while a warm-up is running:
                # comparing the optimized loss above with the best one below would be comparing two different objectives.
                logger.info("\t Validation loss (full KL)        = %s", validation_terms["loss_full_kl"])
                logger.info("\t Validation loss (full KL, BEST)  = %s", best_loss_val)
            if lr_scheduler is not None :
                logger.info("\t Learning rate          = %s", optimizer.param_groups[0]["lr"])
            if training_config.measure_metrics_during_training :
                logger.info("\t Recon. MSE (TRAIN)     = %s", train_metrics_dict["reconstruction_mse"])
                if validation_loader is not None :
                    logger.info("\t Recon. MSE (VALID)     = %s", validation_metrics_dict["reconstruction_mse"])
                    logger.info("\t Recon. MSE (VALID/BEST)= %s", best_reconstruction_mse_val)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # (OPTIONAL) Log the epoch on wandb

        if use_wandb :
            # Note that the losses are already in log_dict: the epoch functions put them there.

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
    Train a generative model built from its configuration while tracking the run on wandb.

    This is the wandb counterpart of :func:`train`: it cross-checks the two configs, builds the model from ``model_config`` (through :func:`~jade.model.get_model`), and delegates to :func:`wandb_train_model` (which opens the wandb run).
    Call :func:`wandb_train_model` directly instead when you already have a built model you want to train.

    Parameters
    ----------
    training_config : training_config
        The training configuration. Its ``wandb`` field (a :class:`~jade.config.wandb_training.wandb_training_config`) must be set, otherwise a ``ValueError`` is raised.
    model_config : model_config
        The model configuration. The model to train is built from it; it is also logged on wandb for reproducibility.
    train_dataset : torch.utils.data.Dataset
        Dataset used for training.
    validation_dataset : torch.utils.data.Dataset, optional
        Dataset used for validation, by default None.
    dataset_config : dataset_config, optional
        The dataset configuration, by default None. It is not used during training; it is only logged on wandb for reproducibility.

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
    Train an already-built generative model while tracking the run on wandb.

    This is the wandb counterpart of :func:`train_model`: it opens a wandb run (and, optionally, a model :class:`wandb.Artifact`), then calls :func:`train_model` inside it.
    For every detail about the training itself, refer to :func:`train_model`.

    Parameters
    ----------
    training_config : training_config
        The training configuration. Its ``wandb`` field must be set, otherwise a ``ValueError`` is raised.
    model : torch.nn.Module
        The model to train.
    train_dataset : torch.utils.data.Dataset
        Dataset used for training.
    validation_dataset : torch.utils.data.Dataset, optional
        Dataset used for validation, by default None.
    model_config : model_config, optional
        The model configuration, by default None. It is not used during training (this function receives an already-built ``model``); it is only logged on wandb for reproducibility.
    dataset_config : dataset_config, optional
        The dataset configuration, by default None. Same logic as ``model_config``.

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
    # The only cycle-specific part is which training function runs inside the run, which is why train_model is passed as first argument.
    return wandb_utils.run_training_with_wandb(
        train_model, training_config, model, train_dataset, validation_dataset,
        model_config   = model_config,
        dataset_config = dataset_config,
    )
