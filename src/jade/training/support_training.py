"""
Support functions shared across the training modules.

This module holds the code that is used by more than one training file (or that would make the other files too long/noisy to read):

* the factories that turn a config into the corresponding torch object (:func:`get_dataloader`, :func:`get_optimizer`, :func:`get_lr_scheduler`, :func:`get_loss_function`),
* the helpers used during training (:func:`backup_every_epoch`, :func:`update_log_dict_metrics`, :func:`set_seed`),
* a few (optional) memory-tracking utilities.

The configs are kept torch-free (so they stay lightweight and serializable); this module is the place where a config is actually materialized into a :mod:`torch` object.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Full module imports
import gc
import logging
import os
import time

# Specific imports
from threading import Thread

# Third party imports
import numpy as np
import torch

# Internal imports
from jade.config.training import training_config as training_config_class
from jade.config.dataloader import dataloader_config
from jade.config.optimizers import get_optimizer_config
from jade.config.lr_schedulers import get_lr_scheduler_config
from jade.config.loss_functions import get_loss_config
from jade.config.model import get_model_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module logger

logger = logging.getLogger(__name__)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dataloader factory

def get_dataloader(dataset, batch_size : int, dataloader_config_to_use : dataloader_config = None) -> torch.utils.data.DataLoader :
    """
    Create and return the dataloader for a dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to wrap.
    batch_size : int
        Mini-batch size.
    dataloader_config_to_use : dataloader_config or dict or None, optional
        Dataloader configuration. If ``None`` a default :class:`~jade.config.dataloader.dataloader_config` is used; a dict is promoted to a :class:`~jade.config.dataloader.dataloader_config`.

    Returns
    -------
    torch.utils.data.DataLoader
        The dataloader.
    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Normalize the config (accept None / dict / dataloader_config)

    if dataloader_config_to_use is None :
        dataloader_config_to_use = dataloader_config()
    elif isinstance(dataloader_config_to_use, dict) :
        dataloader_config_to_use = dataloader_config.from_dict(dataloader_config_to_use)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Build the dataloader

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size         = batch_size,
        shuffle            = dataloader_config_to_use.shuffle,
        num_workers        = dataloader_config_to_use.num_workers,
        prefetch_factor    = dataloader_config_to_use.prefetch_factor,
        pin_memory         = dataloader_config_to_use.pin_memory,
        persistent_workers = dataloader_config_to_use.persistent_workers,
    )

    return loader

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Optimizer factory

def get_optimizer(optimizer_config_to_use, model : torch.nn.Module) -> torch.optim.Optimizer :
    """
    Create the torch optimizer described by an optimizer config.

    Parameters
    ----------
    optimizer_config_to_use : optimizer_config or dict
        The optimizer configuration (or a dict promoted through :func:`~jade.config.optimizers.get_optimizer_config`).
        The concrete optimizer is chosen from the ``name`` field.
    model : torch.nn.Module
        The model whose parameters are optimized.

    Returns
    -------
    torch.optim.Optimizer
        The optimizer.
    """

    # Accept both a config instance and a plain dict.
    config = get_optimizer_config(optimizer_config_to_use)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Dispatch on the optimizer name

    if config.name == "adam" :
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr                     = config.lr,
            betas                  = tuple(config.betas),
            eps                    = config.eps,
            weight_decay           = config.weight_decay,
            decoupled_weight_decay = config.decoupled_weight_decay,
            amsgrad                = config.amsgrad,
            maximize               = config.maximize,
        )
    elif config.name == "adamw" :
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = config.lr,
            betas        = tuple(config.betas),
            eps          = config.eps,
            weight_decay = config.weight_decay,
            amsgrad      = config.amsgrad,
            maximize     = config.maximize,
        )
    elif config.name == "sgd" :
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr           = config.lr,
            momentum     = config.momentum,
            dampening    = config.dampening,
            weight_decay = config.weight_decay,
            nesterov     = config.nesterov,
            maximize     = config.maximize,
        )
    elif config.name == "rmsprop" :
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr           = config.lr,
            alpha        = config.alpha,
            eps          = config.eps,
            weight_decay = config.weight_decay,
            momentum     = config.momentum,
            centered     = config.centered,
        )
    elif config.name == "lbfgs" :
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr               = config.lr,
            max_iter         = config.max_iter,
            max_eval         = config.max_eval,
            tolerance_grad   = config.tolerance_grad,
            tolerance_change = config.tolerance_change,
            history_size     = config.history_size,
            line_search_fn   = config.line_search_fn,
        )
    else :
        # This should never happen (the config validates the name), but keep a clear error just in case.
        raise ValueError("unknown optimizer name {}".format(repr(config.name)))

    return optimizer

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Learning-rate scheduler factory

def get_lr_scheduler(lr_scheduler_config_to_use, optimizer : torch.optim.Optimizer) :
    """
    Create the torch learning-rate scheduler described by a scheduler config.

    Parameters
    ----------
    lr_scheduler_config_to_use : lr_scheduler_config or dict
        The scheduler configuration (or a dict promoted through :func:`~jade.config.lr_schedulers.get_lr_scheduler_config`).
        The concrete scheduler is chosen from the ``name`` field.
    optimizer : torch.optim.Optimizer
        The optimizer whose learning rate is scheduled.

    Returns
    -------
    torch.optim.lr_scheduler.LRScheduler
        The learning-rate scheduler.
    """

    # Accept both a config instance and a plain dict.
    config = get_lr_scheduler_config(lr_scheduler_config_to_use)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Dispatch on the scheduler name

    if config.name == "exponential_lr" :
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = config.gamma)
    elif config.name == "step_lr" :
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = config.step_size, gamma = config.gamma)
    elif config.name == "cosine_annealing_lr" :
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config.T_max, eta_min = config.eta_min)
    elif config.name == "cosine_annealing_warm_restarts" :
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = config.T_0, T_mult = config.T_mult, eta_min = config.eta_min)
    elif config.name == "cyclic_lr" :
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr        = config.base_lr,
            max_lr         = config.max_lr,
            step_size_up   = config.step_size_up,
            step_size_down = config.step_size_down,
            mode           = config.mode,
            gamma          = config.gamma,
            cycle_momentum = config.cycle_momentum,
        )
    else :
        # This should never happen (the config validates the name), but keep a clear error just in case.
        raise ValueError("unknown lr scheduler name {}".format(repr(config.name)))

    return lr_scheduler

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Loss-function factory

def get_loss_function(loss_config_to_use) -> torch.nn.Module :
    """
    Create the torch loss module described by a loss config.

    Parameters
    ----------
    loss_config_to_use : loss_config or dict
        The loss configuration (or a dict promoted through :func:`~jade.config.loss_functions.get_loss_config`).
        The concrete loss is chosen from the ``name`` field.

    Returns
    -------
    torch.nn.Module
        The loss module. Remember that :class:`torch.nn.CrossEntropyLoss` expects logits, while :class:`torch.nn.NLLLoss` expects log-probabilities.
    """

    # Accept both a config instance and a plain dict.
    config = get_loss_config(loss_config_to_use)

    # The class weights are stored as a plain list in the config; turn them into a tensor for torch.
    weight = torch.tensor(config.weight, dtype = torch.float) if config.weight is not None else None

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Dispatch on the loss name

    if config.name == "cross_entropy" :
        loss_function = torch.nn.CrossEntropyLoss(
            weight          = weight,
            ignore_index    = config.ignore_index,
            reduction       = config.reduction,
            label_smoothing = config.label_smoothing,
        )
    elif config.name == "nll" :
        loss_function = torch.nn.NLLLoss(
            weight       = weight,
            ignore_index = config.ignore_index,
            reduction    = config.reduction,
        )
    else :
        # This should never happen (the config validates the name), but keep a clear error just in case.
        raise ValueError("unknown loss name {}".format(repr(config.name)))

    return loss_function

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Cross checks between configs

def cross_check_config(training_config, model_config) -> None :
    """
    Check that a training config and a model config are mutually consistent.

    Some parameters live in different configs but must agree with each other; this is the single place where those cross-config constraints are verified, so :func:`~jade.training.classifier.train` can fail fast with a clear message before building or training anything.

    The checks currently enforced are:

    * **loss vs last-layer activation** - a ``nll`` loss expects log-probabilities, so the classifier must end with a :class:`torch.nn.LogSoftmax` (model ``use_softmax = True``); a ``cross_entropy`` loss expects raw logits, so it must end with a :class:`torch.nn.Identity` (model ``use_softmax = False``). See :func:`get_loss_function` and :class:`~jade.model.EEGNet.EEGNet_Classifier`.
    * **class weights vs number of classes** - when the loss carries a per-class ``weight`` list, its length must equal the model ``n_classes``.

    New cross-config constraints should be added here.

    Parameters
    ----------
    training_config : training_config or dict
        The training configuration (an instance of :class:`~jade.config.training.training_config`; a dict is accepted and promoted automatically).
    model_config : model_config or dict
        The model configuration (an instance of a :class:`~jade.config.model.generic.model_config` subclass; a dict is accepted and promoted automatically).

    Raises
    ------
    ValueError
        If any cross-config constraint is violated.
    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Normalize the inputs (accept a dict or an already-built config)

    if isinstance(training_config, dict) :
        training_config = training_config_class.from_dict(training_config)
    model_config = get_model_config(model_config)

    # The loss name is the discriminator carried by the loss config.
    loss_name = training_config.loss.name

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Loss vs last-layer activation

    # use_softmax is a classifier concern: a model that does not expose it (e.g. a pure feature extractor) simply skips this check.
    use_softmax = getattr(model_config, "use_softmax", None)
    if use_softmax is not None :
        if loss_name == "nll" and not use_softmax :
            raise ValueError("loss 'nll' expects log-probabilities: build the model with use_softmax = True (LogSoftmax head)")
        if loss_name == "cross_entropy" and use_softmax :
            raise ValueError("loss 'cross_entropy' expects raw logits: build the model with use_softmax = False (Identity head)")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Class weights vs number of classes

    # A per-class weight list must have exactly one entry per class.
    weight = getattr(training_config.loss, "weight", None)
    n_classes = getattr(model_config, "n_classes", None)
    if weight is not None and n_classes is not None and len(weight) != n_classes :
        raise ValueError("the loss has {} class weights but the model has {} classes".format(len(weight), n_classes))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Helpers used during training

def backup_every_epoch(path_to_save : str, epoch : int, model, optimizer = None, lr_scheduler = None) -> None :
    """
    Backup the model (and, optionally, the optimizer and the lr scheduler) every epoch.

    For even epochs the files are saved with the suffix ``_backup_A``, for odd epochs with ``_backup_B``.
    The reason is simply to have a "backup of the backup": training on an HPC it may happen that the run is interrupted while a file is being written, corrupting it.
    Keeping the last two epochs (A and B) guarantees that at least one file is intact and can be used to restart the training, which is especially handy for long runs split across several jobs.

    Parameters
    ----------
    path_to_save : str
        Folder where the backup files are written.
    epoch : int
        Current epoch (its parity selects the A/B slot).
    model : torch.nn.Module
        The model to back up.
    optimizer : torch.optim.Optimizer, optional
        The optimizer to back up, by default None (not saved).
    lr_scheduler : torch.optim.lr_scheduler.LRScheduler, optional
        The scheduler to back up, by default None (not saved).
    """

    suffix = "_backup_A" if epoch % 2 == 0 else "_backup_B"

    # Save the model
    path_to_save_model = os.path.join(path_to_save, "model{}.pth".format(suffix))
    torch.save(model.state_dict(), path_to_save_model)

    # (OPTIONAL) Save the optimizer
    if optimizer is not None :
        path_to_save_optimizer = os.path.join(path_to_save, "optimizer{}.pth".format(suffix))
        torch.save(optimizer.state_dict(), path_to_save_optimizer)

    # (OPTIONAL) Save the lr scheduler
    if lr_scheduler is not None :
        path_to_save_lr_scheduler = os.path.join(path_to_save, "lr_scheduler{}.pth".format(suffix))
        torch.save(lr_scheduler.state_dict(), path_to_save_lr_scheduler)


def update_log_dict_metrics(metrics_dict : dict, log_dict : dict, label : str = None) -> None :
    """
    Copy the metrics of ``metrics_dict`` into ``log_dict`` (the dict logged on wandb).

    The keys of ``metrics_dict`` are added to ``log_dict``, optionally prefixed/suffixed with ``label``.
    For example, with ``metrics_dict = {'accuracy' : 0.9}`` and ``label = 'train'`` the key ``'accuracy_train'`` is added; with ``label = None`` the key ``'accuracy'`` is added as is.

    Parameters
    ----------
    metrics_dict : dict
        The metrics to copy.
    log_dict : dict
        The destination dict (updated in place).
    label : str, optional
        Suffix appended to every key, by default None (keys copied unchanged).
    """

    for key, value in metrics_dict.items() :
        if label is not None :
            log_dict["{}_{}".format(key, label)] = value
        else :
            log_dict["{}".format(key)] = value


def set_seed(seed : int) -> None :
    """
    Seed numpy and torch (CPU and CUDA) for reproducibility.

    Parameters
    ----------
    seed : int
        The seed to use.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() :
        torch.cuda.manual_seed_all(seed)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Memory stats
# The following utilities are optional helpers to profile the memory footprint of a piece of code.

def cpu_memory_usage_in_gb(func, *args, **kwargs) -> float :
    """
    Track the peak CPU memory (RSS) used while running ``func``.
    Copied from https://www.analyticsvidhya.com/blog/2024/10/memory-efficient-model-weight-loading-in-pytorch/

    Parameters
    ----------
    func : callable
        The function to run and profile.
    *args, **kwargs
        Arguments forwarded to ``func``.

    Returns
    -------
    float
        The peak memory usage (in GB) above the baseline.
    """

    # psutil is imported lazily so it is only required when this profiling helper is actually used.
    import psutil

    process = psutil.Process(os.getpid())
    baseline_mem = process.memory_info().rss / 1024 ** 3  # in GB
    mem_usage = []
    done = False

    def monitor_memory() :
        while not done :
            mem_usage.append(process.memory_info().rss / 1024 ** 3)  # Convert to GB
            time.sleep(0.1)

    monitor_thread = Thread(target = monitor_memory)
    monitor_thread.start()

    func(*args, **kwargs)
    done = True
    monitor_thread.join()

    peak_mem_usage_gb = max(mem_usage) - baseline_mem
    return peak_mem_usage_gb


def gpu_start_memory_tracking() -> None :
    """
    Reset the CUDA peak-memory counter (start of a tracking window).
    Copied from https://www.analyticsvidhya.com/blog/2024/10/memory-efficient-model-weight-loading-in-pytorch/
    """

    if torch.cuda.is_available() :
        torch.cuda.reset_peak_memory_stats()
    else :
        logger.warning("GPU memory tracking requested but CUDA is not available.")


def gpu_print_memory_usage() -> None :
    """
    Log the maximum GPU memory allocated so far.
    Copied from https://www.analyticsvidhya.com/blog/2024/10/memory-efficient-model-weight-loading-in-pytorch/
    """

    max_gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert bytes to GB
    logger.info("Maximum GPU memory allocated: %.1f GB", max_gpu_memory)


def gpu_cleanup() -> None :
    """
    Force a garbage collection and empty the CUDA cache.
    Copied from https://www.analyticsvidhya.com/blog/2024/10/memory-efficient-model-weight-loading-in-pytorch/
    """

    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)  # Allow time for memory to clear
    torch.cuda.reset_peak_memory_stats()
    max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
    logger.info("Maximum GPU memory allocated: %.1f GB", max_memory_allocated)
