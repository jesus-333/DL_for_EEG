"""
Generic test/evaluation function for a classifier model.

:func:`test` evaluates an (already trained) model on a dataset: it computes the loss and, optionally, the classification metrics, and returns them in a log dict ready to be logged on wandb.
It reuses :func:`~jade.training.classifier_epoch.validation_epoch_function` for the forward pass and loss, and :func:`~jade.training.metrics.compute_metrics` for the metrics.

The ``label`` argument prefixes/suffixes every logged key, so the same function can be reused to evaluate on different sets under different names (e.g. ``'test'`` here, or ``'server'`` for the central-server evaluation of a future federated-learning setting).

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Full module imports
import logging

# Internal imports
from jade.config.training import training_config as training_config_class
from jade.training import support_training
from jade.training import metrics
from jade.training.classifier_epoch import validation_epoch_function

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module logger

logger = logging.getLogger(__name__)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Test function

def test(training_config, model, test_dataset, label : str = "test") :
    """
    Evaluate a model on a dataset and return the loss and the metrics.

    Parameters
    ----------
    training_config : training_config or dict
        The training configuration. Only a few fields are used here: ``batch_size``, ``device``, ``dataloader``, ``loss`` and ``measure_metrics_during_training``.
        An instance of :class:`~jade.config.training.training_config` (a dict is accepted and promoted automatically).
    model : torch.nn.Module
        The model to evaluate.
    test_dataset : torch.utils.data.Dataset
        Dataset to evaluate on. Each item must be a ``(data, label)`` pair.
    label : str, optional
        Name used to save the loss and the metrics, by default ``'test'``.
        For example, with ``label = 'test'`` the loss is stored under ``'test_loss'`` and a metric ``accuracy`` under ``'accuracy_test'``.

    Returns
    -------
    test_loss : float
        The average loss over the test set.
    log_dict : dict
        Dictionary with the loss and (optionally) the metrics, ready to be logged. Note that this function does not log anything on wandb itself; the caller is expected to do it.
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Setup for the test

    # Accept a plain dict and promote it to a proper training_config.
    if isinstance(training_config, dict) :
        training_config = training_config_class.from_dict(training_config)

    device = training_config.device

    # Create the dataloader for the test set.
    test_loader = support_training.get_dataloader(test_dataset, training_config.batch_size, training_config.dataloader)

    # Move the model to the device.
    model.to(device)

    # Create the loss function (same config as the training).
    loss_function = support_training.get_loss_function(training_config.loss)
    loss_function.to(device)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Forward pass and loss computation on the test set

    log_dict = {}
    test_loss = validation_epoch_function(model, test_loader, loss_function, device, log_dict)

    # Save the test loss under the correct key.
    # Note that validation_epoch_function saves the loss under the key 'validation_loss', so it is renamed here.
    del log_dict["validation_loss"]
    log_dict["{}_loss".format(label)] = test_loss

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # (OPTIONAL) Compute the classification metrics (accuracy, F1, ...)

    if training_config.measure_metrics_during_training :
        # Compute the metrics.
        test_metrics_dict = metrics.compute_metrics(model, test_loader, device)

        # Save the metrics in the log dict (with the given label as suffix).
        support_training.update_log_dict_metrics(test_metrics_dict, log_dict, label = label)

    return test_loss, log_dict
