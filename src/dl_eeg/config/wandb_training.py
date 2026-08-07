"""
Weights & Biases (wandb) training configuration.

:class:`~dl_eeg.config.wandb_training.wandb_training_config` groups every parameter that is specific to logging a training run on wandb.
It is kept separate from :class:`~dl_eeg.config.training.training_config` (which references it as an optional nested field) so that a run that does not use wandb carries no wandb-related noise.

The values are consumed by :func:`~dl_eeg.training.classifier.wandb_train` (which sets up :func:`wandb.init` and the model :class:`wandb.Artifact`) and by :func:`~dl_eeg.training.classifier.train` (which logs the per-epoch metrics).
Note that ``wandb`` is an optional dependency of the package; it only needs to be installed if this config is used.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from dataclasses import dataclass
from typing import Optional

# Internal imports
from dl_eeg.config.generic import generic_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Wandb training config class

@dataclass
class wandb_training_config(generic_config) :
    """
    Parameters used to log a training run on wandb.

    Parameters
    ----------
    project_name : str
        Name of the wandb project the run is logged to. Required (non-empty).
    model_artifact_name : str
        Base name of the wandb model :class:`wandb.Artifact`. Required (non-empty).
        The trained artifact is logged with the suffix ``'_trained'`` appended.
    entity : str or None, default None
        The wandb entity (user or team) that owns the project. ``None`` uses the default entity.
    name_training_run : str or None, default None
        Human-readable name of the run. ``None`` lets wandb assign a random name.
    notes : str or None, default None
        Free-text notes attached to the run.
    log_model_artifact : bool, default True
        Whether to upload the model weights as a wandb artifact.
    log_freq : int, default 1
        How often (in epochs) :func:`wandb.watch` logs the gradients/parameters. Must be positive.
    debug : bool, default False
        Not used by the training logic; only handy to quickly filter debug runs in the wandb UI.
    """

    project_name        : str = ""
    model_artifact_name : str = ""
    entity              : Optional[str] = None
    name_training_run   : Optional[str] = None
    notes               : Optional[str] = None
    log_model_artifact  : bool = True
    log_freq            : int = 1
    debug               : bool = False


    def __post_init__(self) -> None :
        """
        Validate the wandb parameters after initialization.
        """

        # Mandatory identifiers
        if not self.project_name :
            raise ValueError("project_name must be a non-empty string when wandb logging is used")
        if not self.model_artifact_name :
            raise ValueError("model_artifact_name must be a non-empty string when wandb logging is used")

        # Logging frequency (bool is rejected, being a subclass of int).
        if not isinstance(self.log_freq, int) or isinstance(self.log_freq, bool) or self.log_freq <= 0 :
            raise ValueError("log_freq must be a positive int, got {}".format(repr(self.log_freq)))
