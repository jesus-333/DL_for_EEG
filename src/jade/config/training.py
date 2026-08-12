"""
Training configuration.

:class:`~jade.config.training.training_config` gathers every parameter consumed by the classification training loop in :mod:`jade.training.classifier`.
It follows the same design as the other configs of the package: a :func:`dataclasses.dataclass` inheriting from :class:`~jade.config.generic.generic_config`, with validation in ``__post_init__`` and nested configs promoted from plain dicts (exactly like :class:`~jade.config.dataset.dataset_config` promotes its ``filter`` field).

The optimizer, the (optional) learning-rate scheduler, the loss function, the dataloader and the (optional) wandb logging are all nested configs:

* ``optimizer``    -> a :class:`~jade.config.optimizers.generic.optimizer_config` (Adam by default)
* ``lr_scheduler`` -> an :class:`~jade.config.lr_schedulers.generic.lr_scheduler_config` or ``None`` (no scheduling)
* ``loss``         -> a :class:`~jade.config.loss_functions.generic.loss_config` (cross-entropy by default)
* ``dataloader``   -> a :class:`~jade.config.dataloader.dataloader_config`
* ``wandb``        -> a :class:`~jade.config.wandb_training.wandb_training_config` or ``None`` (no wandb logging)

Serialized to TOML, each nested config becomes its own section (e.g. ``[optimizer]``, ``[loss]``, ``[wandb]``).
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from dataclasses import dataclass, field
from typing import Optional

# Internal imports
from jade.config.generic import generic_config
from jade.config.dataloader import dataloader_config
from jade.config.wandb_training import wandb_training_config
from jade.config.optimizers import optimizer_config, adam_config, get_optimizer_config
from jade.config.lr_schedulers import lr_scheduler_config, get_lr_scheduler_config
from jade.config.loss_functions import loss_config, cross_entropy_config, get_loss_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Training config class

@dataclass
class training_config(generic_config) :
    """
    Hyper-parameters and options of a classification training run.

    Parameters
    ----------
    epochs : int, default 100
        Number of training epochs. Must be positive.
    batch_size : int, default 32
        Mini-batch size passed to every dataloader. Must be positive.
    device : str, default 'cpu'
        Device the training runs on (e.g. ``'cpu'``, ``'cuda'``, ``'cuda:0'``).
    optimizer : optimizer_config, default adam_config()
        Optimizer configuration (nested). A plain dict is promoted to the proper
        :class:`~jade.config.optimizers.generic.optimizer_config` subclass.
    lr_scheduler : lr_scheduler_config or None, default None
        Learning-rate scheduler configuration (nested). ``None`` disables scheduling.
        The scheduler is stepped once per epoch.
    loss : loss_config, default cross_entropy_config()
        Loss-function configuration (nested). Remember to pair it with the model's output:
        cross-entropy for logits (``use_softmax = False``), NLL for log-probabilities (``use_softmax = True``).
    dataloader : dataloader_config, default dataloader_config()
        Dataloader configuration (nested). ``batch_size`` is taken from this config's parent, not from here.
    wandb : wandb_training_config or None, default None
        Wandb logging configuration (nested). ``None`` disables wandb logging.
        Use :func:`~jade.training.classifier.wandb_train` (from configs) or :func:`~jade.training.classifier.wandb_train_model` (from an already-built model) to actually open the wandb run.
    path_to_save_model : str, default 'model_weights'
        Folder where the model checkpoints are written. Created if it does not exist.
    epoch_to_save_model : int, default 0
        Period (in epochs) of the periodic checkpoints. A value ``<= 0`` disables them, so the model is saved only at the end of the training (``model_END.pth``) and, if a validation set is given, whenever the validation loss improves (``model_BEST.pth``).
    backup_model_every_epoch : bool, default False
        If True, an alternating A/B backup of the model (and optimizer/scheduler) is written every epoch (see :func:`~jade.training.support_training.backup_every_epoch`).
    measure_metrics_during_training : bool, default True
        If True, classification metrics (accuracy, F1, ...) are computed at every epoch.
    print_var : bool, default True
        If True, the per-epoch losses/metrics are logged at INFO level.
    seed : int or None, default None
        Optional random seed. If given, must be non-negative; it is used to seed the run
        (see :func:`~jade.training.support_training.set_seed`) and as the wandb run id.
    path_past_weights : str or None, default None
        Optional path to a ``state_dict`` to load into the model before training (to resume a past training).
    path_past_optimizer : str or None, default None
        Optional path to a ``state_dict`` to load into the optimizer before training.
    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Core hyper-parameters

    epochs     : int = 100
    batch_size : int = 32
    device     : str = "cpu"

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Nested configs (optimizer / scheduler / loss / dataloader / wandb)

    optimizer    : optimizer_config                = field(default_factory = adam_config)
    lr_scheduler : Optional[lr_scheduler_config]   = None
    loss         : loss_config                     = field(default_factory = cross_entropy_config)
    dataloader   : dataloader_config               = field(default_factory = dataloader_config)
    wandb        : Optional[wandb_training_config] = None

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Checkpoint / logging options

    path_to_save_model              : str = "model_weights"
    epoch_to_save_model             : int = 0
    backup_model_every_epoch        : bool = False
    measure_metrics_during_training : bool = True
    print_var                       : bool = True
    seed                            : Optional[int] = None

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Optional training resume

    path_past_weights   : Optional[str] = None
    path_past_optimizer : Optional[str] = None


    def __post_init__(self) -> None :
        """
        Validate the training parameters and promote every nested config.
        """

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Core hyper-parameters (bool is rejected where an int is expected, being a subclass of int)

        if not isinstance(self.epochs, int) or isinstance(self.epochs, bool) or self.epochs <= 0 :
            raise ValueError("epochs must be a positive int, got {}".format(repr(self.epochs)))

        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size <= 0 :
            raise ValueError("batch_size must be a positive int, got {}".format(repr(self.batch_size)))

        if not isinstance(self.device, str) or not self.device :
            raise ValueError("device must be a non-empty string, got {}".format(repr(self.device)))

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Promote the nested configs (a dict coming from a file becomes the proper dataclass)

        # Optimizer (polymorphic: the concrete subclass is chosen from its 'name').
        self.optimizer = get_optimizer_config(self.optimizer)

        # Learning-rate scheduler (optional).
        if self.lr_scheduler is not None :
            self.lr_scheduler = get_lr_scheduler_config(self.lr_scheduler)

        # Loss function (polymorphic).
        self.loss = get_loss_config(self.loss)

        # Dataloader.
        if isinstance(self.dataloader, dict) :
            self.dataloader = dataloader_config.from_dict(self.dataloader)
        elif not isinstance(self.dataloader, dataloader_config) :
            raise ValueError("dataloader must be a dataloader_config (or a dict), got {}".format(type(self.dataloader)))

        # Wandb logging (optional).
        if self.wandb is not None :
            if isinstance(self.wandb, dict) :
                self.wandb = wandb_training_config.from_dict(self.wandb)
            elif not isinstance(self.wandb, wandb_training_config) :
                raise ValueError("wandb must be a wandb_training_config (or a dict) or None, got {}".format(type(self.wandb)))

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Checkpoint / logging options

        # A non-positive period disables the periodic checkpoints: pick a period the epoch loop can never hit,
        # so only model_END.pth (and model_BEST.pth, when validating) are written.
        if not isinstance(self.epoch_to_save_model, int) or isinstance(self.epoch_to_save_model, bool) or self.epoch_to_save_model <= 0 :
            self.epoch_to_save_model = self.epochs + 1

        # Seed
        if self.seed is not None :
            if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0 :
                raise ValueError("seed must be a non-negative int or None, got {}".format(repr(self.seed)))
