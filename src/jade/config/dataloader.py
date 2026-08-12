"""
Dataloader configuration.

:class:`~jade.config.dataloader.dataloader_config` collects the parameters forwarded to :class:`torch.utils.data.DataLoader` when the training loop wraps a dataset (see :func:`~jade.training.support_training.get_dataloader`).
Note that ``batch_size`` is NOT here: it lives in :class:`~jade.config.training.training_config`, since it is a training hyper-parameter shared by every loader.

A few torch parameters constrain each other (``prefetch_factor`` and ``persistent_workers`` only make sense when ``num_workers > 0``); ``__post_init__`` gently fixes the inconsistent combinations and logs a warning instead of raising, mirroring the behavior of the old code.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Full module imports
import logging

# Specific imports
from dataclasses import dataclass
from typing import Optional

# Internal imports
from jade.config.generic import generic_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module logger

logger = logging.getLogger(__name__)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dataloader config class

@dataclass
class dataloader_config(generic_config) :
    """
    Parameters forwarded to :class:`torch.utils.data.DataLoader`.

    Parameters
    ----------
    num_workers : int, default 0
        Number of subprocesses used for data loading. ``0`` means the data is loaded in the main process.
    prefetch_factor : int or None, default None
        Number of batches prefetched per worker. Must be ``None`` when ``num_workers = 0`` (torch requirement); when ``num_workers > 0`` and it is left ``None`` it is set to ``2``.
    pin_memory : bool, default False
        If True, the loader copies tensors into pinned (page-locked) memory before returning them.
    persistent_workers : bool, default False
        If True, the worker processes are kept alive between epochs. Only valid when ``num_workers > 0``.
    shuffle : bool, default True
        Whether to reshuffle the data at every epoch.
    """

    num_workers        : int = 0
    prefetch_factor    : Optional[int] = None
    pin_memory         : bool = False
    persistent_workers : bool = False
    shuffle            : bool = True


    def __post_init__(self) -> None :
        """
        Validate the dataloader parameters and fix the inconsistent combinations.
        """

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Number of workers (bool is rejected, being a subclass of int)

        if not isinstance(self.num_workers, int) or isinstance(self.num_workers, bool) or self.num_workers < 0 :
            raise ValueError(f"num_workers must be a non-negative int, got {repr(self.num_workers)}")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Prefetch factor (only meaningful when num_workers > 0)

        if self.num_workers == 0 :
            # torch requires prefetch_factor to be None when there are no workers.
            if self.prefetch_factor is not None :
                logger.warning("prefetch_factor is only used when num_workers > 0. Forcing it to None (was %s).", self.prefetch_factor)
                self.prefetch_factor = None
        else :
            # With workers, fall back to torch's usual default of 2 if the user did not set it.
            if self.prefetch_factor is None :
                self.prefetch_factor = 2
            elif self.prefetch_factor <= 0 :
                raise ValueError(f"prefetch_factor must be positive, got {repr(self.prefetch_factor)}")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Persistent workers (only valid when num_workers > 0)

        if self.persistent_workers and self.num_workers == 0 :
            logger.warning("persistent_workers is only valid when num_workers > 0. Forcing it to False.")
            self.persistent_workers = False
