"""
L-BFGS optimizer configuration.

:class:`~jade.config.optimizers.lbfgs.lbfgs_config` mirrors the parameters of :class:`torch.optim.LBFGS`.
Every parameter defaults to torch's value; the optimizer is built from this config by :func:`~jade.training.support_training.get_optimizer`.

Note that L-BFGS is a full-batch optimizer that needs a ``closure`` re-evaluating the model; the standard mini-batch training loop is not tailored for it, so use it only if you know what you are doing.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

# Specific imports
from dataclasses import dataclass
from typing import Optional

# Internal imports
from jade.config.optimizers.generic import optimizer_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# L-BFGS config class

@dataclass
class lbfgs_config(optimizer_config) :
    """
    Configuration for :class:`torch.optim.LBFGS`.

    Parameters
    ----------
    lr : float, default 1.0
        Learning rate. Must be positive.
    max_iter : int, default 20
        Maximum number of iterations per optimization step. Must be positive.
    max_eval : int or None, default None
        Maximum number of function evaluations per step. ``None`` lets torch use ``max_iter * 1.25``.
    tolerance_grad : float, default 1e-7
        Termination tolerance on the first order optimality.
    tolerance_change : float, default 1e-9
        Termination tolerance on the function value/parameter changes.
    history_size : int, default 100
        Update history size. Must be positive.
    line_search_fn : str or None, default None
        Either ``'strong_wolfe'`` or ``None``.
    """

    name : str = "lbfgs"

    lr               : float = 1.0
    max_iter         : int = 20
    max_eval         : Optional[int] = None
    tolerance_grad   : float = 1e-7
    tolerance_change : float = 1e-9
    history_size     : int = 100
    line_search_fn   : Optional[str] = None


    def __post_init__(self) -> None :
        """
        Validate the L-BFGS parameters and freeze the optimizer name.
        """

        # Freeze the discriminator so the config always serializes as an L-BFGS config.
        self.name = "lbfgs"

        # Learning rate
        if self.lr <= 0 :
            raise ValueError("lr must be positive, got {}".format(repr(self.lr)))

        # Iteration/history counters
        if self.max_iter <= 0 :
            raise ValueError("max_iter must be positive, got {}".format(repr(self.max_iter)))
        if self.history_size <= 0 :
            raise ValueError("history_size must be positive, got {}".format(repr(self.history_size)))

        # Line search function
        if self.line_search_fn is not None and self.line_search_fn != "strong_wolfe" :
            raise ValueError("line_search_fn must be 'strong_wolfe' or None, got {}".format(repr(self.line_search_fn)))
