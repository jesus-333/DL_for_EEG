"""
Logging utilities.

Helpers to configure Python's :mod:`logging` for people who do not want to set up their own logger.

The rest of ``jade`` follows the standard library-logging best practice: every module logs through ``logging.getLogger(__name__)`` and the :mod:`jade.training` package attaches a :class:`logging.NullHandler`, so importing the package never configures logging on its own.
:func:`~jade.training.logging_utils.setup_logging` is the opt-in convenience that wires up a sensible root configuration (a console handler, an optional file handler and a readable format) in a single call.

For now this module only exposes :func:`~jade.training.logging_utils.setup_logging`, but it is meant to collect any future logging-related helper.

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

# Specific imports
from typing import Optional

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Module constants

# Default log record format and date format, used by setup_logging when the caller does not provide their own.
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Logging setup

def setup_logging(level : int = logging.INFO, logfile : Optional[str] = None, format : Optional[str] = None, datefmt : Optional[str] = None) -> None :
    """
    Configure the root logger with a console handler (and, optionally, a file handler).

    This is a convenience for people who do not want to set up their own logger: it wires a sensible, readable configuration in a single call.
    It relies on ``logging.basicConfig(..., force = True)``, so it can be re-called (e.g. in a notebook) without stacking duplicate handlers.
    Since it configures the root logger, every ``jade`` module (each logging through ``logging.getLogger(__name__)``) is captured automatically.

    Parameters
    ----------
    level : int, default logging.INFO
        The logging level of the root logger (e.g. ``logging.DEBUG``, ``logging.INFO``, ``logging.WARNING``).
    logfile : str or None, default None
        Path to a file where the logs are written too.
        ``None`` (the default) logs only to the console (``stdout``).
        When a path is given, its parent directory is created if it does not exist yet.
    format : str or None, default None
        The log record format string (see :class:`logging.Formatter`).
        ``None`` (the default) uses :data:`DEFAULT_LOG_FORMAT` (``"%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"``).
    datefmt : str or None, default None
        The date/time format string used by ``%(asctime)s`` (see :func:`time.strftime`).
        ``None`` (the default) uses :data:`DEFAULT_DATE_FORMAT` (``"%Y-%m-%d %H:%M:%S"``).
    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Resolve the (optional) format strings

    # When the caller does not pass a format, fall back to the package defaults.
    if format is None : format = DEFAULT_LOG_FORMAT
    if datefmt is None : datefmt = DEFAULT_DATE_FORMAT

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Build the handlers

    # Always log to stdout.
    handlers = [logging.StreamHandler(sys.stdout)]

    # Optionally also log to a file (creating its parent directory if needed).
    if logfile is not None :
        log_directory = os.path.dirname(logfile)
        if log_directory : os.makedirs(log_directory, exist_ok = True)
        handlers.append(logging.FileHandler(logfile))

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Configure the root logger

    # force = True removes any handler already attached to the root logger, so re-calling this function (e.g. in a notebook) does not stack duplicate handlers.
    logging.basicConfig(
        level    = level,
        format   = format,
        datefmt  = datefmt,
        handlers = handlers,
        force    = True,
    )
