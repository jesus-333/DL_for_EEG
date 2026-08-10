# `jade.training.logging_utils`

Source: `src/jade/training/logging_utils.py`

Logging helpers for people who do not want to configure Python's `logging` themselves. `jade` follows the standard library-logging best practice (every module logs through `logging.getLogger(__name__)`, and `jade.training` attaches a `NullHandler`, so importing never configures logging), which means **nothing is printed until the application sets up logging**. This module is the opt-in escape hatch: a single call wires a sensible root configuration.

For now it exposes only `setup_logging`, but it is meant to collect any future logging-related helper. It is **torch-free** (stdlib `logging`/`os`/`sys` only).

## `setup_logging(level = logging.INFO, logfile = None) -> None`

Configure the **root** logger with a console handler (`stdout`) and, optionally, a file handler.

| Param | Type | Meaning | Default |
|-------|------|---------|---------|
| `level` | int | Root logger level (`logging.DEBUG` / `logging.INFO` / `logging.WARNING` / …). | `logging.INFO` |
| `logfile` | str or None | If given, logs are also written to this file; its parent directory is created if it does not exist. `None` → console only. | `None` |

Behavior:
- Builds the handler list: always a `logging.StreamHandler(sys.stdout)`; when `logfile` is set, also a `logging.FileHandler(logfile)` (its parent directory is created via `os.makedirs(..., exist_ok=True)` when the path has one).
- Calls `logging.basicConfig(level=..., format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=..., force=True)`.
- **`force=True`** removes any handler already attached to the root logger, so the function can be re-called (e.g. in a notebook) **without stacking duplicate handlers**.
- Because it configures the **root** logger, every `jade` module (which logs via `logging.getLogger(__name__)` and propagates to root) is captured automatically — you see the `jade.training.*` records without touching any `jade` internals.
- Returns `None`.

Example line produced by the format: `2025-01-01 12:00:00 | INFO     | jade.training.classifier | ...`.

## When to use it (and when not)

- **Use it** for a quick script / notebook where you just want `jade`'s INFO logs on screen (and optionally in a file).
- **Do not** call it from library code or if the host application already configures logging — `force=True` would replace the application's handlers. It is a convenience for the *end user*, mirroring the `NullHandler` contract described in [`__init__.md`](__init__.md).
