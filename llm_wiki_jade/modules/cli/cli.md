# `jade.cli.cli`

Source: `src/jade/cli/cli.py`

The **entry-point module** of the cli: the single place where the `jade` console-script functions are declared. Every function here builds *only* its `argparse` parser and then hands the parsed arguments to the matching implementation in a thematic file of the subpackage. By convention the entry-point function and its implementation **share the same name**.

## Imports & load cost
- Module level: `from __future__ import annotations`, `import argparse`. Nothing else — importing `cli.py` is cheap.
- Each entry-point function imports its implementation **lazily** (inside the function body), so the heavy `jade` code (and its optional deps) is only loaded when the command actually runs.

## Console-script wiring
Each function is exposed as a terminal command through the `[project.scripts]` table of `pyproject.toml`:
```toml
[project.scripts]
download-dataset-moabb = "jade.cli.cli:download_dataset_moabb"
```
Multi-word command names use hyphens (`download-dataset-moabb`), per the coding style.

---

## `download_dataset_moabb() -> None`

Entry point of the `download-dataset-moabb` command.

1. Lazily imports `jade.cli.download_dataset.download_dataset_moabb` (the implementation, same name).
2. Builds an `argparse.ArgumentParser` (`prog="download-dataset-moabb"`) with:

   | Argument | Type | Required / default | Meaning |
   |----------|------|--------------------|---------|
   | `--dataset_name` | str | required | moabb dataset name (case-insensitive). |
   | `--subject` | int | default `None` | single subject; `None` → all subjects. |
   | `--dataset_folder` | str | default `None` | download/look-up folder; `None` → moabb default `~/mne_data/`. |

3. `parser.parse_args()` → `args`, then calls `download_dataset_moabb(args)` (the implementation).

Takes **no Python arguments** — it reads `sys.argv` through `argparse`, as befits a console-script entry point. See [`download_dataset.md`](./download_dataset.md) for the implementation it delegates to.
