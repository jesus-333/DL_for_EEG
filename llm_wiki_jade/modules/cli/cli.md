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
create-default-config = "jade.cli.cli:default_config"
download-project-runs-metrics = "jade.cli.cli:download_project_runs_metrics"
```
Multi-word command names use hyphens (`download-dataset-moabb`, `create-default-config`, `download-project-runs-metrics`), per the coding style.

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

---

## `default_config() -> None`

Entry point of the `create-default-config` command.

1. Lazily imports `jade.cli.default_config.default_config` (the implementation, same name).
2. Builds an `argparse.ArgumentParser` (`prog="create-default-config"`) with:

   | Argument | Type | Required / default | Meaning |
   |----------|------|--------------------|---------|
   | `--path` | str | default `"."` | Directory where the `default_config` folder is created. |
   | `--type` | str (`choices=["train","model","dataset","all"]`) | default `"all"` | Which configs to create. |

3. `parser.parse_args()` → `args`, then calls `default_config(args)` (the implementation).

Like the other entry point it takes **no Python arguments**. See [`default_config.md`](./default_config.md) for the implementation (which copies the packaged TOML skeleton tree).

---

## `download_project_runs_metrics() -> None`

Entry point of the `download-project-runs-metrics` command.

1. Lazily imports `jade.cli.wandb.download_project_runs_metrics` (the implementation, same name).
2. Builds an `argparse.ArgumentParser` (`prog="download-project-runs-metrics"`) with:

   | Argument | Type | Required / default | Meaning |
   |----------|------|--------------------|---------|
   | `--entity` | str | required | wandb entity (user or team) that owns the project. |
   | `--project` | str | required | wandb project whose runs metrics are downloaded. |
   | `--path_save` | str | default `None` | output JSON path; `None` → `./wandb_metrics.json`. |
   | `--timeout` | int | default `60` | wandb public-API GraphQL timeout (s); raises the 19 s wandb default that times out on many-run projects. |

3. `parser.parse_args()` → `args`, then calls `download_project_runs_metrics(args)` (the implementation).

Like the other entry points it takes **no Python arguments**. See [`wandb.md`](./wandb.md) for the implementation (which uses the wandb public API and reconstructs the run-history helpers it needs locally).
