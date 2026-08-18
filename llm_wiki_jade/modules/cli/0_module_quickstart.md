# `jade.cli` — Module Quickstart (READ FIRST)

> Read [`../../0_wiki_quickstart.md`](../../0_wiki_quickstart.md) first. This page is the read-first overview of the `cli` module; open a per-file page only for field-level detail.

## Purpose

`jade.cli` exposes a handful of `jade` functionalities as **console scripts** (terminal commands). It is deliberately thin: the cli **never introduces new behavior**, it only exposes / facilitates access to code that already lives elsewhere in the package. When a command needs a capability that does not exist yet, that capability is added to the appropriate module (e.g. `dataset`), *not* to the cli.

## Structure (the two-layer pattern)

```
cli/
  cli.py               <- entry points: one function per command, argparse only, then delegate
  download_dataset.py  <- thematic implementation file (receives parsed args, calls jade code)
  default_config.py    <- thematic implementation file (copies the packaged TOML skeleton configs)
  wandb.py             <- thematic implementation file (downloads a wandb project's run metrics via wandb.Api)
  ...                  <- one more thematic file per future command family
```

- **`cli.py` is the single entry point.** Each function here builds *only* its `argparse` parser and then calls the matching implementation in a thematic file. It contains no business logic. Every function in `cli.py` is the target of a console script declared in `[project.scripts]` of `pyproject.toml`.
- **Thematic files** (e.g. `download_dataset.py`) hold the implementation: they take the already-parsed `argparse.Namespace` and forward it to the existing `jade` function that does the real work.
- **Naming convention:** the entry-point function in `cli.py` and its implementation in the thematic file share the **same name** (e.g. both are `download_dataset_moabb`). Multi-word command *names* use hyphens (`download-dataset-moabb`), per the coding style.

```
$ download-dataset-moabb --dataset_name ...      (console script)
        │
        ▼
jade.cli.cli.download_dataset_moabb()            (argparse only)
        │  parses argv → args
        ▼
jade.cli.download_dataset.download_dataset_moabb(args)   (thin wrapper)
        │
        ▼
jade.dataset.download_dataset(name, subjects, dataset_folder)   (the real work)
```

## Available commands

| Command | Entry point (`cli.py`) | Implementation | Wraps |
|---------|------------------------|----------------|-------|
| `download-dataset-moabb` | `download_dataset_moabb()` | [`download_dataset.py`](./download_dataset.md) → `download_dataset_moabb(args)` | [`jade.dataset.download_dataset`](../dataset/moabb_dataset.md) |
| `create-default-config` | `default_config()` | [`default_config.py`](./default_config.md) → `default_config(args)` | copies the packaged `config/default_config/` TOML skeletons |
| `download-project-runs-metrics` | `download_project_runs_metrics()` | [`wandb.py`](./wandb.md) → `download_project_runs_metrics(args)` | the wandb public API (`wandb.Api`) — no pre-existing `jade` function |

### `download-dataset-moabb`

Pre-download one of the [datasets provided by moabb](https://moabb.neurotechx.com/docs/dataset_summary.html) so you do not have to wait for the download later.

| Argument | Type | Default | Meaning |
|----------|------|---------|---------|
| `--dataset_name` | str | required | moabb dataset name (case-insensitive), e.g. `BNCI2014_001`. |
| `--subject` | int | `None` | Single subject to download; `None` → all subjects. |
| `--dataset_folder` | str | `None` | Where to store the data; `None` → moabb default (`~/mne_data/`). Single global setting — see the gotcha in [`moabb_dataset.md`](../dataset/moabb_dataset.md). |

```bash
# all subjects, default folder
download-dataset-moabb --dataset_name BNCI2014_001

# one subject, custom folder
download-dataset-moabb --dataset_name BNCI2014_001 --subject 1 --dataset_folder /data/eeg
```

Note: moabb downloads the **full raw recordings** (every session), so there is no train/test split at download time (that split only matters in `get_dataset`).

### `create-default-config`

Create a `default_config` folder of commented TOML **skeleton** configs (every possible parameter, with reasonable values), meant to be copied and edited.

| Argument | Type | Default | Meaning |
|----------|------|---------|---------|
| `--path` | str | `.` (current dir) | Where the `default_config` folder is created. |
| `--type` | str (`train`/`model`/`dataset`/`all`) | `all` | Which configs to create. |

```bash
# everything, in the current directory
create-default-config

# only the model config, elsewhere
create-default-config --type model --path /tmp/configs
```

This is the one command that does not wrap an existing `jade` function: it **copies package data** (the TOML tree shipped under `jade/config/default_config/`). The copy logic still lives in the thematic implementation file (`default_config.py`), so `cli.py` stays argparse-only — the two-layer pattern is preserved. Multi-choice sections come as one file per option, with the main file pointing at them: optimizer/scheduler/loss (and a `wandb.toml`) under `train/` (referenced by `training_clf.toml`) and one file per filter type under `dataset/filter/` (referenced by `dataset.toml`). The optional sections (`[lr_scheduler]`, `[wandb]`, `[filter]`) ship commented-out in the main file. Those per-option files are **header-less** (no `[section]` line — only a leading comment names the section), so they load flat and can be nested directly under the parent config's key or pasted under the matching `[section]` in the main file; see [default_config.md](./default_config.md#option-files-are-header-less-no-section).

### `download-project-runs-metrics`

Download the metrics of **every run** in a wandb project and save them to a JSON file, ready for offline plotting/analysis without hitting wandb again.

| Argument | Type | Default | Meaning |
|----------|------|---------|---------|
| `--entity` | str | required | wandb entity (user or team) that owns the project. |
| `--project` | str | required | wandb project whose runs are exported. |
| `--path_save` | str | `None` | output JSON path; `None` → `./wandb_metrics.json` in the current directory. |

```bash
# every run of entity/project → ./wandb_metrics.json
download-project-runs-metrics --entity my_team --project my_project

# custom output path
download-project-runs-metrics --entity my_team --project my_project --path_save /data/metrics.json
```

The output is a nested JSON: metric name → run key (`"{run_name}_{run_id}"`) → list of the metric's values over the run. It needs the optional `extra` (`wandb`); a missing `wandb` surfaces the `pip install jade[extra]` hint at call time.

**Note — a small exception to "the cli adds no new behavior".** This command needs two helpers (read a run's full history, strip the `parameters/`+`gradients/` columns that `wandb.watch` logs) that do not exist elsewhere in `jade` yet. Rather than open a submodule under `cli/`, they were (per the author's decision) reconstructed as module-level helpers inside `wandb.py`. If more wandb functionality lands later, promote them to a dedicated `jade` module and reduce the cli file back to a thin wrapper. The run history is read with `run.scan_history()` (full, unsampled — every logged step), not the sampled `run.history()`. See [`wandb.md`](./wandb.md).

## Adding a new command (checklist)

1. Add the real capability to the proper module (never to the cli) if it does not exist yet.
2. Add a thematic file `cli/<theme>.py` with an implementation function `foo(args)` that forwards the parsed args.
3. Add an entry-point function `foo()` in `cli.py`: build the `argparse` parser, then call the implementation (same name).
4. Register the console script in `[project.scripts]` of `pyproject.toml` (`foo-command = "jade.cli.cli:foo"`, hyphenated name).
5. Update this wiki (this page, the index, a new per-file page, and the top-level quickstart/index).

## Dependencies

`jade.cli.cli` imports only `argparse` at module load; the implementation is imported lazily inside each entry-point function, so importing `cli.py` stays light. A command inherits whatever the wrapped `jade` code needs — e.g. `download-dataset-moabb` needs the optional `dataset` extra (`moabb`/`mne`) and `download-project-runs-metrics` needs the optional `extra` (`wandb`), each surfacing the usual `pip install jade[<extra>]` hint if missing.
