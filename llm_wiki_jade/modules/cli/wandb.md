# `jade.cli.wandb`

Source: `src/jade/cli/wandb.py`

Implementation (thematic) file behind the `download-project-runs-metrics` command. It receives the already-parsed `argparse` arguments from [`jade.cli.cli.download_project_runs_metrics`](./cli.md) and talks to the wandb public API (`wandb.Api`).

**Small exception to the "cli adds no new behavior" rule.** The two helpers this command needs — reading a run history and stripping the model parameters/gradients — do not exist elsewhere in `jade` yet, so (as agreed with the author) they are reconstructed here as module-level helpers instead of adding a submodule under `cli/`. If more wandb functionality is added later they can be promoted to a dedicated `jade` module and this file reduced back to a thin wrapper.

## Imports
- `import json` — to serialize the collected metrics.
- `import pandas as pd` — the run history is materialized into a `DataFrame` (pandas is a core dependency).
- `import wandb` — the public API (`wandb.Api`) and its error type (`wandb.errors.CommError`). `wandb` is the optional `extra` of the package; since `cli.py` imports this module **lazily** (only when the command runs), a missing `wandb` surfaces the usual `pip install jade[extra]` hint only at call time. The file name is `wandb.py` but `import wandb` is an absolute import, so it resolves to the installed third-party package, not the module itself (no shadowing).

---

## `get_run_information(run) -> dict`

Collect from a single wandb run the information needed to export its metrics.

The history is downloaded with **`run.scan_history()`** — the full, unsampled history (every logged step), as opposed to the sampled `run.history()` (~500 rows). Because different metrics can be logged at different steps, the resulting `DataFrame` is sparse (gaps become `NaN`), which is handled downstream.

**Parameter:** `run` — a run object returned by `wandb.Api().runs(...)`.

**Returns** a dict with:

| Field | Type | Meaning |
|-------|------|---------|
| `name` | str | Human-readable run name. |
| `id` | str | Unique run id. |
| `history` | pandas.DataFrame | Full run history (one column per logged key, one row per logged step). |

---

## `remove_parameters_from_history(run_history) -> pandas.DataFrame`

Drop the columns logged by `wandb.watch` — the per-layer **model parameters and gradients**, which land under the `parameters/` and `gradients/` column prefixes (see [`training/classifier.py`](../training/classifier.md), where `wandb_train_model` calls `wandb.watch(model, log="all", ...)`). These are model internals, not training metrics.

**Parameter:** `run_history` — the `history` DataFrame from `get_run_information`.

**Returns** the same history without any `parameters/*` / `gradients/*` column (a no-op when the run was not tracked with `wandb.watch`).

---

## `download_project_runs_metrics(args) -> None`

Download the metrics of every run in a wandb project and save them to a JSON file.

**Parameter:** `args` — an `argparse.Namespace` (produced in `cli.py`) with the fields:

| Field | Type | Meaning |
|-------|------|---------|
| `args.entity` | str | wandb entity (user or team) that owns the project. |
| `args.project` | str | wandb project whose runs are exported. |
| `args.path_save` | str or None | output JSON path; `None` → `./wandb_metrics.json` in the current working directory. |

**Body (thematic blocks):**
1. *Argument checks.* `entity`/`project` are required (the argparse parser already enforces this; the `None` guards also protect direct programmatic calls). `path_save` falls back to `./wandb_metrics.json` (with a printed notice).
2. *Download.* Inside a `try` (guarding `wandb.errors.CommError` and any other `Exception`, both printed as a friendly message): `wandb.Api().runs(f"{entity}/{project}")`, then per run — `get_run_information` → `remove_parameters_from_history` → for every remaining column, skip the wandb internals (columns starting with `_`: `_step`, `_runtime`, `_timestamp`, …) and store `metrics_df[metric].dropna().tolist()` under `metrics_dict[metric]["{run_name}_{run_id}"]`. The `dropna()` removes the `NaN` left by the sparse full history, so each list is the clean curve of that metric.
3. *Save.* `json.dump(metrics_dict, ..., indent=4)`.

**Output shape** — metric name → run key (`"{run_name}_{run_id}"`) → list of values:
```json
{
    "train_loss": {
        "run_name_run_id": [1.0, 0.9, 0.8],
        "run_name_run_id": [1.1, 1.0, 0.9]
    },
    "accuracy": {
        "run_name_run_id": [0.1, 0.2, 0.3]
    }
}
```

**Returns** `None` (its effect is the JSON file written to `path_save`).
