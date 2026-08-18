# `jade.cli` — Module Index

Table of contents for the `cli` module wiki. Start with the quickstart; open a per-file page for exact arguments and delegation details.

| Page | Source file | Description |
|------|-------------|-------------|
| [`0_module_quickstart.md`](./0_module_quickstart.md) | — | **Read first.** Purpose, the two-layer entry-point/implementation pattern, the available commands, and the checklist for adding a new one. |
| [`__init__.md`](./__init__.md) | `src/jade/cli/__init__.py` | Package surface: docstring only (describes the thin-cli design). No eager imports. |
| [`cli.md`](./cli.md) | `src/jade/cli/cli.py` | The entry points: one `argparse`-only function per command, each declared as a console script in `[project.scripts]` and delegating to a thematic implementation. |
| [`download_dataset.md`](./download_dataset.md) | `src/jade/cli/download_dataset.py` | Implementation of `download-dataset-moabb`: forwards the parsed args to `jade.dataset.download_dataset`. |
| [`default_config.md`](./default_config.md) | `src/jade/cli/default_config.py` | Implementation of `create-default-config`: copies the commented TOML skeleton tree (`config/default_config/`) to a chosen location. |
| [`wandb.md`](./wandb.md) | `src/jade/cli/wandb.py` | Implementation of `download-project-runs-metrics`: downloads the metrics of all runs in a wandb project (via `wandb.Api`) and saves them to a JSON file. Reconstructs the two run-history helpers it needs locally. |
