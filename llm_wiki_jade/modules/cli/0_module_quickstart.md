# `jade.cli` — Module Quickstart (READ FIRST)

> Read [`../../0_wiki_quickstart.md`](../../0_wiki_quickstart.md) first. This page is the read-first overview of the `cli` module; open a per-file page only for field-level detail.

## Purpose

`jade.cli` exposes a handful of `jade` functionalities as **console scripts** (terminal commands). It is deliberately thin: the cli **never introduces new behavior**, it only exposes / facilitates access to code that already lives elsewhere in the package. When a command needs a capability that does not exist yet, that capability is added to the appropriate module (e.g. `dataset`), *not* to the cli.

## Structure (the two-layer pattern)

```
cli/
  cli.py               <- entry points: one function per command, argparse only, then delegate
  download_dataset.py  <- thematic implementation file (receives parsed args, calls jade code)
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

## Adding a new command (checklist)

1. Add the real capability to the proper module (never to the cli) if it does not exist yet.
2. Add a thematic file `cli/<theme>.py` with an implementation function `foo(args)` that forwards the parsed args.
3. Add an entry-point function `foo()` in `cli.py`: build the `argparse` parser, then call the implementation (same name).
4. Register the console script in `[project.scripts]` of `pyproject.toml` (`foo-command = "jade.cli.cli:foo"`, hyphenated name).
5. Update this wiki (this page, the index, a new per-file page, and the top-level quickstart/index).

## Dependencies

`jade.cli.cli` imports only `argparse` at module load; the implementation is imported lazily inside each entry-point function, so importing `cli.py` stays light. A command inherits whatever the wrapped `jade` code needs — e.g. `download-dataset-moabb` needs the optional `dataset` extra (`moabb`/`mne`), which surfaces the usual `pip install jade[dataset]` hint if missing.
