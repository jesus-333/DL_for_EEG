# `jade.cli.download_dataset`

Source: `src/jade/cli/download_dataset.py`

Implementation (thematic) file behind the `download-dataset-moabb` command. It receives the already-parsed `argparse` arguments from [`jade.cli.cli.download_dataset_moabb`](./cli.md) and forwards them to the existing dataset-module function — **no new behavior is added here**.

## Imports
- `from jade.dataset import moabb_dataset` — the moabb loader module; the command calls `moabb_dataset.download_dataset(...)`. The heavy `moabb`/`mne` imports only fire when it actually runs.

---

## `download_dataset_moabb(args) -> None`

Thin wrapper that pre-downloads a moabb dataset so the download does not block a later training run. Note that moabb downloads the **full raw recordings** (every session), so there is no train/test split at download time.

**Parameter:** `args` — an `argparse.Namespace` (produced in `cli.py`) with the fields:

| Field | Type | Meaning |
|-------|------|---------|
| `args.dataset_name` | str | moabb dataset name (case-insensitive). |
| `args.subject` | int or None | single subject; `None` → all subjects. |
| `args.dataset_folder` | str or None | where to store the data; `None` → moabb default `~/mne_data/`. |

**Body:** a single call —
```python
moabb_dataset.download_dataset(
    dataset_name   = args.dataset_name,
    subjects       = args.subject,
    dataset_folder = args.dataset_folder,
)
```
`download_dataset` accepts a single int (or `None`) for `subjects`, which matches the `--subject` argument directly.

**Returns** `None` (the return value of `download_dataset` — the moabb dataset object — is not needed by the cli).

See [`../dataset/moabb_dataset.md`](../dataset/moabb_dataset.md) for `download_dataset` itself, including the `dataset_folder` redirection and its single-global-folder gotcha.
