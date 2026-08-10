# `jade.cli.default_config`

Source: `src/jade/cli/default_config.py`

Implementation of the `create-default-config` command (the argparse side lives in [`cli.py`](cli.md), same function name `default_config`). It writes a `default_config` folder of **commented TOML skeleton configs** at a chosen location, by copying files shipped as package data — so the TOML content lives in the config package, not hardcoded in Python.

The output is a *skeleton*: every possible parameter is present with a reasonable value and a comment, meant to be edited (not used as-is). This is the one cli command that copies data files rather than wrapping an existing `jade` function.

## Module constants

- `DEFAULT_CONFIG_FOLDER_NAME = "default_config"` — the folder that is both the package-data source and the created output.
- `TYPE_TO_SUBFOLDER = {"train": "train", "model": "model", "dataset": "dataset"}` — maps the `--type` choice to a subfolder (`"all"` is handled separately).

## `default_config(args) -> None`

| `args` field | Type | Meaning |
|---|---|---|
| `args.path` | str | Directory where the `default_config` folder is created (current directory if not given). |
| `args.type` | str | One of `train` / `model` / `dataset` / `all`. |

Behaviour:
1. Locate the packaged skeleton tree with `importlib.resources.files("jade.config").joinpath("default_config")` (works for an installed **and** an editable/source install).
2. Destination is `<args.path>/default_config`.
3. `--type all` → `shutil.copytree(source, destination, dirs_exist_ok=True)` copies the whole tree; otherwise only the matching subfolder is copied.
4. Print a short confirmation with the absolute output path.

## The shipped tree (`src/jade/config/default_config/`)

```
train/
  training_clf.toml             # classifier training config: flat fields + default [optimizer]/[loss]/[dataloader], with comments pointing to the option files
  optimizer/{adam,adamw,sgd,rmsprop,lbfgs}.toml
  lr_scheduler/{exponential_lr,step_lr,cosine_annealing_lr,cosine_annealing_warm_restarts,cyclic_lr}.toml
  loss/{cross_entropy,nll}.toml
model/
  EEGNet.toml                   # from EEGNet_config() defaults (use_softmax = false, matching the default cross_entropy loss)
  pooling/{avg,blur,sinc,dwt,max}.toml
dataset/
  dataset.toml                  # dataset_config incl. its (optional) [filter] section
```

Multi-choice sections (optimizer / lr_scheduler / loss) get one file per option; the main `training_clf.toml` (the classifier training config) carries a filled-in default section plus a comment pointing at the alternatives. Each file's values track the corresponding config's defaults, and every shipped TOML round-trips through the matching `from_toml`. The shipped `EEGNet.toml` (`use_softmax = false`, Identity head → raw logits) and `training_clf.toml` (default `cross_entropy` loss, which expects logits) are deliberately consistent, so the pair passes [`cross_check_config`](../training/support_training.md) out of the box. One deliberate exception to the "track defaults" rule: `dataset.toml` keeps a filled-in `[filter]` section (an 8–32 Hz band-pass) as a ready-to-edit reference even though `dataset_config.filter` now defaults to `None` (no filtering) — the section is explicitly marked **optional** in a comment, so removing/commenting it out keeps the raw data.

## Packaging note

The `.toml` files are guaranteed into the wheel by a `[tool.hatch.build.targets.wheel.force-include]` entry in `pyproject.toml` mapping `src/jade/config/default_config` → `jade/config/default_config` (additive to the default Python-file selection).
