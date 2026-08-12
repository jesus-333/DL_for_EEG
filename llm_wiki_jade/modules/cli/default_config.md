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
  dataset.toml                  # dataset_config; its [filter] section is optional, shipped commented-out
  filter/{bandpass,bandstop,lowpass,highpass,notch}.toml   # one ready-to-copy [filter] table per filter type
```

Multi-choice sections come as **one file per option** under a subfolder, and the main file references them in a comment:

- **Required** picks ship a filled-in default in the main file plus a pointer to the alternatives: `[optimizer]` (Adam; options in `train/optimizer/`) and `[loss]` (cross-entropy; options in `train/loss/`).
- **Optional** picks ship **commented-out** in the main file with a pointer to their options: `[lr_scheduler]` in `training_clf.toml` (options in `train/lr_scheduler/`) and `[filter]` in `dataset.toml` (options in `dataset/filter/`, one file per filter type — bandpass / bandstop / lowpass / highpass / notch), since `dataset_config.filter` defaults to `None` (no filtering, raw data kept).
- `[dataloader]` in `training_clf.toml` is the one optional section shipped **filled-in** (torch's DataLoader defaults) but explicitly marked optional in a comment — remove it to use those defaults, since `training_config.dataloader` defaults to `None`.

Each file's values track the corresponding config's defaults, and every shipped **main** TOML (`training_clf.toml`, `dataset.toml`, `EEGNet.toml`) round-trips through the matching `from_toml` (the option-file snippets under the subfolders are meant to be copied into a main file, not loaded standalone). Finally, the shipped `EEGNet.toml` (`use_softmax = false`, Identity head → raw logits) and `training_clf.toml` (default `cross_entropy` loss, which expects logits) are deliberately consistent, so the pair passes [`cross_check_config`](../training/support_training.md) out of the box.

## Packaging note

The `.toml` files are guaranteed into the wheel by a `[tool.hatch.build.targets.wheel.force-include]` entry in `pyproject.toml` mapping `src/jade/config/default_config` → `jade/config/default_config` (additive to the default Python-file selection).
