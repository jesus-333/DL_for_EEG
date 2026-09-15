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
  wandb.toml                    # header-less [wandb] settings (load flat / nest under `wandb`); the [wandb] section of training_clf.toml is optional, shipped commented-out
model/
  EEGNet.toml                   # from EEGNet_config() defaults (use_softmax = false, matching the default cross_entropy loss)
  EEGConformer.toml             # from EEGConformer_config() defaults
  ShallowFBCSPNet.toml          # from ShallowFBCSPNet_config() defaults
  Deep4Net.toml                 # from Deep4Net_config() defaults (four [pooling] stages)
  TIDNet.toml                   # from TIDNet_config() defaults
  pooling/{avg,blur,sinc,dwt,max}.toml
dataset/
  dataset.toml                  # dataset_config; its [filter] section is optional, shipped commented-out
  filter/{bandpass,bandstop,lowpass,highpass,notch}.toml   # one header-less [filter] settings file per filter type (load flat / nest under `filter`)
```

Multi-choice sections come as **one file per option** under a subfolder, and the main file references them in a comment:

- **Required** picks ship a filled-in default in the main file plus a pointer to the alternatives: `[optimizer]` (Adam; options in `train/optimizer/`) and `[loss]` (cross-entropy; options in `train/loss/`).
- **Optional** picks ship **commented-out** in the main file with a pointer to their option file(s): `[lr_scheduler]` in `training_clf.toml` (options in `train/lr_scheduler/`); `[filter]` in `dataset.toml` (options in `dataset/filter/`, one file per filter type — bandpass / bandstop / lowpass / highpass / notch), since `dataset_config.filter` defaults to `None` (no filtering, raw data kept); and `[wandb]` in `training_clf.toml` (header-less settings in `train/wandb.toml`), since `training_config.wandb` defaults to `None` (no wandb logging).
- `[dataloader]` in `training_clf.toml` is the one optional section shipped **filled-in** (torch's DataLoader defaults) but explicitly marked optional in a comment — remove it to use those defaults, since `training_config.dataloader` defaults to `None`.

Each file's values track the corresponding config's defaults, and every shipped **main** TOML (`training_clf.toml`, `dataset.toml`, and the five model files `EEGNet.toml` / `EEGConformer.toml` / `ShallowFBCSPNet.toml` / `Deep4Net.toml` / `TIDNet.toml`) round-trips through the matching `from_toml` (the option files under the subfolders carry **no** `[section]` header — see [below](#option-files-are-header-less-no-section) — so they can be nested directly under the main config's key **or** pasted under the matching `[section]` in a main file). Finally, the shipped `EEGNet.toml` (`use_softmax = false`, Identity head → raw logits) and `training_clf.toml` (default `cross_entropy` loss, which expects logits) are deliberately consistent, so the pair passes [`cross_check_config`](../training/support_training.md) out of the box.

## Option files are header-less (no `[section]`)

The **option files** under the subfolders — `train/optimizer/*.toml`, `train/loss/*.toml`, `train/lr_scheduler/*.toml`, `train/wandb.toml` and `dataset/filter/*.toml` — ship **without** the `[optimizer]` / `[loss]` / `[lr_scheduler]` / `[wandb]` / `[filter]` header at the top of the file. Only their leading comment names the section they belong to. This is intentional: it lets each file be consumed in the natural ways **without producing a doubly-nested config**.

- **Nest the loaded dict directly.** `toml.load("nll.toml")` returns the settings flat (`{"name": "nll", ...}`), so a build script can drop it straight under the parent key: `training_config_dict["loss"] = toml.load("nll.toml")`. This is exactly what [`training/0_prepare_config.py`](../../../training/0_prepare_config.py) does for `loss` / `optimizer` / `lr_scheduler` / `wandb` (and `filter` under the dataset config). Had the file kept its `[loss]` header, `toml.load` would return `{"loss": {...}}`, the assignment would nest it a second time (`{"loss": {"loss": {...}}}`), and `training_config.from_dict` would then read an empty discriminator and raise `ValueError: unknown loss name ''` (the real fields sit one level too deep for the polymorphic dispatch).
- **Load it on its own.** `loss_config.from_toml("nll.toml")` (and the equivalents `optimizer_config.from_toml`, `lr_scheduler_config.from_toml`, `filter_config.from_toml`, `wandb_training_config.from_toml`) now returns the right config, because the flat keys are read directly and the polymorphic `from_dict` dispatches on `name`.
- **Paste into a main file.** To assemble everything in one TOML instead, copy the file's keys **under** the corresponding header in the main file (e.g. the filter settings go under `[filter]` in `dataset.toml`, an optimizer under `[optimizer]` in `training_clf.toml`). Each option file's leading comment states which header to use.

The **main files** (`training_clf.toml`, `dataset.toml`, the model files), by contrast, **do** carry their sections inline (`[optimizer]`, `[loss]`, `[dataloader]`, `[pooling]`, …), because they are loaded whole via `from_toml`, where those sections are the legitimate nested tables. The header-less `model/pooling/*.toml` files already followed this convention (they list `pooling_type = …` flat, meant to go under the `[pooling.params*]` sub-table of a model file — `[pooling.params_1]` / `[pooling.params_2]` for `EEGNet.toml`, `[pooling.params_1]` … `[pooling.params_4]` for `Deep4Net.toml`, the single `[pooling.params]` for the others).

## Packaging note

The `.toml` files are guaranteed into the wheel by a `[tool.hatch.build.targets.wheel.force-include]` entry in `pyproject.toml` mapping `src/jade/config/default_config` → `jade/config/default_config` (additive to the default Python-file selection).
