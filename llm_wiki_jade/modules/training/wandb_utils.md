# `training/wandb_utils.py` — everything about a wandb run

Source: `src/jade/training/wandb_utils.py`

All the [Weights & Biases](https://wandb.ai) plumbing of a training run, extracted from `classifier.py` so the two training cycles ([`classifier`](classifier.md) and [`vae`](vae.md)) share it instead of duplicating it.

**The payoff of the extraction:** neither `classifier.py` nor `vae.py` imports `wandb` at all any more. This module owns the optional import.

Imports `logging`, `contextlib.contextmanager` and — defensively — `wandb`.

---

## Why `wandb_utils` and not `wandb`

`jade/training/wandb.py` doing `import wandb` would actually work: python 3 always uses absolute imports (PEP 328), and `jade/cli/wandb.py` already proves it. The problem is not breakage but **readability**: `from jade.training import wandb` would be genuinely ambiguous to a reader, and any `from . import wandb` would shadow the real package. The chosen name removes the ambiguity and matches the existing `logging_utils.py` precedent in the same package.

---

## Availability

| Name | Purpose |
|---|---|
| `WANDB_IS_AVAILABLE` | `True` when wandb is installed. |
| `require_wandb()` | Raises the `ImportError` naming the `extra` optional dependency. |

Importing this module **never** fails; the error is raised only when a tracked run is actually requested. `wandb` is the optional `extra` group of the package.

---

## Run setup

| Function | Purpose |
|---|---|
| `build_config_to_log(training_config, model_config=None, dataset_config=None)` | The dict of configs attached to the run. Only the training config is mandatory; the other two are logged **for reproducibility only** and do not affect training. |
| `build_run_id(training_config)` | `f"{seed}_{name_training_run}"`, or `None` when either is missing (wandb then generates its own). A stable id is what lets a run be resumed. |
| `create_model_artifact(wandb_config, config_to_log)` | The `wandb.Artifact` holding the weights, or `None` when `log_model_artifact` is False. |
| `wandb_run(training_config, model_config=None, dataset_config=None)` | A `@contextmanager` wrapping `wandb.init(...)` (job type `"train"`, `resume="allow"`), yielding `(run, wandb_model_artifact)` and calling `run.log_artifact(...)` on a clean exit. |

---

## Logging during the training

| Function | Purpose |
|---|---|
| `watch_model(model, loss_function, wandb_config)` | `wandb.watch(..., log="all", log_graph=True)` — tracks the per-layer parameters and gradients. |
| `add_file_to_artifact(artifact, file_path)` | `add_file` + `wandb.save`. **A no-op on a `None` artifact**, so the callers need no guard. |
| `log_epoch(log_dict)` | A thin `wandb.log` wrapper; it exists so a training cycle never imports wandb. |

---

## Key fact — `run_training_with_wandb`

```python
run_training_with_wandb(train_model_function, training_config, model, train_dataset,
                        validation_dataset=None, model_config=None, dataset_config=None)
```

This is the **generic body of every tracked run**. The only thing that differs between a classifier and a generative model is *which* training function runs inside the run, so it is received as the first argument.

The two public `wandb_train_model` functions stay concrete and thin:

```python
# in classifier.py (and, with its own train_model, in vae.py)
return wandb_utils.run_training_with_wandb(
    train_model, training_config, model, train_dataset, validation_dataset,
    model_config = model_config, dataset_config = dataset_config,
)
```

so **no `train_model_function=` keyword is ever exposed on a public entry point**, and `classifier.wandb_train_model`'s six-positional-argument signature is untouched — which matters, because `training/EEGNet/train_eegnet.py` and `train_pooling.py` call it positionally.

---

## Not to be confused with `jade/cli/wandb.py`

[`cli/wandb.py`](../cli/wandb.md) also touches wandb, but from the other end: it reads a run **back** after it is over (`wandb.Api`, run histories, the `download-project-runs-metrics` command), while this module only **writes** during a run. They stay separate for that reason; only the cross-reference in the cli docstring was updated.
