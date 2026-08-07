# `config/wandb_training.py`

Source: `src/jade/config/wandb_training.py`

Defines `wandb_training_config`, grouping every parameter specific to logging a training run on Weights & Biases (wandb). Kept separate from [`training_config`](training.md) (which references it as an **optional** nested field, `wandb=None` by default) so a run that does not use wandb carries no wandb-related noise. `wandb` itself is an optional dependency, needed only if this config is used.

Consumed by `jade.training.classifier.wandb_train` (sets up `wandb.init` and the model `wandb.Artifact`) and `jade.training.classifier.train` (logs per-epoch metrics).

## Class `wandb_training_config(generic_config)`

`@dataclass`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `project_name` | `str` | `""` | wandb project the run is logged to. **Required** (non-empty). |
| `model_artifact_name` | `str` | `""` | Base name of the model `wandb.Artifact`. **Required** (non-empty). The trained artifact is logged with the suffix `"_trained"` appended. |
| `entity` | `str \| None` | `None` | wandb entity (user/team) owning the project; `None` = default entity. |
| `name_training_run` | `str \| None` | `None` | Human-readable run name; `None` lets wandb assign a random one. |
| `notes` | `str \| None` | `None` | Free-text notes attached to the run. |
| `log_model_artifact` | `bool` | `True` | Whether to upload the model weights as a wandb artifact. |
| `log_freq` | `int` | `1` | How often (in epochs) `wandb.watch` logs gradients/parameters. Must be positive. |
| `debug` | `bool` | `False` | Not used by the training logic; only handy to filter debug runs in the wandb UI. |

## `__post_init__` — validation

Raises `ValueError` if:
- `project_name` is empty.
- `model_artifact_name` is empty.
- `log_freq` is not a positive `int` (`bool` rejected).

## Gotchas

- The `"_trained"` suffix convention on the artifact name is applied by the consumer (`classifier`), not here.
- Because this config is only nested optionally, its mandatory fields (`project_name`, `model_artifact_name`) are validated only when someone actually constructs it — `training_config` with `wandb=None` never triggers this.
