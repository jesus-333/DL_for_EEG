# `jade.training.custom_loss_function` — Module Index

Table of contents for the `custom_loss_function` subpackage wiki. Start with the [quickstart](0_module_quickstart.md).

| Page | Source | One-line description |
|---|---|---|
| [0_module_quickstart.md](0_module_quickstart.md) | — | Read-first overview of the subpackage. |
| [0_module_index.md](0_module_index.md) | — | This page. |
| [__init__.md](__init__.md) | `custom_loss_function/__init__.py` | `CUSTOM_LOSS_REGISTRY` / `get_custom_loss_function` factory and the public API. |
| [elbo.md](elbo.md) | `custom_loss_function/elbo.py` | `elbo_loss` (the VAE objective) and the `elbo_terms` named tuple. |
| [support_loss.md](support_loss.md) | `custom_loss_function/support_loss.py` | The reconstruction terms + registry, `kl_divergence_normal`, `compute_kl_weight`. |

> The matching config is [`elbo_config`](../../config/loss_functions/elbo.md), in `jade.config.loss_functions`.
