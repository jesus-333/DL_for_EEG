# `custom_loss_function/__init__.py` — registry and factory

Source: `src/jade/training/custom_loss_function/__init__.py`

## `CUSTOM_LOSS_REGISTRY`

```python
CUSTOM_LOSS_REGISTRY = {
    "elbo" : elbo_loss,
}
```

The keys must match the ones of [`LOSS_CONFIG_REGISTRY`](../../config/loss_functions/__init__.md): a loss is selected by the `name` of its config, so the two registries are two halves of the same mapping (config → implementation), exactly as `MODEL_CONFIG_REGISTRY` and `MODEL_REGISTRY` are.

## `get_custom_loss_function(config) -> nn.Module`

Accepts a `loss_config` instance or a plain dict (promoted through `get_loss_config`), reads `config.name` case-insensitively, and calls the registered class **with the whole config** — the same one-argument contract as `get_model`.

Raises `ValueError` listing the known names for anything else.

```python
>>> from jade.config import elbo_config
>>> from jade.training.custom_loss_function import get_custom_loss_function
>>> loss_function = get_custom_loss_function(elbo_config(beta = 0.5))
```

## Public API

Re-exports `elbo_loss`, `elbo_terms`, the registry and the factory, plus the [`support_loss`](support_loss.md) module and its four public helpers (`RECONSTRUCTION_LOSS_REGISTRY`, `get_reconstruction_loss_function`, `kl_divergence_normal`, `compute_kl_weight`), so the stateless pieces can be used and tested without reaching into the submodule.
