# `jade.config.loss_functions` — Subpackage Index

Table of contents. Start with the [subpackage quickstart](0_module_quickstart.md). Parent: [config index](../0_module_index.md).

| File | Page | Description |
|---|---|---|
| `__init__.py` | [__init__.md](__init__.md) | `LOSS_CONFIG_REGISTRY` + `get_loss_config` factory + re-exports. |
| `generic.py` | [generic.md](generic.md) | `loss_config` base class + polymorphic `from_dict`. |
| `cross_entropy.py` | [cross_entropy.md](cross_entropy.md) | `cross_entropy_config` → `torch.nn.CrossEntropyLoss` (expects logits). |
| `nll.py` | [nll.md](nll.md) | `nll_config` → `torch.nn.NLLLoss` (expects log-probabilities). |
| `elbo.py` | [elbo.md](elbo.md) | `elbo_config` → `jade.training.custom_loss_function.elbo.elbo_loss` (a **custom** loss, with no `torch.nn` counterpart; expects a generative model). |
