# `jade.dataset.__init__`

Source: `src/jade/dataset/__init__.py`

Package surface for the `dataset` subpackage. It makes the two PyTorch dataset classes available immediately, while the heavier moabb loader is exposed **lazily** so that importing the package never triggers the optional `moabb`/`mne` dependencies.

## What is imported eagerly

```python
from jade.dataset.dataset_time import EEG_Dataset, EEG_Dataset_ChWi
```

`dataset_time` only needs `numpy`, `torch`, and `matplotlib` (always installed), so these two classes are safe to import at package-load time. See [`dataset_time.md`](./dataset_time.md).

## `__all__`

```python
__all__ = ["EEG_Dataset", "EEG_Dataset_ChWi", "download_dataset", "get_dataset"]
```

Note that `download_dataset` and `get_dataset` appear in `__all__` even though they are **not** imported at the top of the module. They are resolved on demand by `__getattr__` (below). This is deliberate: `from jade.dataset import *` and attribute access both work, but the moabb import is deferred until one of them is actually touched.

## `__getattr__(name)` — lazy re-export (PEP 562)

| Aspect | Detail |
|--------|--------|
| **Mechanism** | [PEP 562](https://peps.python.org/pep-0562/) module-level `__getattr__`. Python calls it only when a normal attribute lookup on the module fails, i.e. only for names not already bound in the module namespace. |
| **Parameter** | `name : str` — the attribute being accessed. |
| **Returns** | For `name in ("download_dataset", "get_dataset")`: imports `jade.dataset.moabb_dataset` **inside the function** and returns `getattr(moabb_dataset, name)`. |
| **Raises** | `AttributeError` (`module '<name>' has no attribute '<name>'`) for any other name — matching normal module behavior. |
| **Side effect** | First access to either loader name triggers `import jade.dataset.moabb_dataset`, which in turn (when its functions run) imports `moabb`/`mne`. If the `dataset` extra is not installed, the error surfaces later — inside the loader functions via `_import_moabb_mne()` — not at attribute-access time (importing the `moabb_dataset` module itself does not import moabb/mne; those imports are also deferred). |

### Why this matters (design decision)

`moabb` and `mne` are heavy, slow-to-import, optional dependencies (the `dataset` extra in `pyproject.toml`: `pip install jade[dataset]`). Anyone who only wants the `torch` datasets — e.g. to wrap their own in-memory EEG array in `EEG_Dataset` — should not pay that import cost, nor be forced to install those packages. The lazy `__getattr__` gives a clean public API (`jade.dataset.get_dataset`) while confining the heavy imports to the code path that genuinely needs them.

There are effectively **two layers of laziness**:

1. This module-level `__getattr__` defers importing `moabb_dataset` until `download_dataset`/`get_dataset` is accessed.
2. Inside `moabb_dataset`, `_import_moabb_mne()` defers importing `moabb`/`mne` until a function that needs them actually runs (see [`moabb_dataset.md`](./moabb_dataset.md)).

### Gotcha

Because `download_dataset`/`get_dataset` are surfaced via `__getattr__` rather than a real binding, IDEs and static analyzers may not "see" them, and `hasattr(jade.dataset, "get_dataset")` will trigger the import attempt. Functionally, though, `jade.dataset.get_dataset` and `from jade.dataset import get_dataset` both work as expected.
