# `config/generic.py`

Source: `src/jade/config/generic.py`

Defines `generic_config`, the fieldless `@dataclass` base class that **every** config in `jade.config` inherits from (directly, or via the family bases `optimizer_config` / `lr_scheduler_config` / `loss_config`). It supplies the shared (de)serialization helpers plus a human-readable multi-line `__str__`; field definitions and validation are left to subclasses.

## Module-level

- Imports the optional `toml` package inside a `try/except ImportError`. On failure `toml` is set to `None` and the exception is stashed in `_TOML_IMPORT_ERROR`, so TOML methods can raise a clear, chained error only when actually used.
- `from __future__ import annotations` is active, so **all annotations are strings at runtime**; this is why `from_dict` must call `typing.get_type_hints` to resolve them (see below).

## Private helpers

### `_require_toml()`
Raises a descriptive `ImportError` (chained from `_TOML_IMPORT_ERROR`) if the optional `toml` package is unavailable. Called at the top of `to_toml`/`from_toml`. No-op when `toml` is importable.

### `_config_type_of(annotation) -> type | None`
Given a **resolved** field annotation, returns the `generic_config` subclass it references, or `None`.

- Direct annotation that *is* a `generic_config` subclass → returns it.
- `Optional[X]` / `Union[X, None]` → scans `typing.get_args` and returns the first member that is a `generic_config` subclass.
- Anything else → `None`.

This is how `from_dict` knows which fields are nested configs (so it can recurse) versus plain values.

### `_strip_none(value) -> value`
Recursively removes `None` entries from dicts/lists. Used only by `to_toml`, because **TOML has no representation for null**. Dropped fields simply fall back to their dataclass defaults on the next `from_toml`. Dicts drop `None` values; lists/tuples are cleaned element-wise (and returned as lists); scalars pass through unchanged.

## Class `generic_config`

`@dataclass` with **no fields**. Provides three symmetric (de)serialization pairs plus a pretty multi-line `__str__` (for human-readable printing).

### `to_dict() -> dict`
Thin wrapper over `dataclasses.asdict(self)`. Recurses into nested config fields (they become nested dicts), mirroring the section structure used by the TOML/JSON writers.

### `from_dict(data: dict) -> generic_config` (classmethod)
Rebuilds a config instance from a (possibly nested) dict.

- Raises `TypeError` if `data` is not a dict.
- Resolves annotations via `typing.get_type_hints(cls)` (needed because of `from __future__ import annotations`).
- Iterates `data.items()`; **unknown keys are silently ignored** (so forward/backward-compatible files still load).
- For each known field: if `_config_type_of(hint)` returns a nested config class **and** the value is a dict, it recurses (`nested_cls.from_dict(value)`); otherwise the raw value is used as-is.
- Returns `cls(**kwargs)` — so subclass `__post_init__` validation runs on the rebuilt instance.

Gotcha: the recursion dispatches on `nested_cls.from_dict`, so the family base classes' **overridden** polymorphic `from_dict` (keyed on `name`) participates automatically when a nested field is an `optimizer_config`/`lr_scheduler_config`/`loss_config`.

### `to_toml(path)` / `from_toml(path)` (the latter is a classmethod)
- `to_toml`: `_require_toml()`, then `_strip_none(self.to_dict())`, then `toml.dump` to `path` (UTF-8). `None` fields are omitted. Returns `None`.
- `from_toml`: `_require_toml()`, `toml.load`, then `cls.from_dict(data)`. Nested configs become `[section]` tables in the file.
- `path` accepts `str` or `os.PathLike`.

### `to_json(path, indent=4)` / `from_json(path)` (the latter is a classmethod)
- `to_json`: `json.dump(self.to_dict(), file, indent=indent)`. **No `_strip_none`** — JSON keeps `null`, so `None` fields survive the round-trip (unlike TOML).
- `from_json`: `json.load`, then `cls.from_dict(data)`.

### `__str__() -> str` (pretty multi-line printing)
Human-readable representation used by `print(config)` / `str(config)`: **one parameter per line**, aligned to the column right after the class opening parenthesis, with **nested configs indented one level further** (aligned under their own name). Delegates to the private recursive helper `_pretty_str(indent=0)`, which walks `dataclasses.fields(self)`, recurses (`value._pretty_str(...)`) when a field value `isinstance(..., generic_config)` and uses `repr(value)` otherwise.

- Only `__str__` is overridden; **`repr(config)` is left as the compact single-line dataclass default** (so REPL/Jupyter bare display and logging stay one line). `@dataclass` never generates `__str__`, so defining it once on the base is inherited by every config without being shadowed.
- Values are shown with their plain `repr` (no spaces around `=`), e.g. `name='eegnet_classifier'`, `c_kernel_1=(1, 64)`, `params_1={}`, `filter=None`.

Example (`print(EEGNet_config())`):
```
EEGNet_config(name='eegnet_classifier',
              use_bias=False,
              ...
              pooling=eegnet_pooling_config(pooling_type_1='default',
                                            pooling_type_2='default',
                                            ...
                                            params_2={}))
```

## Design notes / gotchas

- Validation contract: subclasses implement `__post_init__` and raise `ValueError` on bad values. `generic_config` itself validates nothing.
- `from_dict` tolerating unknown keys + `to_toml` dropping `None` means the on-disk representation is intentionally lossy in a forward-compatible way; do not rely on TOML files preserving explicit `None` fields.
- `str(config)` (pretty, multi-line) and `repr(config)` (compact, single-line) differ on purpose: `print(config)` is readable, while `repr` stays terse for logs and REPL/Jupyter display. Both remain valid, reconstructable Python.
