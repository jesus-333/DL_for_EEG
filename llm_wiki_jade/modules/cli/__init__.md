# `jade.cli.__init__`

Source: `src/jade/cli/__init__.py`

Docstring-only package marker for the `cli` subpackage. It carries **no eager imports** (importing `jade.cli` is free), and states the module's design contract:

- `cli.py` holds one entry-point function per command; each only defines its `argparse` parser and delegates to a thematic implementation file.
- the thematic files (e.g. `download_dataset.py`) receive the parsed arguments and call already-existing `jade` code.
- **rule:** the cli never introduces new behavior — it only exposes / facilitates access to functionality that already lives elsewhere in the package.
- the commands are registered as console scripts in the `[project.scripts]` table of `pyproject.toml`.
