"""
Command-line interface (cli) subpackage.

It exposes a handful of ``jade`` functionalities as console scripts. The design is intentionally thin:

* :mod:`jade.cli.cli` holds one entry-point function per command; each only defines its ``argparse`` parser and then delegates to a thematic implementation file.
* the thematic files (e.g. :mod:`jade.cli.download_dataset`) receive the parsed arguments and call the already-existing ``jade`` code that does the actual work.

As a rule the cli never introduces new behavior: it only exposes / facilitates access to functionality that already lives elsewhere in the package.
The commands themselves are registered as console scripts in the ``[project.scripts]`` table of ``pyproject.toml``.
"""
