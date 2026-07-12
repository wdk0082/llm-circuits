"""Sphinx configuration for the llm-circuits documentation.

Build with ``make docs`` from the repo root, or::

    uv run --group docs sphinx-build -b html docs docs/_build/html

Autodoc imports the package, so the docs must be built in an environment where
``llm-circuits`` is installed (``uv sync --group docs`` does this).
"""

from __future__ import annotations

import shutil
from importlib.metadata import version as _pkg_version
from pathlib import Path

# -- Paths -------------------------------------------------------------------
DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent

# The multilingual reproduction notebook is one of the two worked examples.
# Keep a single source of truth in ``notebooks/`` and copy it into the docs tree
# at build time so myst-nb can render it.  Execution stays off (see below): the
# docs show the stored outputs from the reviewed run.
_NB_SRC = REPO_ROOT / "notebooks" / "multilingual.ipynb"
_NB_DST = DOCS_DIR / "notebooks" / "multilingual.ipynb"
if _NB_SRC.exists():
    _NB_DST.parent.mkdir(exist_ok=True)
    shutil.copyfile(_NB_SRC, _NB_DST)

# -- Project information -----------------------------------------------------
project = "llm-circuits"
author = "Zixuan Wang"
copyright = "2026, Zixuan Wang"  # noqa: A001
release = _pkg_version("llm-circuits")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # a few docstrings use NumPy-style Parameters/Returns
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_nb",  # Markdown pages (.md) + rendered notebooks (.ipynb)
    "sphinx_copybutton",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST / notebooks --------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist", "attrs_inline"]
myst_heading_anchors = 3
nb_execution_mode = "off"  # render stored outputs; never execute during a build

# -- Autodoc / napoleon ------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}
autodoc_typehints = "signature"
# Docstrings are a mix of Google-style (Args:/Returns:/Yields:) and NumPy-style
# (Parameters/Returns underlines); napoleon parses both by default.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
# circuit-tracer (a git dependency) and torch_xla (TPU-only) are imported lazily
# inside functions; mock them so the API docs build without a GPU/TPU present or
# the circuit-tracer git install.
autodoc_mock_imports = ["circuit_tracer", "torch_xla"]

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
intersphinx_timeout = 10  # bound inventory fetches so a slow host can't hang the build

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = f"llm-circuits {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/wdk0082/llm-circuits/",
    "source_branch": "main",
    "source_directory": "docs/",
}
