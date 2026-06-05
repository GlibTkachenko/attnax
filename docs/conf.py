# SPDX-License-Identifier: Apache-2.0

"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list, see:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------

from __future__ import annotations

import dataclasses
import enum
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import attnax  # noqa: E402

# -- Project information -----------------------------------------------------

project = "Attnax"
author = "The Attnax authors"
copyright = "2025, The Attnax Authors"
release = attnax.__version__
version = release

# -- General configuration ---------------------------------------------------

extensions = [
  "sphinx.ext.autodoc",
  "sphinx.ext.autosummary",
  "sphinx.ext.doctest",
  "sphinx.ext.napoleon",
  "sphinx.ext.intersphinx",
  "sphinx.ext.viewcode",
  "sphinx.ext.mathjax",
  "sphinx_design",
  "myst_nb",
]

# so we don't have to do the canonical imports on every doctest
doctest_global_setup = """
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import attnax
"""

master_doc = "index"
language = "en"
source_suffix = [".rst", ".md", ".ipynb"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for autodoc -----------------------------------------------------

autosummary_generate = False
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_class_signature = "mixed"
autodoc_default_options = {
  "members": True,
  "undoc-members": False,
  "show-inheritance": True,
  "member-order": "bysource",
  "exclude-members": (
    "__repr__, __str__, __weakref__, __init__, __new__, __post_init__"
  ),
}

# Google-style docstrings via Napoleon.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_rtype = False

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
  "python": ("https://docs.python.org/3", None),
  "jax": ("https://jax.readthedocs.io/en/latest/", None),
  "flax": ("https://flax.readthedocs.io/en/latest/", None),
  "optax": ("https://optax.readthedocs.io/en/latest/", None),
}

# -- Options for myst-nb -----------------------------------------------------

myst_enable_extensions = [
  "colon_fence",
  "deflist",
  "fieldlist",
  "linkify",
  "tasklist",
]
myst_heading_anchors = 3

# Notebooks are committed without outputs; never re-run them at build time.
nb_execution_mode = "off"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = f"Attnax {release}"
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"
html_theme_options = {
  "repository_url": "https://github.com/glibtkachenko/attnax",
  "use_repository_button": True,
  "use_issues_button": True,
  "use_edit_page_button": True,
  "repository_branch": "main",
  "path_to_docs": "docs",
  "home_page_in_toc": False,
  "launch_buttons": {
    "colab_url": "https://colab.research.google.com",
    "notebook_interface": "jupyterlab",
  },
  "logo": {
    "image_light": "_static/logo.png",
    "image_dark": "_static/logo.png",
    "alt_text": "Attnax",
  },
}
html_static_path = ["_static"]
html_css_files: list[str] = ["style.css"]
templates_path = ["_templates"]

# -- Suppress duplicate dataclass / enum member entries ----------------------
#
# Sphinx >= 8 auto-lists every dataclass field and enum member as a separate
# attribute. We already document them via Napoleon ``Args:`` / ``Attributes:``
# sections, so drop the duplicates.

_current_class: list[type | None] = [None]


def _track_class_signature(app, what, name, obj, options, signature, return_annotation):
  if what == "class" and isinstance(obj, type):
    _current_class[0] = obj


def _skip_struct_members(app, what, name, obj, skip, options):
  if skip:
    return skip
  cls = _current_class[0]
  if cls is None:
    return skip
  if dataclasses.is_dataclass(cls) and name in getattr(
    cls, "__dataclass_fields__", {}
  ):
    return True
  if issubclass(cls, enum.Enum) and name in getattr(cls, "__members__", {}):
    return True
  return skip


def setup(app):
  app.connect("autodoc-process-signature", _track_class_signature)
  app.connect("autodoc-skip-member", _skip_struct_members)
