"""Sphinx configuration for BT Framework documentation."""

import sys
from pathlib import Path

# Add source directory to path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Project information
project = "BT Framework"
copyright = "2024, BT Framework Team"
author = "BT Framework Team"

# Version information
try:
    import bt

    version = bt.__version__
    release = bt.__version__
except ImportError:
    version = "1.0.0"
    release = "1.0.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",  # Google/NumPy style docstrings
    "myst_parser",  # Markdown support
    "sphinx_copybutton",  # Copy button for code blocks
    "sphinx_design",  # Modern UI components
]

# MyST Parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    "special-members": "__init__",
}

autodoc_typehints = "description"
autoclass_content = "both"

# Autosummary settings
autosummary_generate = True

# HTML output configuration
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]

# Theme options
html_theme_options = {
    "canonical_url": "",
    "analytics_id": "",
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#2980B9",
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Doctest settings
doctest_global_setup = """
import numpy as np
import pandas as pd
from bt import BacktestFramework
"""

# Coverage settings
coverage_modules = [
    "bt.core",
    "bt.strategies",
    "bt.reporting",
    "bt.monitoring",
]

# Napoleon settings (Google/NumPy style)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Copy button settings
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Todo settings
todo_include_todos = True

# Linkcheck settings (for CI)
linkcheck_ignore = [
    r"https://github\.com/.*#.*",  # GitHub anchors
    r"https://pypi\.org/.*",  # PyPI links
]

# Custom CSS
html_context = {
    "display_github": True,
    "github_user": "your-org",
    "github_repo": "bt-framework",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
