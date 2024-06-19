# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from pathlib import Path
import sys


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
import xaitk_jatic  # noqa: E402


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'xaitk-jatic'
copyright = '2022, Kitware, Inc.'
author = 'Kitware, Inc.'
release = xaitk_jatic.__version__

site_url = "https://jatic.pages.jatic.net/kitware/xaitk-jatic/"
repo_url = "https://gitlab.jatic.net/jatic/kitware/xaitk-jatic"
repo_name = "xaitk-jatic"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx-prompt",
]

suppress_warnings = [
    # Suppressing duplicate label warning from autosectionlabel extension.
    # This happens a lot across files that happen to talk about the same
    # topics.
    "autosectionlabel.*",
]

templates_path = []  # ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []  # ['_static']
