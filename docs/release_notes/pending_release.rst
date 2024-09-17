Pending Release Notes
=====================

Updates / New Features
----------------------

CI/CD

* Swapped out pipeline to use a shared pipeline.

* Added a mirroring job to replace builtin gitlab mirroring due to LFS issue.

* Numerous changes to help automated the CI/CD process.

* `poetry.lock` file updated for the dev environment.

* Updates to dependencies to support the new CI/CD.
  
Documentation

* Added Read the Docs configuration files

* Added a Containers section to documentation

* Added ``AUKUS.rst`` to Containers documentations

* Added sphinx's `autosummary` template for recursively populating
  docstrings from the module level down to the class method level.

* Added support for `sphinx-click` to generate documentation for python
  `click` functions.

* Updated config for `black` to set max line length to 120

Fixes
-----
