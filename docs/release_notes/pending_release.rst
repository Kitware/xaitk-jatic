Pending Release Notes
=====================

Updates / New Features
----------------------

CI/CD

* Added Dockerfile to the ``build`` job to build the docker image.

* Updated ``index.rst``, ``installation.rst``, and ``README.md``  based on ``devel-jatic``.

* Added automated scanning and testing of the ``aukus`` deployment container.

Documentation

* Added warning to use Poetry only in a virtual environment per Poetry documentation.

* Clarified that ``poetry<2.0`` is currently required.

Tests

* Added ``skipif`` marker for ``test_notebooks.py`` if ``tools`` extras is not installed.

Fixes
-----

* Fix docker-entrypoint to match a previous code changes

* Fix ``jatic-perturbations.ipynb`` error with Albumentation's gaussian blur.

* Removed ``Optional`` and ``Union`` type hints.

* Update pytest and ruff configurations

* Fix broken link in README
