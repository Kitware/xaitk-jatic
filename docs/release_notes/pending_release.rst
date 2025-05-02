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

Fixes
-----

* Fix docker-entrypoint to match a previous code changes

* Fix ``jatic-perturbations.ipynb`` error with Albumentation's gaussian blur.
