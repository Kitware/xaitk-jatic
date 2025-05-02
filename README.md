<!-- :auto badges: -->
[![PyPI - Python Version](https://img.shields.io/pypi/v/xaitk-jatic)](https://pypi.org/project/xaitk-jatic/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xaitk-jatic)
[![Documentation Status](https://readthedocs.org/projects/xaitk-jatic/badge/?version=latest)](https://xaitk-jatic.readthedocs.io/en/latest/?badge=latest)
<!-- :auto badges: -->

# XAITK JATIC Integrations

"Bucket" to house various examples and resources related to
[`xaitk-saliency`](https://github.com/xaitk/xaitk-saliency)
integration and scaling for JATIC use as part of the JATIC program.

<!-- :auto installation: -->
## Installation
`xaitk-jatic` installation has been tested on Unix and Linux systems.

To install the current version via `pip`:
```bash
pip install xaitk-jatic[<extra1>,<extra2>,...]
```

To install the current version via `conda-forge`:
```bash
conda install -c conda-forge xaitk-jatic
```

Certain plugins may require additional runtime dependencies. Details on these requirements can be found
[here](https://xaitk-jatic.readthedocs.io/en/latest/implementations.html).

For more detailed installation instructions, visit the
[installation documentation](https://xaitk-jatic.readthedocs.io/en/latest/installation.html).
<!-- :auto installation: -->

<!-- :auto getting-started: -->
## Getting Started
Explore usage examples of the `xaitk-jatic` package in various contexts using the Jupyter notebooks provided in the
`./docs/examples/` directory.
<!-- :auto getting-started: -->

<!-- :auto documentation: -->
## Documentation
Documentation for both release snapshots and the latest main branch is available on
[ReadTheDocs](https://xaitk-jatic.readthedocs.io/en/latest/).

To build the Sphinx-based documentation locally for the latest reference:
```bash
# Install dependencies
poetry install --sync --with main,linting,tests,docs
# Navigate to the documentation root
cd docs
# Build the documentation
poetry run make html
# Open the generated documentation in your browser
firefox _build/html/index.html
```
<!-- :auto documentation: -->

<!-- :auto contributing: -->
## Contributing
Contributions are encouraged!

The following points help ensure contributions follow development practices.

- Follow the [JATIC Design Principles](https://cdao.pages.jatic.net/public/program/design-principles/).
- Adopt the Git Flow branching strategy.
- Detailed release information is available in [docs/release_process.rst](./docs/release_process.rst).
- Additional contribution guidelines and issue reporting steps can be found in [CONTRIBUTING.md](./CONTRIBUTING.md).
<!-- :auto contributing: -->

<!-- :auto developer-tools: -->
### Developer Tools

Ensure the source tree is acquired locally before proceeding.

#### Poetry Install

You can install using [Poetry](https://python-poetry.org/):

> [!WARNING]
> Users unfamiliar with Poetry should use caution. See
> [installation documentation](https://xaitk-jatic.readthedocs.io/en/latest/installation.html#from-source) for more
> information.

```bash
poetry install --with main,linting,tests,docs --extras "<extra1> <extra2> ..."
```

#### Pre-commit Hooks
Pre-commit hooks ensure that code complies with required linting and formatting guidelines. These hooks run
automatically before commits but can also be executed manually. To bypass checks during a commit, use the `--no-verify`
flag.

To install and use pre-commit hooks:
```bash
# Install required dependencies
poetry install --sync --with main,linting,tests,docs
# Initialize pre-commit hooks for the repository
poetry run pre-commit install
# Run pre-commit checks on all files
poetry run pre-commit run --all-files
```
<!-- :auto developer-tools: -->

<!-- :auto license: -->
## License
[Apache 2.0](./LICENSE)
<!-- :auto license: -->

<!-- :auto contacts: -->
## Contacts

**Principal Investigator / Product Owner**: Brian Hu (Kitware) @brian.hu

**Scrum Master / Maintainer**: Brandon RichardWebster (Kitware) @b.richardwebster

**Deputy Scrum Master / Maintainer**: Emily Veenhuis (Kitware) @emily.veenhuis

**Project Manager**: Keith Fieldhouse (Kitware) @keith.fieldhouse

**Program Representative**: Austin Whitesell (MITRE) @awhitesell
<!-- :auto contacts: -->

<!-- :auto acknowledgment: -->
Acknowledgment
--------------

This material is based upon work supported by the Chief Digital and Artificial Intelligence Office under Contract No.
519TC-23-9-2032. The views and conclusions contained herein are those of the author(s) and should not be interpreted as
necessarily representing the official policies or endorsements, either expressed or implied, of the U.S. Government.
<!-- :auto acknowledgment: -->
