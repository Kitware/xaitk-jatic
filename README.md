# XAITK JATIC Integrations

## Description
"Bucket" to house various examples and resources related to
[`xaitk-saliency`](https://github.com/xaitk/xaitk-saliency)
integration and scaling for JATIC use as part of the JATIC program.

## Installation
The following steps assumes the source tree has been acquired locally.

Install the current version via pip:
```bash
pip install .
```

Alternatively, [Poetry](https://python-poetry.org/) can also be used:
```bash
poetry install
```

## Usage
We provide a number of examples based on Jupyter notebooks in the
`./examples/` directory to show usage of the `xaitk-saliency`
package with other tools in the ML ecosystem. Similarly, demo
notebooks associated with each increment can be found in `./demos/`.

Reflections on integrating `xaitk-saliency` with these tools can be
found in `./docs/platform_reflections/`.

## Documentation
Documentation snapshots for releases as well as the latest master are hosted on
ReadTheDocs.

The sphinx-based documentation may also be built locally for the most
up-to-date reference:
```bash
# Install dependencies
poetry install
# Navigate to the documentation root.
cd docs
# Build the docs.
poetry run make html
# Open in your favorite browser!
firefox _build/html/index.html
```

## Contributing

- We follow the general guidelines outlined in the
[JATIC Software Development Plan](https://gitlab.jatic.net/jatic/docs/sdp/-/blob/main/Branch,%20Merge,%20Release%20Strategy.md).
- The Git Flow branching strategy is used.
- See `docs/releasing/release_process.rst` for detailed release information.
- See `CONTRIBUTING.md` for additional contributing information.

## License
Apache 2.0

**POC**: Brian Hu @brian.hu
**DPOC**: Paul Tunison @paul.tunison
