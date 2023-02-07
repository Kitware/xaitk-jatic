Pending Release Notes
=====================
This is the initial release of this repository which hosts integration
documentation, examples and code related to integrating xaitk-saliency
components with the CDAO needs and use-cases.

This first release defines an xaitk-cdao package that is currently a
placeholder for future functionality, as well as a few initial example jupyter
notebooks. Each notebook explains how to integrate a particular platform
(HuggingFace, MLFlow, or PyTorchLightning) with the xaitk-saliency API to
generate saliency maps while utlizing the tools and functionality which that
platform provides.

Updates / New Features
----------------------

Docs

* Added initial sphinx-based documentation infrastructure, including these
  release notes.

* Added the beginning of documents to detail reflections and lessons learned
  when working the use of xaitk-saliency with those platforms.

* Added PyTorch Lightning reflections, especially concerning scalability.

* Added MLFlow reflections.

Examples

* Added an example notebook for integrating the use of HuggingFace with
  xaitk-saliency.

* Added an example notebook for integrating the use of MLFlow with
  xaitk-saliency.

* Added an example notebook with strategies for integrating the use of
  PyTorch Lightning with xaitk-saliency. An additional notebook
  benchmarking these strategies was also added.

Scripts

* Reuse a public helper script previously developed to assist in pending
  release notes transition upon a release.

Fixes
-----
