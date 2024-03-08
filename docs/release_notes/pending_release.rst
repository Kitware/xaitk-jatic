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

* Updated the README with more detailed description and usage of the package
  including branch/merge/release strategy as well as listing points of contact
  (POCs).

* Added the beginning of documents to detail reflections and lessons learned
  when working the use of xaitk-saliency with those platforms.

* Added PyTorch Lightning reflections, especially concerning scalability.

* Added MLFlow reflections.

* Added reflections on applying the JATIC object detection protocol to
  xaitk-saliency.

* Added reference to original XAITK documentation.

* Added ability to render documentation on Gitlab Pages.

* Added sphinx auto documentation for JATIC object detection and image
  classification protocol adapters.

Examples

* Added an example notebook for integrating the use of HuggingFace with
  xaitk-saliency.

* Added an example notebook for integrating the use of MLFlow with
  xaitk-saliency.

* Added an example notebook with strategies for integrating the use of
  PyTorch Lightning with xaitk-saliency. An additional notebook
  benchmarking these strategies was also added.

* Added an example notebook demonstrating using the JATIC protocol-based
  detector with xaitk-saliency.

* Added an example notebook demonstrating using the JATIC protocol-based
  image classifier with xaitk-saliency.

* Added an example notebook for integrating the use of Gradio with
  xaitk-saliency.

* Added a demo notebook for increment 0 work exploring the JATIC object
  detector protocol, trame GUI, and other increment 0 progress.

* Added an example notebook demonstrating model comparison with Gradio and
  xaitk-saliency.

* Added an example notebook for integrating the use of HuggingFace Accelerate
  with xaitk-saliency. An additional notebook benchmarking this strategy was
  also added.

* Added a demo notebook for increment 1 work exploring the tracking of model,
  dataset and saliency map parameters using MLFlow.

* Added an example notebook exploring image perturbations and the effect on
  saliency maps using JATIC's augmentation protocol.

* Added an example notebook for testing the interoperability of xaitk-saliency
  with TwoSix's armory tool.

* Added an example notebook for integrating the use of Shared Interest with
  xaitk-saliency.

* Update README.md

Interoperability

* Added ``DetectImageObjects`` implementation, ``JATICDetector``, to allow
  for interoperability with object detectors following the ``ObjectDetector``
  protocol from JATIC.

* Added ``ClassifyImage`` implementation, ``JATICImageClassifier``, to allow
  for interoperability with image classifiers following the ``Classifier``
  protocol from JATIC.

License

* Add Apache 2.0 license

Scripts

* Reuse a public helper script previously developed to assist in pending
  release notes transition upon a release.

* Added a script that performs a CI check for changes to the release notes
  folder in a Merge Request.

Utils

* Added a CLI script as to generate saliency maps on COCO detections. This
  includes the functionality for operating on MAITE object detectors.

Fixes
-----

* `torch` is now an optional requirement.

* Fix typing errors for newer `pyright` versions.
