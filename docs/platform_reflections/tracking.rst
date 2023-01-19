##################
Tracking Platforms
##################


MLFlow
======

Compatibility:

- No modifications to ``xaitk-saliency`` source code as part of the MLFlow integration.

Customizability:

- 3 different API levels to log information.

  - High level API uses the ``mlflow.sklearn.log_model`` function.
  - Mid level API uses the ``mlflow`` object to log information.
  - Low level API uses the MlflowClient object to log information. Quoting MLFlow documentation - “The ``mlflow.client`` module provides a Python CRUD interface to MLflow Experiments, Runs, Model Versions, and Registered Models. This is a lower level API that directly translates to MLflow REST API calls.”

Interactability:

- The main interactive nature of the MLFlow UI is in using the filter field to enter a query string based on a given set of metric thresholds and tag values to retrieve the corresponding MLFlow runs.
- By clicking on each run, we can access the MLFlow dashboard containing the following sections:

  - Frontend unmodifiable sections - Parameters, Metrics and Artifacts
  - Frontend modifiable sections - Description and Tags
- Updating the parameters, metrics and artifacts is strictly at the backend by querying using the ``mflow.search_runs()`` function and calling the ``mlflow.run()`` function on the same ``run_id``.

Scalability:

- Generating saliency maps for a given set of input samples is highly scalable. An existing MLFlow run can be updated to log the necessary saliency map images and numpy arrays of the updated input.
- Generating metric values for a large dataset requires MLFlow runs of the order of ``O(n)`` where n is the number of image samples.
