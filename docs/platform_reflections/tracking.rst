##################
Tracking Platforms
##################


MLFlow
======

**Introduction**:

We implement an end-to-end pipeline example using MLFlow's ``Experiment`` and
``Run`` capabilities. Each experiment represents each stage of the pipeline as
follows:

- <Algorithm name> - ``Train``
- <Algorithm name> - ``Evaluation``
- <Algorithm name> - ``Saliency Map Generation``

By separating and defining each ML pipeline stage as an MLFlow ``Experiment``,
it is possible to store and query per-stage information in a scalable manner.
Under each experiment, the information logging happens as part of MLFlow's
``Run-based`` workflow. Each MLFLow ``Run`` opens up a dashboard with 4 main
information subsections:

- ``Parameters`` - Specifically used in the ``Train`` stage to store model
  parameters.
- ``Metrics`` - Different metrics are stored across different stages and are
  used, most importantly, in querying specific runs based on a given threshold
  value.
- ``Tags`` - Key-value pair entries containing information that is useful for
  querying and also to link runs across different experiments.
- ``Artifacts`` - Used for storing images, numpy arrays, model metadata, etc.

**Train**:

Under the ``Train`` experiment, we create a single MLFlow run to log model
``parameters`` like learning rate, epochs, optimizer, hidden layer size, etc.
Additionally, we store some basic experiment info as ``tags``. Finally, in the
``artifacts`` section, we store the model.pkl and files containing info about
the python environment and dependencies.

**Evaluation**:

Under the ``Evaluation`` experiment, we create ``k`` MLFlow runs for ``k`` different
image samples from a given input dataset. In this experiment, for each run, we store
the ``Predicted_class_conf`` and class-wise confidence scores under ``metrics`` and
the ``Image_id``, ``GT_class`` and ``Predicted_class`` under ``tags``. By linking each
image sample to an MLFlow run, it is possible to make use of MLFlow's run filter and
query capabilities which helps to retrieve (both in the UI and at the backend using
``mlflow.search_runs()``) the image samples based on a given set of ``tags`` and ``metrics``.
Note: A call to ``mlflow.search_runs()`` with a specific filter string returns a pandas
DataFrame with the necessary query results.

**Saliency Map Generation**:

Under the ``Saliency Map Generation`` experiment, we create a single MLFlow run based on the
``Image_id`` and related information obtained through the query result from the ``Evaluation``
experiment. In this run, we store the saliency map numpy array and saliency map visualizations
(image) under ``artifacts`` for the queried ``Image_id``. In addition, images stored under
``artifacts`` can be previewed in the MLFlow UI. Furthermore, based on the design choice for
output generation, images and numpy arrays can be stored as separate files for each class or a
single file containing information from all classes. Most importantly, it is possible to edit
an existing run to store and preview saliency maps for a different ``Image_id`` based on an
updated backend query.

**MLFlow UI**:

Interacting with the MLFlow UI can be done side-by-side based on calling specific MLFlow functions
from Jupyter notebook. Creating each of the experiments mentioned above requires a call to
``mlflow.client.MlflowClient.create_experiment()``. Starting a run requires setting up a context
manager using ``mlflow.run()``. Logging all the necessary information happens within this context
manager codeblock. The MLFlow portions of the Jupyter notebook are setup in a way where each cell
corresponds to creating the experiment and run(s) corresponding to each stage discussed above.
Finally, in the last section of the Jupyter notebook, we discuss an example to edit/update an
existing MLFlow run based on modified query results.

**Summarizing based on specific attributes**:

- **Compatibility**:

  - No modifications to ``xaitk-saliency`` source code as part of the MLFlow integration.

- **Customizability**:

  - 3 different API levels to log information.

    - High level API uses the ``mlflow.sklearn.log_model`` function.
    - Mid level API uses the ``mlflow`` object to log information.
    - Low level API uses the MlflowClient object to log information. Quoting MLFlow
      documentation - “The ``mlflow.client`` module provides a Python CRUD interface to
      MLflow Experiments, Runs, Model Versions, and Registered Models. This is a lower level
      API that directly translates to MLflow REST API calls.”

- **Interactability**:

  - The main interactive nature of the MLFlow UI is in using the filter field to enter a query
    string based on a given set of metric thresholds and tag values to retrieve the corresponding
    MLFlow runs.
  - By clicking on each run, we can access the MLFlow dashboard containing the following sections:

    - Frontend unmodifiable sections - Parameters, Metrics and Artifacts
    - Frontend modifiable sections - Description and Tags
  - Updating the parameters, metrics and artifacts is strictly at the backend by querying using the
    ``mflow.search_runs()`` function and calling the ``mlflow.run()`` function on the same ``run_id``.

- **Scalability**:

  - Generating saliency maps for a given set of input samples is highly scalable. An existing MLFlow
    run can be updated to log the necessary saliency map images and numpy arrays of the updated input.
  - Generating metric values for a large dataset requires MLFlow runs of the order of ``O(n)`` where
    n is the number of image samples.
