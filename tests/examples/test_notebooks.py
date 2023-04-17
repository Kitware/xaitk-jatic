import pytest

from jatic_toolbox.testing.pyright import list_error_messages, pyright_analyze


@pytest.mark.filterwarnings("ignore:Jupyter is migrating its paths")
@pytest.mark.parametrize("filepath, expected_num_errors", [
    ("examples/huggingface/xaitk-huggingface.ipynb", 0),
    ("examples/lightning/xaitk-lightning.ipynb", 1),
    ("examples/mlflow/MNIST_MLFlow_scikit_saliency.ipynb", 0),
    ("examples/jatic_toolbox/jatic-object-detector-protocol.ipynb", 0),
    ("examples/jatic_toolbox/jatic-image-classifier-protocol.ipynb", 0),
    ("demos/increment0-demo.ipynb", 0),
    ("demos/demo_utils.py", 0)
])
def test_pyright_nb(filepath: str, expected_num_errors: int) -> None:
    results = pyright_analyze(filepath)
    assert results["summary"]["errorCount"] <= expected_num_errors, list_error_messages(
        results
    )
