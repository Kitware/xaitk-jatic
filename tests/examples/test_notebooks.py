import pytest

from maite.testing.pyright import list_error_messages, pyright_analyze


@pytest.mark.filterwarnings("ignore:Jupyter is migrating its paths")
@pytest.mark.parametrize("filepath, expected_num_errors", [
    ("examples/armory/armory_smqtk_detect_objects.py", 0),
    ("examples/armory/xaitk-armory.ipynb", 0),
    ("examples/gradio/gr_component_state.py", 0),
    ("examples/gradio/model-comparison.ipynb", 0),
    ("examples/gradio/xaitk-gradio.ipynb", 0),
    ("examples/huggingface/xaitk-accelerate.ipynb", 0),
    ("examples/huggingface/xaitk-huggingface.ipynb", 0),
    ("examples/jatic-object-detector-protocol.ipynb", 0),
    ("examples/jatic-image-classifier-protocol.ipynb", 0),
    ("examples/jatic-perturbations.ipynb", 0),
    ("examples/lightning/xaitk-lightning.ipynb", 1),
    ("examples/mlflow/MNIST_MLFlow_scikit_saliency.ipynb", 0),
    ("examples/shared_interest/xaitk-shared_interest.ipynb", 0)
])
def test_pyright_nb(filepath: str, expected_num_errors: int) -> None:
    results = pyright_analyze(filepath)[0]
    assert results["summary"]["errorCount"] <= expected_num_errors, list_error_messages(
        results
    )
