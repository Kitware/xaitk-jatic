import pytest
from maite._internals.testing.pyright import list_error_messages, pyright_analyze

from xaitk_jatic.interop.object_detection.dataset import is_usable


@pytest.mark.skipif(not is_usable, reason="Extra 'xaitk-jatic[tools]' not installed.")
@pytest.mark.filterwarnings("ignore:Jupyter is migrating its paths")
@pytest.mark.parametrize(
    ("filepath", "expected_num_errors"),
    [
        ("docs/examples/armory/armory_smqtk_detect_objects.py", 0),
        ("docs/examples/armory/xaitk-armory.ipynb", 0),
        ("docs/examples/gradio/gr_component_state.py", 0),
        ("docs/examples/gradio/model-comparison.ipynb", 0),
        ("docs/examples/gradio/xaitk-gradio.ipynb", 0),
        ("docs/examples/huggingface/xaitk-accelerate.ipynb", 0),
        ("docs/examples/huggingface/xaitk-huggingface.ipynb", 0),
        ("docs/examples/jatic-object-detector-protocol.ipynb", 0),
        ("docs/examples/jatic-image-classifier-protocol.ipynb", 0),
        ("docs/examples/jatic-perturbations.ipynb", 0),
        ("docs/examples/lightning/xaitk-lightning.ipynb", 2),
        ("docs/examples/mlflow/MNIST_MLFlow_scikit_saliency.ipynb", 1),
        ("docs/examples/shared_interest/xaitk-shared_interest.ipynb", 0),
    ],
)
def test_pyright_nb(filepath: str, expected_num_errors: int) -> None:
    results = pyright_analyze(filepath)[0]
    assert results["summary"]["errorCount"] <= expected_num_errors, list_error_messages(results)
