from unittest.mock import MagicMock
import numpy as np

from smqtk_core.configuration import configuration_test_helper

from xaitk_cdao.interop.jatic_toolbox.image_classification import JATICImageClassifier
from jatic_toolbox.protocols import (
    Classifier,
    HasLogits,
    HasProbs
)


class TestJATICImageClassifier:
    def test_configuration(self) -> None:
        """ Test configuration stability """
        mock_classifier = MagicMock(spec=Classifier)
        labels = ["A", "B", "C"]

        inst = JATICImageClassifier(mock_classifier, labels)
        for i in configuration_test_helper(inst):
            assert i._classifier == mock_classifier
            assert i._labels == labels

    def test_smoketest(self) -> None:
        """
        Run on a dummy image for basic sanity.
        No value assertions, this is for making sure that as-is functionality
        does not error for a mostly trivial case.
        """
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        labels = ["A", "B", "C"]

        mock_classifier_prob_output = MagicMock(
            spec=HasProbs,
            probs=np.asarray([[0.25, 0.15, 0.6]])
        )
        mock_classifier_logit_output = MagicMock(
            spec=HasLogits,
            logits=np.asarray([[1.3, 0.2, -0.4]])
        )
        mock_classifier = MagicMock(
            spec=Classifier,
        )
        mock_classifier.side_effect = [mock_classifier_prob_output, mock_classifier_logit_output]

        inst = JATICImageClassifier(mock_classifier, labels)

        # Probabilities
        res = list(inst.classify_images([dummy_image]))
        expected_res = [
            {"A": 0.25, "B": 0.15, "C": 0.6}
        ]
        assert res == expected_res

        # Logits
        res = [{key: round(d[key], 2) for key in d} for d in inst.classify_images([dummy_image])]
        expected_res = [
            {"A": 0.66, "B": 0.22, "C": 0.12}
        ]
        assert res == expected_res
