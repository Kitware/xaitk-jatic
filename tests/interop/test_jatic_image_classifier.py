import numpy as np
import pytest
from contextlib import nullcontext as does_not_raise
from typing import (
    Any, Callable, ContextManager, Iterator, Optional, Protocol, Sequence
)
from unittest.mock import MagicMock
from scipy.special import softmax  # type:ignore

from smqtk_core.configuration import configuration_test_helper
from smqtk_classifier.interfaces.classification_element import CLASSIFICATION_DICT_T

from xaitk_cdao.interop.image_classification import JATICImageClassifier
from maite.protocols import (
    ImageClassifier, HasLogits, HasProbs, HasScores, SupportsArray
)


class TestJATICImageClassifier:
    # HasProbs
    dummy_probs = np.asarray([[0.25, 0.15, 0.6]])
    dummy_prob_out = MagicMock(
        spec=HasProbs,
        probs=dummy_probs
    )

    # HasLogits
    dummy_logits = np.asarray([[1.3, 0.2, -0.4]])
    dummy_logit_probs = softmax(dummy_logits[0])
    dummy_logits_out = MagicMock(
        spec=HasLogits,
        logits=dummy_logits
    )

    # HasScores
    dummy_scores = np.asarray([[0.8, 0.15]])
    dummy_score_labels = np.asarray([[2, 1]])
    dummy_score_probs = [0, 0.15, 0.8]
    dummy_scores_out = MagicMock(
        spec=HasScores,
        scores=dummy_scores,
        labels=dummy_score_labels
    )

    # Unknown output
    class _FakeOutput(Protocol):
        fake_output: SupportsArray
    dummy_fake_output = np.asarray([[0.3, 0.2]])
    dummy_unknown_out = MagicMock(
        spec=_FakeOutput,
        fake_output=dummy_fake_output
    )

    dummy_labels1 = ["A", "B", "C"]
    dummy_labels2 = ["D", "E", "F"]

    @pytest.mark.parametrize("classifier, preprocessor, img_batch_size, expectation", [
        (MagicMock(spec=ImageClassifier), lambda x: x, 3,
            pytest.raises(NotImplementedError, match=r"Constructor arg")),
        (MagicMock(spec=ImageClassifier), None, 1,
            pytest.raises(NotImplementedError, match=r"Constructor arg")),
    ])
    def test_configuration(
        self,
        classifier: ImageClassifier,
        preprocessor: Optional[Callable[[SupportsArray], SupportsArray]],
        img_batch_size: int,
        expectation: ContextManager
    ) -> None:
        """ Test configuration stability """

        inst = JATICImageClassifier(
            classifier=classifier,
            preprocessor=preprocessor,
            img_batch_size=img_batch_size
        )
        with expectation:
            for i in configuration_test_helper(inst):
                # TODO: Update assertions appropriately once get_config/from_config are implemented
                """
                assert i._classifier == classifier
                assert i._preprocessor == preprocessor
                assert i._img_batch_size == img_batch_size
                """

    @staticmethod
    def _generate_mock_classifier(side_effect: Any, labels: Sequence[str]) -> ImageClassifier:
        """
        Generate a mock classifier with the given side effect(s) and get_labels return.
        """
        mock_classifier = MagicMock(
            spec=ImageClassifier,
        )
        mock_classifier.side_effect = side_effect
        mock_classifier.get_labels = MagicMock(return_value=labels)
        return mock_classifier

    @pytest.mark.parametrize(
        "classifier_side_effect, labels, preprocessor, img_batch_size, expected_return, expectation",
        [
            ([dummy_prob_out], dummy_labels1, lambda x: x, 1,
                [{la: prob for la, prob in zip(dummy_labels1, dummy_probs[0])}], does_not_raise()),
            ([dummy_logits_out], dummy_labels2, lambda x: x, 3,
                [{la: prob for la, prob in zip(dummy_labels2, dummy_logit_probs)}], does_not_raise()),
            ([dummy_scores_out], dummy_labels1, None, 2,
                [{la: prob for la, prob in zip(dummy_labels1, dummy_score_probs)}], does_not_raise()),
            ([dummy_unknown_out], dummy_labels2, None, 1,
                [{la: prob for la, prob in zip(dummy_labels2, dummy_fake_output)}],
                pytest.raises(ValueError, match=r"Unknown classifier output type")),
        ],
        ids=["HasProbs", "HasLogits", "HasScores", "UnknownOutput"]
    )
    def test_smoketest(
        self,
        classifier_side_effect: Any,
        labels: Sequence[str],
        preprocessor: Optional[Callable[[SupportsArray], SupportsArray]],
        img_batch_size: int,
        expected_return: Iterator[CLASSIFICATION_DICT_T],
        expectation: ContextManager
    ) -> None:
        """
        Ensure we can handle the various expected output types for a JATIC protocol-based classifier.
        """
        mock_classifier = TestJATICImageClassifier._generate_mock_classifier(
            side_effect=classifier_side_effect,
            labels=labels
        )

        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        with expectation:
            inst = JATICImageClassifier(
                classifier=mock_classifier,
                preprocessor=preprocessor,
                img_batch_size=img_batch_size
            )

            res = list(inst.classify_images([dummy_image]))
            assert res == expected_return

    @pytest.mark.parametrize("labels", [
        (dummy_labels1, ),
        (dummy_labels2, ),
    ])
    def test_labels(
        self,
        labels: Sequence[str]
    ) -> None:
        """
        Test that get_labels() properly returns the labels from the original JATIC protocol-based classifier.
        """
        mock_classifier = TestJATICImageClassifier._generate_mock_classifier(
            side_effect=[self.dummy_prob_out],
            labels=labels
        )

        inst = JATICImageClassifier(classifier=mock_classifier)

        assert inst.get_labels() == labels
