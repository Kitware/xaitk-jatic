import numpy as np
import pytest
from typing import ContextManager, Dict, Hashable, Iterator, Sequence, Union
from unittest.mock import MagicMock

from smqtk_core.configuration import configuration_test_helper
from smqtk_classifier.interfaces.classification_element import CLASSIFICATION_DICT_T

from xaitk_jatic.interop.image_classification.model import JATICImageClassifier
import maite.protocols.image_classification as ic


class TestJATICImageClassifier:
    dummy_id2name1 = {0: "A", 1: "B", 2: "C"}
    expected_labels1 = ["A", "B", "C"]
    dummy_id2name2 = {0: "D", 2: "F", 1: "E"}
    expected_labels2 = ["D", "E", "F"]

    dummy_out = np.asarray([0, 0.15, 0.8])

    @pytest.mark.parametrize("classifier, id2name, img_batch_size, expectation", [
        (MagicMock(spec=ic.Model), dummy_id2name1, 3,
            pytest.raises(NotImplementedError, match=r"Constructor arg"))
    ])
    def test_configuration(
        self,
        classifier: ic.Model,
        id2name: Dict[int, Hashable],
        img_batch_size: int,
        expectation: ContextManager
    ) -> None:
        """ Test configuration stability """

        inst = JATICImageClassifier(
            classifier=classifier,
            id2name=id2name,
            img_batch_size=img_batch_size
        )
        with expectation:
            for i in configuration_test_helper(inst):
                # TODO: Update assertions appropriately once get_config/from_config are implemented
                """
                assert i._classifier == classifier
                assert i._id2name == id2name
                assert i._img_batch_size == img_batch_size
                """

    @pytest.mark.parametrize(
        "classifier_output, id2name, img_batch_size, imgs, expected_return",
        [
            ([dummy_out], dummy_id2name1, 2,
                [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)],
                [{la: prob for la, prob in zip(expected_labels1, dummy_out)}]),
            ([dummy_out], dummy_id2name1, 1,
                [np.random.randint(0, 255, (256, 256), dtype=np.uint8)],
                [{la: prob for la, prob in zip(expected_labels1, dummy_out)}]),
            ([dummy_out] * 2, dummy_id2name1, 2,
                np.random.randint(0, 255, (2, 256, 256), dtype=np.uint8),
                [{la: prob for la, prob in zip(expected_labels1, dummy_out)}] * 2)
        ],
        ids=["single 3 channel", "single greyscale", "multiple images"]
    )
    def test_smoketest(
        self,
        classifier_output: ic.TargetBatchType,
        id2name: Dict[int, Hashable],
        img_batch_size: int,
        imgs: Union[np.ndarray, Sequence[np.ndarray]],
        expected_return: Iterator[CLASSIFICATION_DICT_T],
    ) -> None:
        """
        Test that MAITE classifier output is transformed appropriately.
        """
        mock_classifier = MagicMock(
            spec=ic.Model,
            return_value=classifier_output
        )

        inst = JATICImageClassifier(
            classifier=mock_classifier,
            id2name=id2name,
            img_batch_size=img_batch_size
        )

        res = list(inst.classify_images(imgs))
        assert res == expected_return

    @pytest.mark.parametrize("id2name, expected_labels", [
        (dummy_id2name1, expected_labels1),
        (dummy_id2name2, expected_labels2),
    ])
    def test_labels(
        self,
        id2name: Dict[int, Hashable],
        expected_labels: Sequence[Hashable]
    ) -> None:
        """
        Test that get_labels() returns the correct labels.
        """

        inst = JATICImageClassifier(
            classifier=MagicMock(spec=ic.Model),
            id2name=id2name
        )

        assert inst.get_labels() == expected_labels
