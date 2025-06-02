from __future__ import annotations

from collections.abc import Hashable, Iterator, Sequence
from contextlib import AbstractContextManager
from unittest.mock import MagicMock

import maite.protocols.image_classification as ic
import numpy as np
import pytest
from smqtk_classifier.interfaces.classification_element import CLASSIFICATION_DICT_T
from smqtk_core.configuration import configuration_test_helper

from xaitk_jatic.interop.image_classification.model import JATICImageClassifier

rng = np.random.default_rng()


class TestJATICImageClassifier:
    dummy_id_to_name_1 = {0: "A", 1: "B", 2: "C"}
    expected_labels = [0, 1, 2]
    dummy_id_to_name_2 = {0: "D", 2: "D", 1: "E"}

    dummy_out = np.asarray([0, 0.15, 0.8])

    @pytest.mark.parametrize(
        ("classifier", "id_to_name", "img_batch_size", "expectation"),
        [
            (
                MagicMock(spec=ic.Model),
                dummy_id_to_name_1,
                3,
                pytest.raises(NotImplementedError, match=r"Constructor arg"),
            ),
        ],
    )
    def test_configuration(
        self,
        classifier: ic.Model,
        id_to_name: dict[int, Hashable],
        img_batch_size: int,
        expectation: AbstractContextManager,
    ) -> None:
        """Test configuration stability."""
        inst = JATICImageClassifier(classifier=classifier, ids=list(id_to_name.keys()), img_batch_size=img_batch_size)
        with expectation:
            configuration_test_helper(inst)

    @pytest.mark.parametrize(
        (
            "classifier_output",
            "id_to_name",
            "img_batch_size",
            "imgs",
            "expected_return",
        ),
        [
            (
                [dummy_out],
                dummy_id_to_name_1,
                1,
                [rng.integers(0, 255, (3, 256, 256), dtype=np.uint8)],
                [dict(zip(expected_labels, dummy_out))],
            ),
            (
                [dummy_out],
                dummy_id_to_name_1,
                1,
                [rng.integers(0, 255, (256, 256), dtype=np.uint8)],
                [dict(zip(expected_labels, dummy_out))],
            ),
            (
                [dummy_out] * 2,
                dummy_id_to_name_1,
                2,
                rng.integers(0, 255, (2, 256, 256), dtype=np.uint8),
                [dict(zip(expected_labels, dummy_out)), dict(zip(expected_labels, dummy_out))],
            ),
        ],
        ids=["single 3 channel", "single greyscale", "multiple images"],
    )
    def test_smoketest(
        self,
        classifier_output: ic.TargetBatchType,
        id_to_name: dict[int, Hashable],
        img_batch_size: int,
        imgs: np.ndarray | Sequence[np.ndarray],
        expected_return: Iterator[CLASSIFICATION_DICT_T],
    ) -> None:
        """Test that MAITE classifier output is transformed appropriately."""
        mock_classifier = MagicMock(spec=ic.Model, return_value=classifier_output)

        inst = JATICImageClassifier(
            classifier=mock_classifier,
            ids=list(id_to_name.keys()),
            img_batch_size=img_batch_size,
        )

        res = list(inst.classify_images(imgs))
        assert res == expected_return

    @pytest.mark.parametrize(
        ("id_to_name", "expected_labels"),
        [
            (dummy_id_to_name_1, expected_labels),
            (dummy_id_to_name_2, expected_labels),
        ],
    )
    def test_labels(self, id_to_name: dict[int, Hashable], expected_labels: Sequence[Hashable]) -> None:
        """Test that get_labels() returns the correct labels."""
        inst = JATICImageClassifier(classifier=MagicMock(spec=ic.Model), ids=list(id_to_name.keys()))

        assert inst.get_labels() == expected_labels
