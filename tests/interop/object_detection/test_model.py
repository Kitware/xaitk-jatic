from collections.abc import Hashable, Iterable, Sequence
from contextlib import AbstractContextManager
from typing import Union
from unittest.mock import MagicMock

import maite.protocols.object_detection as od
import numpy as np
import pytest
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io import AxisAlignedBoundingBox

from xaitk_jatic.interop.object_detection.model import JATICDetector

rng = np.random.default_rng()


class TestJATICObjectDetector:
    dummy_id_to_name = {0: "A", 1: "B", 2: "C"}
    dummy_boxes = np.asarray([[1, 2, 3, 4], [1, 2, 3, 4], [5, 6, 7, 8]])
    dummy_scores = np.asarray([0.25, 0.75, 0.95])
    dummy_labels = np.asarray([0, 2, 0])
    dummy_out = MagicMock(
        spec=od.ObjectDetectionTarget,
        boxes=dummy_boxes,
        labels=dummy_labels,
        scores=dummy_scores,
    )
    dummy_expected = [
        [
            (AxisAlignedBoundingBox([1, 2], [3, 4]), {0: 0.25, 1: 0.0, 2: 0.75}),
            (AxisAlignedBoundingBox([5, 6], [7, 8]), {0: 0.95, 1: 0.0, 2: 0.0}),
        ],
    ]

    @pytest.mark.parametrize(
        ("detector", "id_to_name", "img_batch_size", "expectation"),
        [
            (
                MagicMock(spec=od.Model),
                dummy_id_to_name,
                2,
                pytest.raises(NotImplementedError, match=r"Constructor arg"),
            ),
        ],
    )
    def test_configuration(
        self,
        detector: od.Model,
        id_to_name: dict[int, Hashable],
        img_batch_size: int,
        expectation: AbstractContextManager,
    ) -> None:
        """Test configuration stability."""
        inst = JATICDetector(detector=detector, ids=id_to_name.keys(), img_batch_size=img_batch_size)
        with expectation:
            configuration_test_helper(inst)

    @pytest.mark.parametrize(
        ("detector_output", "id_to_name", "img_batch_size", "imgs", "expected_return"),
        [
            (
                [dummy_out],
                dummy_id_to_name,
                1,
                [rng.integers(0, 255, (3, 256, 256), dtype=np.uint8)],
                dummy_expected,
            ),
            (
                [dummy_out],
                dummy_id_to_name,
                1,
                [rng.integers(0, 255, (256, 256), dtype=np.uint8)],
                dummy_expected,
            ),
            (
                [dummy_out] * 2,
                dummy_id_to_name,
                2,
                rng.integers(0, 255, (2, 256, 256), dtype=np.uint8),
                [dummy_expected[0]] * 2,
            ),
            (
                [MagicMock(spec=od.ObjectDetectionTarget, boxes=[], labels=[], scores=[])],
                dummy_id_to_name,
                1,
                rng.integers(0, 255, (1, 256, 256), dtype=np.uint8),
                [[]],
            ),
        ],
        ids=["single 3 channel", "single greyscale", "multiple images", "no dets"],
    )
    def test_smoketest(
        self,
        detector_output: Sequence[od.ObjectDetectionTarget],
        id_to_name: dict[int, Hashable],
        img_batch_size: int,
        imgs: Union[np.ndarray, Sequence[np.ndarray]],
        expected_return: Iterable[Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]],
    ) -> None:
        """Test that MAITE detector output is transformed appropriately."""
        mock_detector = MagicMock(spec=od.Model, return_value=detector_output)

        inst = JATICDetector(detector=mock_detector, ids=id_to_name.keys(), img_batch_size=img_batch_size)
        res = list(inst.detect_objects(imgs))
        assert res == expected_return
