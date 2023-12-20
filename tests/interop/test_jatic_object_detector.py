import numpy as np
import pytest
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock
from scipy.special import softmax  # type:ignore
from typing import (
    Any, Callable, ContextManager, Dict, Hashable, Iterable, Literal,
    Optional, Protocol, Sequence, Tuple, Union
)

from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io import AxisAlignedBoundingBox

from xaitk_cdao.interop.object_detection import JATICDetector
from maite.protocols import (
    ObjectDetector, HasDetectionLogits, HasDetectionProbs,
    HasDetectionPredictions, SupportsArray
)


class TestJATICObjectDetector:
    dummy_labels1 = ["A", "B", "C"]
    dummy_labels2 = ["D", "E", "F"]

    dummy_boxes1 = np.asarray([[[1, 2, 3, 4], [1, 2, 3, 4], [5, 6, 7, 8]]])
    dummy_boxes2 = np.asarray([[[9, 10, 11, 12], [9, 10, 11, 12], [9, 10, 11, 12]]])

    @staticmethod
    def _transform_bboxes(all_boxes: np.ndarray) -> Iterable[Iterable[AxisAlignedBoundingBox]]:
        """
        Dummy transform bboxes function.
        """
        out_boxes = list()
        for boxes in all_boxes:
            out_boxes.append([
                AxisAlignedBoundingBox(box[0:2], box[2:4]) for box in boxes
            ])
        return out_boxes

    # HasDetectionProbs
    dummy_probs = np.asarray([[0.15, 0.35, 0.5]])
    dummy_prob_labels = np.asarray([[0, 1, 2]])
    dummy_probs_out = MagicMock(
        spec=HasDetectionProbs,
        boxes=dummy_boxes2,
        labels=dummy_prob_labels,
        probs=dummy_probs
    )
    dummy_probs_expected = [[
        (AxisAlignedBoundingBox([9, 10], [11, 12]), {"D": 0.15, "E": 0.35, "F": 0.5})
    ]]

    # HasDetectionLogits
    dummy_logits = np.asarray([[1.5, 0.4, -0.1]])
    dummy_logit_labels = np.asarray([[0, 1, 2]])
    dummy_logits_out = MagicMock(
        spec=HasDetectionLogits,
        boxes=dummy_boxes2,
        labels=dummy_labels1,
        logits=dummy_logits
    )
    dummy_logit_probs = softmax(dummy_logits[0])
    dummy_logits_expected = [[
        (
            AxisAlignedBoundingBox([9, 10], [11, 12]),
            {
                "A": dummy_logit_probs[0],
                "B": dummy_logit_probs[1],
                "C": dummy_logit_probs[2]
            }
        )
    ]]

    # HasDetectionPredictions
    dummy_scores = np.asarray([[0.25, 0.75, 0.95]])
    dummy_score_labels = np.asarray([[0, 2, 0]])
    dummy_scores_out = MagicMock(
        spec=HasDetectionPredictions,
        boxes=dummy_boxes1,
        labels=dummy_score_labels,
        scores=dummy_scores
    )
    dummy_scores_expected = [[
            (AxisAlignedBoundingBox([1, 2], [3, 4]), {"A": 0.25, "B": 0, "C": 0.75}),
            (AxisAlignedBoundingBox([5, 6], [7, 8]), {"A": 0.95, "B": 0, "C": 0})
    ]]

    # Unknown output
    class _FakeOutput(Protocol):
        boxes: SupportsArray
        fake_output: SupportsArray
    dummy_fake_output = np.asarray([[0.3, 0.2]])
    dummy_unknown_out = MagicMock(
        spec=_FakeOutput,
        boxes=dummy_boxes2,
        fake_output=dummy_fake_output
    )

    @pytest.mark.parametrize("detector, bbox_transform, preprocessor, img_batch_size, expectation", [
        (MagicMock(spec=ObjectDetector), lambda x: x, lambda x: x, 2,
            pytest.raises(NotImplementedError, match=r"Constructor arg")),
        (MagicMock(spec=ObjectDetector), "XYXY", None, 4,
            pytest.raises(NotImplementedError, match=r"Constructor arg")),
    ])
    def test_configuration(
        self,
        detector: ObjectDetector,
        bbox_transform: Union[Literal["XYXY"], Callable[[np.ndarray], Iterable[Iterable[AxisAlignedBoundingBox]]]],
        preprocessor: Optional[Callable[[SupportsArray], SupportsArray]],
        img_batch_size: int,
        expectation: ContextManager
    ) -> None:
        """ Test configuration stability. """

        inst = JATICDetector(
            detector=detector,
            bbox_transform=bbox_transform,
            preprocessor=preprocessor,
            img_batch_size=img_batch_size
        )
        with expectation:
            for i in configuration_test_helper(inst):
                # TODO: Update assertions appropriately once get_config/from_config are implemented
                """
                assert i._detector == detector
                assert i._bbox_transform == bbox_transform
                assert i._preprocessor == preprocessor
                assert i._img_batch_size == img_batch_size
                """

    @staticmethod
    def _generate_mock_detector(side_effect: Any, labels: Sequence[str]) -> ObjectDetector:
        """
        Generate a mock detector with the given side effect(s) and get_labels return.
        """
        mock_detector = MagicMock(
            spec=ObjectDetector,
        )
        mock_detector.side_effect = side_effect
        mock_detector.get_labels = MagicMock(return_value=labels)
        return mock_detector

    @pytest.mark.parametrize(
        "detector_side_effect, labels, bbox_transform, preprocessor, img_batch_size, expected_return, expectation",
        [
            ([dummy_probs_out], dummy_labels2, "XYXY", None, 1, dummy_probs_expected, does_not_raise()),
            ([dummy_logits_out], dummy_labels1, None, None, 2, dummy_logits_expected, does_not_raise()),
            ([dummy_scores_out], dummy_labels1, "XYXY", lambda x: x, 1, dummy_scores_expected, does_not_raise()),
            ([dummy_probs_out], dummy_labels2, "ZZZZ", None, 1, dummy_probs_expected,
                pytest.raises(ValueError, match=r"Cannot transform bounding boxes. Unknown format.")),
            ([dummy_unknown_out], dummy_labels1, "XYXY", None, 2, dummy_logits_expected,
                pytest.raises(ValueError, match=r"Unknown detector output type"))
        ],
        ids=[
            "HasDetectionProbs", "HasDetectionLogits", "HasDetectionPredictions", "UnknownBboxFmt",
            "UnknownOutput"
        ]
    )
    def test_smoketest(
        self,
        detector_side_effect: Any,
        labels: Sequence[str],
        bbox_transform: Optional[Union[Literal["XYXY"],
                                 Callable[[np.ndarray], Iterable[Iterable[AxisAlignedBoundingBox]]]]],
        preprocessor: Optional[Callable[[SupportsArray], SupportsArray]],
        img_batch_size: int,
        expected_return: Iterable[Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]],
        expectation: ContextManager
    ) -> None:
        """
        Ensure we can handle the various expected output types for a JATIC protocol-based classifier.
        """
        if bbox_transform is None:
            bbox_transform = TestJATICObjectDetector._transform_bboxes

        mock_detector = TestJATICObjectDetector._generate_mock_detector(
            side_effect=detector_side_effect,
            labels=labels
        )

        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        with expectation:
            inst = JATICDetector(
                detector=mock_detector,
                bbox_transform=bbox_transform,
                preprocessor=preprocessor,
                img_batch_size=img_batch_size
            )
            res = list(inst.detect_objects([dummy_image]))
            assert res == expected_return
