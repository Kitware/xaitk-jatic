from typing import (
    Callable, Dict, Hashable, Iterable, Literal,
    Optional, Tuple, Union
)
import numpy as np

from smqtk_detection import DetectImageObjects
from smqtk_image_io import AxisAlignedBoundingBox

from maite.protocols import (
    ObjectDetector, HasDetectionLogits, HasDetectionProbs,
    HasDetectionPredictions, SupportsArray
)

from xaitk_cdao.utils.data_typing import to_numpy_array

from scipy.special import softmax  # type:ignore


class JATICDetector(DetectImageObjects):
    """
    Adapter for JATIC object detection protocol.

    :param detector: The JATIC protocol-based detector.
    :param bbox_transform: Predefined bounding box format literal or callable to transform
        the JATIC detector's bboxes to AxisAlignedBoundingBoxes.
    :param preprocessor: Callable that takes a batch of data and returns a batch of data
        for any preprocessing before model inference.
    :param img_batch_size: Image batch size for inference.
    """

    def __init__(
        self,
        detector: ObjectDetector,
        bbox_transform: Union[Literal["XYXY"], Callable[[np.ndarray], Iterable[Iterable[AxisAlignedBoundingBox]]]],
        preprocessor: Optional[Callable[[SupportsArray], SupportsArray]] = None,
        img_batch_size: int = 1
    ):
        self._detector = detector
        self._labels = detector.get_labels()
        self._preprocessor = preprocessor
        self._bbox_transform = bbox_transform
        self._img_batch_size = img_batch_size

    def detect_objects(
      self,
      img_iter: Iterable[np.ndarray]
    ) -> Iterable[Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]:
        all_out = list()
        batch = list()

        # Combine JATIC detections for same bbox into one SMQTK detection
        def _transform_dets(
            bboxes: Iterable[AxisAlignedBoundingBox],
            label_ids: np.ndarray,
            probs: np.ndarray
        ) -> Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]:

            dets_dict: Dict[AxisAlignedBoundingBox, Dict[Hashable, float]] = dict()
            for box, label_id, prob in zip(bboxes, label_ids, probs):
                if box not in dets_dict:
                    dets_dict[box] = {la: 0. for la in self._labels}
                dets_dict[box][self._labels[label_id]] = prob

            return [(box, prob_dict) for box, prob_dict in dets_dict.items()]

        # Transform detections for a single batch
        def _generate_outputs(batch: SupportsArray) -> None:
            if self._preprocessor:
                batch = self._preprocessor(batch)

            detector_output = self._detector(batch)

            # Transform bboxes into correct format
            detector_boxes = to_numpy_array(detector_output.boxes)
            # TODO: Other bounding box formats
            if self._bbox_transform == "XYXY":
                smqtk_boxes = list()
                for boxes in detector_boxes:
                    smqtk_boxes.append([
                        AxisAlignedBoundingBox(box[0:2], box[2:4]) for box in boxes
                    ])
                all_boxes = smqtk_boxes  # type: Iterable[Iterable[AxisAlignedBoundingBox]]
            elif callable(self._bbox_transform):
                all_boxes = self._bbox_transform(detector_boxes)
            else:
                raise ValueError("Cannot transform bounding boxes. Unknown format.")

            # Get probabilities and transform to required output format
            if isinstance(detector_output, HasDetectionLogits):
                all_logits = to_numpy_array(detector_output.logits)
                for logits, out_boxes in zip(all_logits, all_boxes):
                    all_out.append(_transform_dets(
                        bboxes=out_boxes,
                        label_ids=np.arange(len(self._labels)),
                        probs=softmax(logits)
                    ))
            elif isinstance(detector_output, HasDetectionProbs):
                all_probs = to_numpy_array(detector_output.probs)
                for probs, out_boxes in zip(all_probs, all_boxes):
                    all_out.append(_transform_dets(
                        bboxes=out_boxes,
                        label_ids=np.arange(len(self._labels)),
                        probs=probs
                    ))
            elif isinstance(detector_output, HasDetectionPredictions):
                all_scores = to_numpy_array(detector_output.scores)
                all_label_ids = to_numpy_array(detector_output.labels)
                for scores, label_ids, out_boxes in zip(all_scores, all_label_ids, all_boxes):
                    all_out.append(_transform_dets(
                        bboxes=out_boxes,
                        label_ids=label_ids,
                        probs=scores
                    ))
            else:
                raise ValueError("Unknown detector output type")

        # Batch model passes
        for img in img_iter:
            batch.append(img)

            if len(batch) == self._img_batch_size:
                _generate_outputs(batch)
                batch = list()

        # Leftover batch
        if len(batch) > 0:
            _generate_outputs(batch)

        return all_out

    def get_config(self) -> dict:
        raise NotImplementedError(
            "Constructor arguments are not serializable as is and require"
            "further implementation to do so."
        )
