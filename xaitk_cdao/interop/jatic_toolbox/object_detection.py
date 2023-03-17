from typing import Iterable, Tuple, Dict, Hashable
import numpy as np

from smqtk_detection import DetectImageObjects
from smqtk_image_io import AxisAlignedBoundingBox

from jatic_toolbox.protocols import ObjectDetector


class JATICDetector(DetectImageObjects):
    """
    Wrapper for JATIC object detection protocol.
    """

    def __init__(self, detector: ObjectDetector):
        self._detector = detector

    def detect_objects(
      self,
      img_iter: Iterable[np.ndarray]
    ) -> Iterable[Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]:
        detector_output = self._detector(img_iter)

        boxes = detector_output.boxes
        labels = detector_output.labels
        scores = detector_output.scores
        for boxes_per_image, labels_per_image, scores_per_image in zip(boxes, labels, scores):
            dets_dict = {}  # type: Dict[Tuple, Dict[Hashable, float]]
            for box, label, score in zip(boxes_per_image, labels_per_image, scores_per_image):
                b = tuple(box)
                if b not in dets_dict:
                    dets_dict[b] = {}
                dets_dict[b][label] = score

            dets = [(AxisAlignedBoundingBox(box[0:2], box[2:4]), score_dict) for box, score_dict in dets_dict.items()]
            yield dets

    def get_config(self) -> dict:
        return {
            "detector": self._detector
        }
