from typing import Iterable, Tuple, Dict, Hashable, List, Optional
import numpy as np
import math
import logging
import warnings

from smqtk_detection import DetectImageObjects
from smqtk_image_io import AxisAlignedBoundingBox

from jatic_toolbox.protocols.object_detection import (
    ObjectDetection,
    ObjectDetectionOutput,
)


class JATICDetector(DetectImageObjects):
    """
    Wrapper for JATIC object detection protocol.
    """

    def __init__(self, detector: ObjectDetection):
        self._detector = detector

    def detect_objects(
      self,
      img_iter: Iterable[np.ndarray]
    ) -> Iterable[Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]:
        detector_output = self._detector(img_iter)

        res = []
        for output in detector_output:
            dets = []
            for b, scores in zip(output.boxes, output.scores):
                dets.append((AxisAlignedBoundingBox(b.min_vertex, b.max_vertex), scores))
            res.append(dets)

        return res

    # requried by interface
    def get_config(self):
        return {}
