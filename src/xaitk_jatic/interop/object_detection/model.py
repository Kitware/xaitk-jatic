from typing import Dict, Hashable, Iterable, Sequence, Tuple

import maite.protocols.object_detection as od
import numpy as np
from smqtk_detection import DetectImageObjects
from smqtk_image_io import AxisAlignedBoundingBox


class JATICDetector(DetectImageObjects):
    """Adapter for JATIC object detection protocol.

    :param detector: The JATIC protocol-based detector.
    :param id_to_name: Mapping from label IDs to names.
    :param img_batch_size: Image batch size for inference.
    """

    def __init__(
        self,
        detector: od.Model,
        id_to_name: Dict[int, Hashable],
        img_batch_size: int = 1,
    ):
        self._detector = detector
        self._id_to_name = dict(sorted(id_to_name.items()))
        self._img_batch_size = img_batch_size

    def detect_objects(
        self, img_iter: Iterable[np.ndarray]
    ) -> Iterable[Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]:
        all_out = list()

        # Convert from channels last to channels first
        # Channels first is specified in protocols v0.5.0
        def _to_channels_first(img: np.ndarray) -> np.ndarray:
            if img.ndim < 3:
                return img
            return np.moveaxis(img, -1, 0)

        # Combine JATIC detections for same bbox into one DetectImageObject detection
        def _xform_dets(
            bboxes: Iterable[AxisAlignedBoundingBox],
            labels: Sequence[Hashable],
            probs: np.ndarray,
        ) -> Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]:
            dets_dict: Dict[AxisAlignedBoundingBox, Dict[Hashable, float]] = dict()
            for box, label, prob in zip(bboxes, labels, probs):
                if box not in dets_dict:
                    dets_dict[box] = {la: 0.0 for la in self._id_to_name.values()}
                dets_dict[box][label] = prob

            return [(box, prob_dict) for box, prob_dict in dets_dict.items()]

        # Convert bboxes into AxisAlignedBoundingBoxes
        # Protocols specify X0, Y0, X1, Y1 format in v0.5.0
        def _xform_bbox(box: np.ndarray) -> AxisAlignedBoundingBox:
            return AxisAlignedBoundingBox(box[0:2], box[2:4])

        # Convert protocol-based detector output into DetectImageObject format
        def _generate_outputs(batch: Sequence[np.ndarray]) -> None:
            predictions = self._detector(batch)

            for pred in predictions:
                boxes = [_xform_bbox(box) for box in np.asarray(pred.boxes)]
                labels = [self._id_to_name[la] for la in np.asarray(pred.labels)]
                scores = np.asarray(pred.scores)

                all_out.append(_xform_dets(bboxes=boxes, labels=labels, probs=scores))

        # Batch model passes
        batch = list()
        for img in img_iter:
            batch.append(_to_channels_first(img))

            if len(batch) == self._img_batch_size:
                _generate_outputs(batch)
                batch = list()

        # Leftover batch
        if len(batch) > 0:
            _generate_outputs(batch)

        return all_out

    def get_config(self) -> dict:
        raise NotImplementedError(
            "Constructor arguments are not serializable as is and require further implementation to do so."
        )
