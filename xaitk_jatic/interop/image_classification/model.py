from typing import Dict, Hashable, Iterator, Sequence
import numpy as np

from smqtk_classifier.interfaces.classify_image import ClassifyImage, IMAGE_ITER_T
from smqtk_classifier.interfaces.classification_element import CLASSIFICATION_DICT_T

import maite.protocols.image_classification as ic


class JATICImageClassifier(ClassifyImage):
    """
    Adapter for JATIC classifier protocol.

    :param classifier: The JATIC protocol-based classifier.
    :param id2name: Mapping from label IDs to names.
    :param img_batch_size: Image batch size for inference.
    """

    def __init__(
        self,
        classifier: ic.Model,
        id2name: Dict[int, Hashable],
        img_batch_size: int = 1
    ):
        self._classifier = classifier
        self._id2name = dict(sorted(id2name.items()))
        self._img_batch_size = img_batch_size

    def get_labels(self) -> Sequence[Hashable]:
        return [self._id2name[id] for id in sorted(self._id2name.keys())]

    def classify_images(
      self,
      img_iter: IMAGE_ITER_T
    ) -> Iterator[CLASSIFICATION_DICT_T]:
        all_out = list()
        batch = list()

        # Convert from channels last to channels first
        # Channels first is specified in protocols v0.5.0
        def _to_channels_first(img: np.ndarray) -> np.ndarray:
            if img.ndim < 3:
                return img
            return np.moveaxis(img, -1, 0)

        # Transform outputs of a single batch
        def _generate_outputs(batch: Sequence[np.ndarray]) -> None:
            predictions = np.asarray(self._classifier(np.asarray(batch)))

            for pred in predictions:
                all_out.append({self._id2name[id]: score for id, score in enumerate(pred)})

        # Batch model passes
        for img in img_iter:
            batch.append(_to_channels_first(img))

            if len(batch) == self._img_batch_size:
                _generate_outputs(batch)
                batch = list()

        # Leftover batch
        if len(batch) > 0:
            _generate_outputs(batch)

        return iter(all_out)

    def get_config(self) -> dict:
        raise NotImplementedError(
            "Constructor arguments are not serializable as is and require"
            "further implementation to do so."
        )
