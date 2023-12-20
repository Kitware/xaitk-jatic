from typing import Callable, Hashable, Iterable, Iterator, Sequence, Union, Optional
import numpy as np

from smqtk_classifier.interfaces.classify_image import ClassifyImage
from smqtk_classifier.interfaces.classification_element import CLASSIFICATION_DICT_T

from maite.protocols import (
    ImageClassifier, HasLogits, HasProbs, HasScores, SupportsArray
)

from xaitk_cdao.utils.data_typing import to_numpy_array

from scipy.special import softmax  # type:ignore


class JATICImageClassifier(ClassifyImage):
    """
    Adapter for JATIC classifier protocol.

    :param classifier: The JATIC protocol-based classifier.
    :param preprocessor: Callable that takes a batch of data and returns a batch of data
        for any preprocessing before model inference.
    :param img_batch_size: Image batch size for inference.
    """

    def __init__(
        self,
        classifier: ImageClassifier,
        preprocessor: Optional[Callable[[SupportsArray], SupportsArray]] = None,
        img_batch_size: int = 1
    ):
        self._classifier = classifier
        self._labels = classifier.get_labels()
        self._preprocessor = preprocessor
        self._img_batch_size = img_batch_size

    def get_labels(self) -> Sequence[Hashable]:
        return self._labels

    def classify_images(
      self,
      img_iter: Union[np.ndarray, Iterable[np.ndarray]]
    ) -> Iterator[CLASSIFICATION_DICT_T]:
        all_out = list()
        batch = list()

        # Transform outputs of a single batch
        def _generate_outputs(batch: SupportsArray) -> None:
            if self._preprocessor:
                batch = self._preprocessor(batch)

            classifier_output = self._classifier(batch)

            # Get probabilities and transform to required output format
            if isinstance(classifier_output, HasLogits):
                all_logits = to_numpy_array(classifier_output.logits)
                for logits in all_logits:
                    all_out.append(dict(zip(self._labels, softmax(logits))))
            elif isinstance(classifier_output, HasProbs):
                all_probs = to_numpy_array(classifier_output.probs)
                for probs in all_probs:
                    all_out.append(dict(zip(self._labels, probs)))
            elif isinstance(classifier_output, HasScores):
                all_scores = to_numpy_array(classifier_output.scores)
                all_label_ids = to_numpy_array(classifier_output.labels)
                for scores, label_ids in zip(all_scores, all_label_ids):
                    out = {la: 0. for la in self._labels}
                    for score, label_id in zip(scores, label_ids):
                        out[self._labels[label_id]] = score
                    all_out.append(dict(out))
            else:
                raise ValueError("Unknown classifier output type")

        # Batch model passes
        for img in img_iter:
            batch.append(img)

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
