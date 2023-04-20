from typing import Hashable, Iterable, Iterator, Sequence, Union
import numpy as np

from smqtk_classifier.interfaces.classify_image import ClassifyImage
from smqtk_classifier.interfaces.classification_element import CLASSIFICATION_DICT_T

from jatic_toolbox.protocols import Classifier, HasLogits, HasProbs

from xaitk_cdao.utils.data_typing import to_numpy_array

from scipy.special import softmax  # type:ignore


class JATICImageClassifier(ClassifyImage):
    """
    Wrapper for JATIC classifier protocol.
    """

    def __init__(self, classifier: Classifier, labels: Sequence[Hashable]):
        self._classifier = classifier
        self._labels = labels

    def get_labels(self) -> Sequence[Hashable]:
        return self._labels

    def classify_images(
      self,
      img_iter: Union[np.ndarray, Iterable[np.ndarray]]
    ) -> Iterator[CLASSIFICATION_DICT_T]:
        for img in img_iter:
            classifier_output = self._classifier([img])

            if isinstance(classifier_output, HasLogits):
                probs = softmax(to_numpy_array(classifier_output.logits)[0])
            elif isinstance(classifier_output, HasProbs):
                probs = to_numpy_array(classifier_output.probs)[0]
            else:
                raise ValueError("Unknown classifier output type")

            yield dict(zip(self._labels, probs))

    def get_config(self) -> dict:
        return {
            "classifier": self._classifier,
            "labels": self._labels
        }
