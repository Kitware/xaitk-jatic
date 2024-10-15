from typing import Dict, Hashable, List, Tuple

import numpy as np
from maite.protocols.object_detection import Dataset, Model
from smqtk_core.configuration import to_config_dict
from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects
from xaitk_saliency import GenerateObjectDetectorBlackboxSaliency

from xaitk_jatic.interop.object_detection.model import JATICDetector


def compute_sal_maps(
    dataset: Dataset,
    sal_generator: GenerateObjectDetectorBlackboxSaliency,
    blackbox_detector: DetectImageObjects,
    num_classes: int,
) -> Tuple[List[np.ndarray], Dict]:
    """Generate saliency maps for the provided dataset.

    :param dataset: MAITE dataset
    :param sal_generator: ``GenerateObjectDetectorBlackboxSaliency`` generator
    :param blackbox_detector: ``DetectImageObjects`` detector
    :param num_classes: Number of classes potentially predicted by the detector.
    """
    img_sal_maps = list()
    for dset_idx in range(len(dataset)):
        ref_img, dets, _ = dataset[dset_idx]

        scores = np.asarray(dets.scores)
        score_matrix = np.zeros((len(scores), num_classes))
        for idx, (lbl, score) in enumerate(zip(np.asarray(dets.labels), scores)):
            score_matrix[idx][lbl] = score

        img_sal_maps.append(
            sal_generator(
                np.asarray(ref_img),
                np.asarray(dets.boxes),
                score_matrix,
                blackbox_detector,
            )
        )

    return img_sal_maps, to_config_dict(sal_generator)


def sal_on_dets(
    dataset: Dataset,
    sal_generator: GenerateObjectDetectorBlackboxSaliency,
    detector: Model,
    id_to_name: Dict[int, Hashable],
    img_batch_size: int = 1,
) -> Tuple[List[np.ndarray], Dict]:
    """Generate saliency maps for provided dataset.

    :param dataset: MAITE dataset
    :param sal_generator: ``GenerateObjectDetectorBlackboxSaliency`` generator
    :param detector: MAITE detector
    :param id_to_name: Mapping from label IDs to names
    :param img_batch_size: Image batch size for inference
    """
    return compute_sal_maps(
        dataset=dataset,
        sal_generator=sal_generator,
        blackbox_detector=JATICDetector(detector=detector, id_to_name=id_to_name, img_batch_size=img_batch_size),
        num_classes=len(id_to_name),
    )
