"""
This module provides dataset classes for working with object detection data in the JATIC framework.
It includes adapters for COCO-format datasets and general datasets for varying-sized images.

Classes:
    JATICDetectionTarget: A dataclass for storing detection results, including bounding boxes, labels, and scores.
    COCOJATICObjectDetectionDataset: Converts a COCO dataset to be compatible with the JATIC object detection protocol.
    JATICObjectDetectionDataset: A generic wrapper for object detection datasets with varying-sized images.

Dependencies:
    - kwcoco: For working with COCO-format datasets.
    - numpy: For numerical operations.
    - maite.protocols.object_detection: For object detection protocols.
    - PIL: For image processing.
"""

import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from maite.protocols.object_detection import (
    Dataset,
    DatumMetadataType,
    InputType,
    TargetType,
)
from PIL import Image  # type: ignore

try:
    import kwcoco  # type: ignore

    is_usable = True
except ImportError:
    is_usable = False

OBJ_DETECTION_DATUM_T = tuple[InputType, TargetType, DatumMetadataType]

LOG = logging.getLogger(__name__)


@dataclass
class JATICDetectionTarget:
    """Dataclass for the datum-level JATIC output detection format."""

    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


if not is_usable:
    LOG.warning(
        "COCOJATICObjectDetectionDataset requires additional dependencies, please install 'xaitk-jatic[tools]'.",
    )
else:

    class COCOJATICObjectDetectionDataset(Dataset):
        """Dataset class to convert a COCO dataset to a dataset compliant with JATIC's Object Detection protocol.

        Parameters
        ----------
        kwcoco_dataset : kwcoco.CocoDataset
            The kwcoco COCODataset object.
        image_metadata : dict[int, dict[str, Any]]
            A dict of per-image metadata, by image id. Any metadata required by a perturber should be provided.
        skip_no_anns: bool
            If True, do not include images with no annotations in the dataset
        """

        def __init__(  # noqa: C901
            self,
            kwcoco_dataset: kwcoco.CocoDataset,
            image_metadata: dict[int, dict[str, Any]],
            skip_no_anns: bool = False,
        ) -> None:
            """
            Initialize the COCOJATICObjectDetectionDataset.

            Args:
                kwcoco_dataset (kwcoco.CocoDataset): The COCO dataset object.
                image_metadata (dict[int, dict[str, Any]]): Metadata for each image by ID.
                skip_no_anns (bool): Whether to skip images without annotations. Defaults to False.

            Raises:
                ValueError: If metadata is missing for any image in the dataset.
            """
            self._kwcoco_dataset = kwcoco_dataset

            self._image_ids = list()
            self._annotations = dict()

            for _, img_id in enumerate(kwcoco_dataset.imgs.keys()):
                bboxes = np.empty((0, 4))
                labels = []
                scores = []

                if img_id in kwcoco_dataset.gid_to_aids and len(kwcoco_dataset.gid_to_aids[img_id]) > 0:
                    det_ids = kwcoco_dataset.gid_to_aids[img_id]
                    for det_id in det_ids:
                        ann = kwcoco_dataset.anns[det_id]

                        labels.append(ann["category_id"])

                        if "score" in ann:
                            scores.append(ann["score"])
                        elif "prob" in ann:
                            scores.append(max(ann["prob"]))
                        else:
                            scores.append(1.0)

                        x, y, w, h = ann["bbox"]
                        bbox = [x, y, x + w, y + h]
                        bboxes = np.vstack((bboxes, bbox))
                elif skip_no_anns:
                    continue

                img_file = kwcoco_dataset.get_image_fpath(img_id)
                if not img_file.exists():
                    continue
                self._image_ids.append(img_id)
                self._annotations[img_id] = JATICDetectionTarget(
                    boxes=bboxes,
                    labels=np.asarray(labels),
                    scores=np.asarray(scores),
                )

            self._image_metadata = copy.deepcopy(image_metadata)
            self._image_metadata = {
                image_id: image_md for image_id, image_md in self._image_metadata.items() if image_id in self._image_ids
            }
            if len(self._image_metadata) != len(self._image_ids):
                raise ValueError("Image metadata length mismatch, metadata needed for every image.")

        def __len__(self) -> int:
            """Returns the number of images in the dataset."""
            return len(self._image_ids)

        def __getitem__(self, index: int) -> OBJ_DETECTION_DATUM_T:
            """Returns the dataset object at the given index."""
            image_id = self._image_ids[index]
            img_file = self._kwcoco_dataset.get_image_fpath(image_id)
            image = Image.open(img_file)
            width, height = image.size

            gid_to_aids = self._kwcoco_dataset.gid_to_aids

            self._image_metadata[image_id].update(
                dict(
                    id=image_id,
                    image_info=dict(width=width, height=height, file_name=img_file),
                    det_ids=(list(gid_to_aids[image_id]) if image_id in gid_to_aids else list()),
                ),
            )

            image_array = np.asarray(image)
            if image.mode == "L":
                image_array = np.expand_dims(image_array, axis=2)

            input_img, dets, metadata = (
                np.asarray(np.transpose(image_array, axes=(2, 0, 1))),
                self._annotations[image_id],
                self._image_metadata[image_id],
            )

            return input_img, dets, metadata


class JATICObjectDetectionDataset(Dataset):
    """Implementation of the JATIC Object Detection dataset wrapper for dataset images of varying sizes.

    Parameters
    ----------
    imgs : Sequence[np.ndarray]
        Sequence of images.
    dets : Sequence[ObjectDetectionTarget]
        Sequence of detections for each image.
    metadata : Sequence[dict[str, Any]]
        Sequence of custom metadata for each image.
    """

    def __init__(
        self,
        imgs: Sequence[np.ndarray],
        dets: Sequence[TargetType],
        metadata: Sequence[DatumMetadataType],
    ) -> None:
        """
        Initialize the JATICObjectDetectionDataset.

        Args:
            imgs (Sequence[np.ndarray]): Sequence of images in the dataset.
            dets (Sequence[TargetType]): Sequence of detection targets for the images.
            metadata (Sequence[DatumMetadataType]): Sequence of metadata dictionaries.
        """
        self.imgs = imgs
        self.dets = dets
        self.metadata = metadata

    def __len__(self) -> int:
        """
        Get the number of images in the dataset.

        Returns:
            int: The number of images.
        """
        return len(self.imgs)

    def __getitem__(self, index: int) -> OBJ_DETECTION_DATUM_T:
        """
        Retrieve a dataset item by index.

        Args:
            index (int): The index of the dataset item to retrieve.

        Returns:
            OBJ_DETECTION_DATUM_T: A tuple containing:
                - Input image as a numpy array.
                - Detection targets for the image.
                - Metadata for the image.
        """
        return self.imgs[index], self.dets[index], self.metadata[index]
