import abc
from typing import Iterable

from smqtk_core import Plugfigurable
from smqtk_image_io import AxisAlignedBoundingBox

from maite.protocols import SupportsArray

from xaitk_cdao.utils.data_typing import to_numpy_array


class BBoxTransformer(Plugfigurable):
    """
    A bounding box transformer to convert model outputs into AxisAlignedBoundingBox bounding boxes
    """

    @abc.abstractmethod
    def __call__(self, x: SupportsArray) -> Iterable[Iterable[AxisAlignedBoundingBox]]:
        """
        Perform the transformation on the input bounding boxes.

        :param x: Input array-like bounding boxes to be transformed.
        :returns: AxisAlignedBoundingBox bounding boxes
        """


class XYXYBBoxTransformer(BBoxTransformer):
    def __call__(self, x: SupportsArray) -> Iterable[Iterable[AxisAlignedBoundingBox]]:
        """
        Perform the transformation on the input XYXY bounding boxes.

        :param x: Input array-like XYXY bounding boxes to be transformed.
        :returns: AxisAlignedBoundingBox bounding boxes
        """
        detector_boxes = to_numpy_array(x)
        smqtk_boxes = list()
        for boxes in detector_boxes:
            smqtk_boxes.append([
                AxisAlignedBoundingBox(box[0:2], box[2:4]) for box in boxes
            ])

        return smqtk_boxes

    def get_config(self) -> dict:
        return {}
