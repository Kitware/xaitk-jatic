import numpy as np

from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io import AxisAlignedBoundingBox

from xaitk_cdao.interop.bbox_transformer import XYXYBBoxTransformer


class TestXYXYBBoxTransformer:
    def test_configuration(self) -> None:
        """ Test configuration stability """
        inst = XYXYBBoxTransformer()

        for i in configuration_test_helper(inst):
            assert not i.get_config()

    def test_transform(self) -> None:
        """ Test bbox transformation """
        xform = XYXYBBoxTransformer()

        bboxes = np.asarray([[[1, 2, 3, 4]]])

        expected = AxisAlignedBoundingBox(
            min_vertex=(1, 2),
            max_vertex=(3, 4)
        )

        xformed_boxes = xform(bboxes)
        assert list(list(xformed_boxes)[0])[0] == expected
