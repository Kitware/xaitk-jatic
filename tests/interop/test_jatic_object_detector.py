from unittest.mock import MagicMock
import numpy as np

from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io import AxisAlignedBoundingBox

from xaitk_cdao.interop.jatic_toolbox.object_detection import JATICDetector
from jatic_toolbox.protocols import (
    HasObjectDetections,
    ObjectDetector
)


class TestJATICObjectDetector:
    def test_configuration(self) -> None:
        """ Test configuration stability """
        mock_det = MagicMock(spec=ObjectDetector)

        inst = JATICDetector(mock_det)
        for i in configuration_test_helper(inst):
            assert i._detector == mock_det

    def test_smoketest(self) -> None:
        """
        Run on a dummy image for basic sanity.
        No value assertions, this is for making sure that as-is functionality
        does not error for a mostly trivial case (no outputs even expected on
        such a random image).
        """
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        mock_det_output = MagicMock(
                                    spec=HasObjectDetections,
                                    boxes=[[[1, 2, 3, 4], [1, 2, 3, 4], [5, 6, 7, 8]]],
                                    labels=[['lbl1', 'lbl2', 'lbl1']],
                                    scores=[[0.25, 0.75, 0.95]]
                                    )
        mock_det = MagicMock(spec=ObjectDetector, return_value=mock_det_output)

        inst = JATICDetector(mock_det)
        res = list(inst.detect_objects([dummy_image]))
        expected_res = [[
            (AxisAlignedBoundingBox([1, 2], [3, 4]), {'lbl1': 0.25, 'lbl2': 0.75}),
            (AxisAlignedBoundingBox([5, 6], [7, 8]), {'lbl1': 0.95})
        ]]
        assert res == expected_res
