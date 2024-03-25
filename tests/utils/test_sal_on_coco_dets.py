import os
import py  # type: ignore
import pytest
import unittest.mock as mock
from unittest.mock import MagicMock

from tests import DATA_DIR

from maite.protocols import ObjectDetector

from smqtk_detection.impls.detect_image_objects.random_detector import RandomDetector

from xaitk_cdao.interop.bbox_transformer import XYXYBBoxTransformer
from xaitk_cdao.interop.object_detection import JATICDetector
from xaitk_cdao.utils.sal_on_coco_dets import maite_sal_on_coco_dets, sal_on_coco_dets

from xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise import DRISEStack

from importlib.util import find_spec


deps = ['kwcoco']
specs = [find_spec(dep) for dep in deps]
is_usable = all([spec is not None for spec in specs])

dets_file = os.path.join(DATA_DIR, 'test_dets.json')
config_file = os.path.join(DATA_DIR, 'config.json')


class TestSalOnCocoDetsNotUsable:
    """
    These tests make use of the `tmpdir` fixture from `pytest`. Find more
    information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    @mock.patch("xaitk_cdao.utils.sal_on_coco_dets.is_usable", False)
    def test_warning(self, tmpdir: py.path.local) -> None:
        """
        Test that proper warning is displayed when required dependencies are
        not installed.
        """
        output_dir = tmpdir.join('out')

        with pytest.raises(ImportError, match=r"This tool requires additional dependencies"):
            sal_on_coco_dets(
                coco_file=str(dets_file),
                output_dir=str(output_dir),
                sal_generator=DRISEStack(n=1, s=3, p1=0.5),
                blackbox_detector=RandomDetector()
            )

        assert not output_dir.check(dir=1)


@pytest.mark.skipif(not is_usable, reason="Extra 'xaitk-cdao[tools]' not installed.")
class TestSalOnCocoDets:
    """
    These tests make use of the `tmpdir` fixture from `pytest`. Find more
    information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """
    sal_generator = DRISEStack(n=1, s=3, p1=0.5)
    detector = RandomDetector()

    def test_coco_sal_gen(self, tmpdir: py.path.local) -> None:
        """
        Test saliency map generation with RandomDetector, RISEGrid, and
        DRISEScoring.
        """
        output_dir = tmpdir.join('out')

        sal_on_coco_dets(
            coco_file=str(dets_file),
            output_dir=str(output_dir),
            sal_generator=TestSalOnCocoDets.sal_generator,
            blackbox_detector=TestSalOnCocoDets.detector,
            verbose=True
        )

        # expected created directories for image saliency maps
        img_dirs = [output_dir.join(d) for d in ["test_image1", "test_image2"]]
        # detection ids that belong to each image
        img_dets = [[1, 2, 3], [5, 6]]

        assert sorted(output_dir.listdir()) == sorted(img_dirs)
        for img_dir, det_ids in zip(img_dirs, img_dets):
            map_files = [img_dir.join(f"det_{det_id}.jpeg") for det_id in det_ids]
            assert sorted(img_dir.listdir()) == sorted(map_files)

    def test_coco_sal_gen_img_overlay(self, tmpdir: py.path.local) -> None:
        """
        Test saliency map generation with RandomDetector, RISEGrid, and
        DRISEScoring with the overlay image option.
        """
        output_dir = tmpdir.join('out')

        sal_on_coco_dets(
            coco_file=str(dets_file),
            output_dir=str(output_dir),
            sal_generator=TestSalOnCocoDets.sal_generator,
            blackbox_detector=TestSalOnCocoDets.detector,
            overlay_image=True
        )

        # expected created directories for image saliency maps
        img_dirs = [output_dir.join(d) for d in ["test_image1", "test_image2"]]
        # detection ids that belong to each image
        img_dets = [[1, 2, 3], [5, 6]]

        assert sorted(output_dir.listdir()) == sorted(img_dirs)
        for img_dir, det_ids in zip(img_dirs, img_dets):
            map_files = [img_dir.join(f"det_{det_id}.jpeg") for det_id in det_ids]
            assert sorted(img_dir.listdir()) == sorted(map_files)


@pytest.mark.skipif(not is_usable, reason="Extra 'xaitk-cdao[tools]' not installed.")
class TestMAITESalOnCocoDets:
    """
    These tests make use of the `tmpdir` fixture from `pytest`. Find more
    information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    @mock.patch('xaitk_cdao.utils.sal_on_coco_dets.sal_on_coco_dets', return_value=None)
    def test_coco_sal_gen(self, patch: MagicMock, tmpdir: py.path.local) -> None:
        """
        Test workflow with MAITE detector
        """
        output_dir = tmpdir.join('out')

        sal_generator = DRISEStack(n=1, s=3, p1=0.5)
        maite_detector = MagicMock(spec=ObjectDetector)
        bbox_xform = XYXYBBoxTransformer()

        maite_sal_on_coco_dets(
            coco_file=str(dets_file),
            output_dir=str(output_dir),
            sal_generator=sal_generator,
            detector=maite_detector,
            bbox_transform=bbox_xform
        )

        # Confirm sal_on_coco_dets arguments are as expected
        kwargs = patch.call_args.kwargs
        assert kwargs["coco_file"] == str(dets_file)
        assert kwargs["output_dir"] == str(output_dir)
        assert kwargs["sal_generator"] == sal_generator
        assert isinstance(kwargs["blackbox_detector"], JATICDetector)
        assert kwargs["blackbox_detector"]._detector == maite_detector
        assert kwargs["blackbox_detector"]._bbox_transform == bbox_xform
        assert not kwargs["overlay_image"]
        assert not kwargs["verbose"]
