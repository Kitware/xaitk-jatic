import numpy as np
import os
import py  # type: ignore
import pytest
import random
import unittest.mock as mock
from dataclasses import dataclass
from random import randrange
from typing import List, Sequence

from tests import DATA_DIR

from maite.protocols import SupportsArray, HasDetectionPredictions

from smqtk_detection.impls.detect_image_objects.random_detector import RandomDetector

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
    bbox_detector = RandomDetector()

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
            blackbox_detector=TestSalOnCocoDets.bbox_detector,
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
            blackbox_detector=TestSalOnCocoDets.bbox_detector,
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


##########################################################################################
# MAITE Random Detector Classes
##########################################################################################
@dataclass
class MAITEData:
    scores: SupportsArray
    labels: SupportsArray
    boxes: SupportsArray


@dataclass
class MAITEModelMetadata:
    model_name: str = "model_name"
    provider: str = "provider_name"
    task: str = "task"


class RandMAITEModel:
    metadata = MAITEModelMetadata()

    def _gen_bbox(self, img_w: int, img_h: int) -> List[float]:
        xs = [randrange(img_w), randrange(img_w)]
        ys = [randrange(img_h), randrange(img_h)]

        return [min(xs), min(ys), max(xs), max(ys)]

    def __call__(self, data: np.ndarray) -> HasDetectionPredictions:
        scores = list()
        labels = list()
        boxes = list()
        for d in data:
            img_h = d.shape[0]
            img_w = d.shape[1]

            num_dets = randrange(10)

            scores.append([random.uniform(0, 1) for _ in range(num_dets)])
            labels.append([randrange(len(self.get_labels())) for _ in range(num_dets)])
            boxes.append([self._gen_bbox(img_w, img_h) for _ in range(num_dets)])

        return MAITEData(np.asarray(scores), np.asarray(labels), np.asarray(boxes))

    def get_labels(self) -> Sequence[str]:
        return ["cat0", "cat1", "cat2"]
########################################################################################


@pytest.mark.skipif(not is_usable, reason="Extra 'xaitk-cdao[tools]' not installed.")
class TestMAITESalOnCocoDets:
    """
    These tests make use of the `tmpdir` fixture from `pytest`. Find more
    information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    sal_generator = DRISEStack(n=1, s=3, p1=0.5)
    maite_model = RandMAITEModel()

    def test_coco_sal_gen(self, tmpdir: py.path.local) -> None:
        """
        Test saliency map generation with RandomDetector, RISEGrid, and
        DRISEScoring.
        """
        output_dir = tmpdir.join('out')

        maite_sal_on_coco_dets(
            coco_file=str(dets_file),
            output_dir=str(output_dir),
            sal_generator=TestSalOnCocoDets.sal_generator,
            maite_detector=TestMAITESalOnCocoDets.maite_model,
            bbox_transform="XYXY"
        )

        # expected created directories for image saliency maps
        img_dirs = [output_dir.join(d) for d in ["test_image1", "test_image2"]]
        # detection ids that belong to each image
        img_dets = [[1, 2, 3], [5, 6]]

        assert sorted(output_dir.listdir()) == sorted(img_dirs)
        for img_dir, det_ids in zip(img_dirs, img_dets):
            map_files = [img_dir.join(f"det_{det_id}.jpeg") for det_id in det_ids]
            assert sorted(img_dir.listdir()) == sorted(map_files)
