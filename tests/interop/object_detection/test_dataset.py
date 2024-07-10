import numpy as np
import pytest
from pathlib import Path
from PIL import Image  # type: ignore

from xaitk_jatic.interop.object_detection.dataset import JATICObjectDetectionDataset, JATICDetectionTarget

from tests import DATA_DIR
try:
    import kwcoco  # type: ignore
    from xaitk_jatic.interop.object_detection.dataset import COCOJATICObjectDetectionDataset
    is_usable = True
except ImportError:
    is_usable = False

dset_dir = Path(DATA_DIR)


@pytest.mark.skipif(not is_usable, reason="Extra 'xaitk-jatic[tools]' not installed.")
class TestCOCOJATICObjectDetectionDataset:
    if is_usable:
        coco_file = dset_dir / "annotations.json"
        kwcoco_dataset = kwcoco.CocoDataset(coco_file)
        metadata = {1: {"test": 1}, 2: {"test": 2}, 3: {"test": 3}}

        test_img_files = ['test_image1.png', 'test_image2.png']
        test_imgs = [np.array(Image.open(dset_dir / f)) for f in test_img_files]

    test_bboxes = [
        np.array([
            [10., 20., 40., 60.],
            [12., 45., 62., 75.],
            [30., 5., 77., 97.]
        ]),
        np.array([
            [0., 0., 5., 5.],
            [50., 50., 82., 67.],
            [68., 82., 79., 89.]
        ])
    ]

    test_scores = [
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 0.5])
    ]

    def test_len(self) -> None:
        """
        Test length property
        """
        dataset = COCOJATICObjectDetectionDataset(
            kwcoco_dataset=TestCOCOJATICObjectDetectionDataset.kwcoco_dataset,
            image_metadata=TestCOCOJATICObjectDetectionDataset.metadata
        )
        assert len(dataset) == len(TestCOCOJATICObjectDetectionDataset.test_imgs)

    def test_get_item(self) -> None:
        """
        Test indexing
        """
        dataset = COCOJATICObjectDetectionDataset(
            kwcoco_dataset=TestCOCOJATICObjectDetectionDataset.kwcoco_dataset,
            image_metadata=TestCOCOJATICObjectDetectionDataset.metadata
        )

        for idx in range(len(dataset)):
            img, dets, md = dataset[idx]
            assert np.array_equal(img, np.asarray(TestCOCOJATICObjectDetectionDataset.test_imgs[idx]))
            assert np.array_equal(dets.boxes, TestCOCOJATICObjectDetectionDataset.test_bboxes[idx])
            assert np.array_equal(dets.scores, TestCOCOJATICObjectDetectionDataset.test_scores[idx])
            assert md["test"] == TestCOCOJATICObjectDetectionDataset.metadata[md["id"]]["test"]

    def test_bad_metadata(self) -> None:
        """
        Test that an exception is appropriately raised if metadata is missing
        """
        with pytest.raises(ValueError, match=r"Image metadata length mismatch"):
            _ = COCOJATICObjectDetectionDataset(
                kwcoco_dataset=TestCOCOJATICObjectDetectionDataset.kwcoco_dataset,
                image_metadata=dict()
            )


class TestJATICObjectDetectionDataset:
    imgs = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(2)]
    dets = [
                JATICDetectionTarget(
                    boxes=np.asarray([[0., 0., 50., 50.]]),
                    labels=np.asarray([0]),
                    scores=np.asarray([0.25])
                ),
                JATICDetectionTarget(
                    boxes=np.asarray([[0., 0., 3., 4.], [5.6, 10.7, 52.3, 76.0]]),
                    labels=np.asarray([1, 0]),
                    scores=np.asarray([0.34, 0.95])
                )
            ]
    metadata = [{"md": 0}, {"md": 1}]

    def test_len(self) -> None:
        """
        Test length property
        """
        dataset = JATICObjectDetectionDataset(
            imgs=TestJATICObjectDetectionDataset.imgs,
            dets=TestJATICObjectDetectionDataset.dets,
            metadata=TestJATICObjectDetectionDataset.metadata
        )

        assert len(dataset) == len(TestJATICObjectDetectionDataset.imgs)

    def test_get_item(self) -> None:
        """
        Test indexing
        """
        dataset = JATICObjectDetectionDataset(
            imgs=TestJATICObjectDetectionDataset.imgs,
            dets=TestJATICObjectDetectionDataset.dets,
            metadata=TestJATICObjectDetectionDataset.metadata
        )

        for idx in range(len(dataset)):
            img, dets, md = dataset[idx]
            assert np.array_equal(img, TestJATICObjectDetectionDataset.imgs[idx])
            assert dets == TestJATICObjectDetectionDataset.dets[idx]
            assert md == TestJATICObjectDetectionDataset.metadata[idx]
