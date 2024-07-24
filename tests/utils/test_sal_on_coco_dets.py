import logging
import numpy as np
import os
import py  # type: ignore
import pytest
import unittest.mock as mock
from click.testing import CliRunner
from unittest.mock import MagicMock
from smqtk_detection.impls.detect_image_objects.random_detector import RandomDetector
from xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise import DRISEStack

from tests import DATA_DIR

from xaitk_jatic.utils.bin.sal_on_coco_dets import sal_on_coco_dets

from importlib.util import find_spec


deps = ['kwcoco', 'matplotlib']
specs = [find_spec(dep) for dep in deps]
is_usable = all([spec is not None for spec in specs])

dset_dir = DATA_DIR
config_file = os.path.join(DATA_DIR, 'config.json')


class TestSalOnCocoDetsNotUsable:
    """
    These tests make use of the `tmpdir` fixture from `pytest`. Find more
    information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    @mock.patch("xaitk_jatic.utils.bin.sal_on_coco_dets.is_usable", False)
    def test_warning(self, tmpdir: py.path.local) -> None:
        """
        Test that proper warning is displayed when required dependencies are
        not installed.
        """
        output_dir = tmpdir.join('out')

        runner = CliRunner()

        result = runner.invoke(sal_on_coco_dets, [str(dset_dir), str(output_dir), str(config_file)])

        assert result.output.startswith(
            "This tool requires additional dependencies, please install 'xaitk-jatic[tools]'"
        )
        assert not output_dir.check(dir=1)


@pytest.mark.skipif(not is_usable, reason="Extra 'xaitk-jatic[tools]' not installed.")
class TestSalOnCocoDets:
    """
    These tests make use of the `tmpdir` fixture from `pytest`. Find more
    information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """
    mock_return_value = (
        [
            np.random.randint(0, 255, (3, 256, 256), dtype=np.uint8),
            np.random.randint(0, 255, (3, 256, 256), dtype=np.uint8)
        ],
        {
            "type": "xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise.DRISEStack",
            "xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise.DRISEStack": {
                "n": 10,
                "s": 8,
                "p1": 0.5,
                "seed": 0,
                "threads": 4
            }
        }
    )

    def test_config_gen(self, tmpdir: py.path.local) -> None:
        """
        Test the generate configuration file option.
        """
        output_dir = tmpdir.join('out')

        output_config = tmpdir.join('gen_conf.json')

        runner = CliRunner()
        runner.invoke(sal_on_coco_dets,
                      [str(dset_dir), str(output_dir), str(config_file),
                       "-g", str(output_config)])

        # check that config file was created
        assert output_config.check(file=1)
        # check that no output was generated
        assert not output_dir.check(dir=1)

    @pytest.mark.parametrize("overlay_image", [False, True])
    @mock.patch(
        'xaitk_jatic.utils.bin.sal_on_coco_dets.compute_sal_maps',
        return_value=mock_return_value
    )
    def test_compute_sal_maps(
        self,
        compute_sal_maps_patch: MagicMock,
        overlay_image: bool,
        tmpdir: py.path.local
    ) -> None:
        """
        Test that compute_sal_maps is called appropriately and the images are saved correctly.
        """
        output_dir = tmpdir.join('out')

        runner = CliRunner()
        runner_args = [
            str(dset_dir),
            str(output_dir),
            str(config_file),
            "-v"
        ]
        if overlay_image:
            runner_args.append("--overlay-image")
        result = runner.invoke(
            sal_on_coco_dets,
            runner_args,
            catch_exceptions=False
        )

        # Confirm compute_sal_maps arguments are as expected
        kwargs = compute_sal_maps_patch.call_args.kwargs
        assert len(kwargs["dataset"]) == 2
        assert isinstance(kwargs["sal_generator"], DRISEStack)
        assert isinstance(kwargs["blackbox_detector"], RandomDetector)
        assert kwargs["num_classes"] == 3

        # expected created directories for image saliency maps
        img_dirs = [output_dir.join(d) for d in ["test_image1", "test_image2"]]
        # detection ids that belong to each image
        img_dets = [[1, 2, 3], [4, 5, 6]]

        assert result.exit_code == 0
        assert sorted(output_dir.listdir()) == sorted(img_dirs)
        for img_dir, det_ids in zip(img_dirs, img_dets):
            map_files = [img_dir.join(f"det_{det_id}.jpeg") for det_id in det_ids]
            assert sorted(img_dir.listdir()) == sorted(map_files)

    @mock.patch('pathlib.Path.is_file', return_value=False)
    def test_missing_annotations(self, is_file_patch: MagicMock, tmpdir: py.path.local) -> None:
        """
        Check that an exception is appropriately raised if the annotations file is missing.
        """
        output_dir = tmpdir.join('out')

        with pytest.raises(ValueError, match=r"Could not identify annotations file."):
            runner = CliRunner()
            _ = runner.invoke(
                sal_on_coco_dets,
                [
                    str(dset_dir),
                    str(output_dir),
                    str(config_file),
                    "-v"
                ],
                catch_exceptions=False
            )

    @mock.patch('pathlib.Path.is_file', side_effect=[True, False])
    @mock.patch('xaitk_jatic.utils.bin.sal_on_coco_dets.compute_sal_maps', return_value=mock_return_value)
    def test_missing_metadata(
        self,
        _: MagicMock,
        is_file_patch: MagicMock,
        caplog: pytest.LogCaptureFixture,
        tmpdir: py.path.local
    ) -> None:
        """
        Check that the entrypoint is able to continue when a metadata file is not present (as
        long as it's not required by the perturber).
        """
        output_dir = tmpdir.join('out')

        with caplog.at_level(logging.INFO):
            runner = CliRunner()
            result = runner.invoke(
                sal_on_coco_dets,
                [
                    str(dset_dir),
                    str(output_dir),
                    str(config_file),
                    "-v"
                ],
                catch_exceptions=False
            )

            assert result.exit_code == 0

        assert "Could not identify metadata file, assuming no metadata." in caplog.text
