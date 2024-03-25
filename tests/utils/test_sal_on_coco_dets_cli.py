from click.testing import CliRunner
import json
import os
import py  # type: ignore
import pytest
import unittest.mock as mock
from unittest.mock import MagicMock
from maite.protocols import ObjectDetector
from xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise import DRISEStack

from tests import DATA_DIR

from xaitk_cdao.interop.bbox_transformer import XYXYBBoxTransformer
from xaitk_cdao.utils.bin.sal_on_coco_dets_cli import sal_on_coco_dets_cli

from importlib.util import find_spec


deps = ['kwcoco']
specs = [find_spec(dep) for dep in deps]
is_usable = all([spec is not None for spec in specs])

dets_file = os.path.join(DATA_DIR, 'test_dets.json')
config_file = os.path.join(DATA_DIR, 'config.json')
maite_config_file = os.path.join(DATA_DIR, 'maite_config.json')


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

        runner = CliRunner()

        result = runner.invoke(sal_on_coco_dets_cli, [str(dets_file), str(output_dir), str(config_file)])

        assert result.output.startswith("This tool requires additional dependencies, please install 'xaitk-cdao[tools]'"
                                        " and confirm dependencies for selected model are installed\n")
        assert not output_dir.check(dir=1)


@pytest.mark.skipif(not is_usable, reason="Extra 'xaitk-cdao[tools]' not installed.")
class TestSalOnCocoDets:
    """
    These tests make use of the `tmpdir` fixture from `pytest`. Find more
    information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    def test_coco_sal_gen(self, tmpdir: py.path.local) -> None:
        """
        Test saliency map generation with RandomDetector, RISEGrid, and
        DRISEScoring.
        """
        output_dir = tmpdir.join('out')

        runner = CliRunner()
        result = runner.invoke(sal_on_coco_dets_cli,
                               [str(dets_file), str(output_dir),
                                str(config_file), "-v"])

        # expected created directories for image saliency maps
        img_dirs = [output_dir.join(d) for d in ["test_image1", "test_image2"]]
        # detection ids that belong to each image
        img_dets = [[1, 2, 3], [5, 6]]

        assert result.exit_code == 0
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

        runner = CliRunner()
        result = runner.invoke(sal_on_coco_dets_cli,
                               [str(dets_file), str(output_dir), str(config_file),
                                "--overlay-image"])

        # expected created directories for image saliency maps
        img_dirs = [output_dir.join(d) for d in ["test_image1", "test_image2"]]
        # detection ids that belong to each image
        img_dets = [[1, 2, 3], [5, 6]]

        assert result.exit_code == 0
        assert sorted(output_dir.listdir()) == sorted(img_dirs)
        for img_dir, det_ids in zip(img_dirs, img_dets):
            map_files = [img_dir.join(f"det_{det_id}.jpeg") for det_id in det_ids]
            assert sorted(img_dir.listdir()) == sorted(map_files)

    def test_config_gen(self, tmpdir: py.path.local) -> None:
        """
        Test the generate configuration file option.
        """
        output_dir = tmpdir.join('out')

        output_config = tmpdir.join('gen_conf.json')

        runner = CliRunner()
        runner.invoke(sal_on_coco_dets_cli,
                      [str(dets_file), str(output_dir), str(config_file),
                       "-g", str(output_config)])

        # check that config file was created
        assert output_config.check(file=1)
        # check that no output was generated
        assert not output_dir.check(dir=1)

    def test_missing_detector(self, tmpdir: py.path.local) -> None:
        """
        Test missing detector configuration.
        """
        output_dir = tmpdir.join('out')

        bad_config_file = tmpdir.join('bad_config.json')

        # Remove
        with open(config_file) as f:
            bad_config = json.load(f)
        del bad_config["DetectImageObjects"]

        with open(bad_config_file, 'w') as f:
            json.dump(bad_config, f, indent=4)

        runner = CliRunner()
        with pytest.raises(ValueError, match=r"Could not identify object detector"):
            _ = runner.invoke(
                sal_on_coco_dets_cli,
                [
                    str(dets_file),
                    str(output_dir),
                    str(bad_config_file)
                ],
                catch_exceptions=False
            )

        # check that no output was generated
        assert not output_dir.check(dir=1)

    @pytest.mark.parametrize("config_key", ["model_name", "provider", "BBoxTransformer"])
    def test_bad_config(self, config_key: str, tmpdir: py.path.local) -> None:
        """
        Test bad configurations.
        """
        output_dir = tmpdir.join('out')

        bad_config_file = tmpdir.join('bad_config.json')

        # Remove
        with open(maite_config_file) as f:
            bad_config = json.load(f)
        del bad_config["ObjectDetector"][config_key]

        with open(bad_config_file, 'w') as f:
            json.dump(bad_config, f, indent=4)

        runner = CliRunner()
        with pytest.raises(ValueError, match=r"Missing required ObjectDetector configuration:"):
            _ = runner.invoke(
                sal_on_coco_dets_cli,
                [
                    str(dets_file),
                    str(output_dir),
                    str(bad_config_file)
                ],
                catch_exceptions=False
            )

        # check that no output was generated
        assert not output_dir.check(dir=1)

    @mock.patch('xaitk_cdao.utils.bin.sal_on_coco_dets_cli.maite_sal_on_coco_dets', return_value=None)
    @mock.patch('xaitk_cdao.utils.bin.sal_on_coco_dets_cli.load_model', return_value=MagicMock(spec=ObjectDetector))
    def test_maite_coco_sal_gen(
        self,
        load_model_patch: MagicMock,
        maite_sal_patch: MagicMock,
        tmpdir: py.path.local
    ) -> None:
        """
        Test workflow with MAITE Detector.
        """
        output_dir = tmpdir.join('out')

        runner = CliRunner()
        _ = runner.invoke(
            sal_on_coco_dets_cli,
            [
                str(dets_file),
                str(output_dir),
                str(maite_config_file),
                "-v"
            ]
        )

        # Confirm maite_sal_on_coco_dets arguments are as expected
        kwargs = maite_sal_patch.call_args.kwargs
        assert kwargs["coco_file"] == str(dets_file)
        assert kwargs["output_dir"] == str(output_dir)
        assert isinstance(kwargs["sal_generator"], DRISEStack)
        assert not kwargs["overlay_image"]
        assert kwargs["verbose"]
        assert isinstance(kwargs["detector"], ObjectDetector)
        assert isinstance(kwargs["bbox_transform"], XYXYBBoxTransformer)
        assert kwargs["preprocessor"] is None
        assert kwargs["img_batch_size"] == 2

        # Confirm load_model arguments are as expected
        kwargs = load_model_patch.call_args.kwargs
        assert kwargs["model_name"] == "model_name"
        assert kwargs["provider"] == "provider"
        assert kwargs["task"] == "object-detection"
        assert kwargs["use_cuda"]
