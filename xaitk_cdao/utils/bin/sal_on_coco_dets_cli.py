import click  # type: ignore
import json
from typing import TextIO

from maite import load_model

from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects
from smqtk_core.configuration import from_config_dict

from xaitk_saliency import GenerateObjectDetectorBlackboxSaliency

from xaitk_cdao.utils.sal_on_coco_dets import maite_sal_on_coco_dets, sal_on_coco_dets


@click.command(context_settings={"help_option_names": ['-h', '--help']})
@click.argument('coco_file', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=False))
@click.argument('config_file', type=click.File(mode='r'))
@click.option(
    '--overlay-image',
    is_flag=True,
    help='overlay saliency map on images with bounding boxes, RGB images are converted to grayscale'
)
@click.option('--verbose', '-v', count=True, help='print progress messages')
def sal_on_coco_dets_cli(
    coco_file: str,
    output_dir: str,
    config_file: TextIO,
    overlay_image: bool,
    verbose: bool
) -> None:
    """
    Generate saliency maps for detections in a COCO format file and write them
    to disk. Maps for each detection are written out in subdirectories named
    after their corresponding image file.

    \b
    COCO_FILE - COCO style annotation file with detections to compute saliency
        for.
    OUTPUT_DIR - Directory to write the saliency maps to.
    CONFIG_FILE - Configuration file for the object detector and
        GenerateObjectDetectorBlackboxSaliency implementations to use.

    \f
    :param coco_file: COCO style annotation file with detections to compute
        saliency for.
    :param output_dir: Directory to write the saliency maps to.
    :param config_file: Config file specifying the object detector and
        ``GenerateObjectDetectorBlackboxSaliency`` implementations to use.
    :param overlay_image: Overlay saliency maps on images with bounding boxes.
        RGB images are converted to grayscale. Default is to write out saliency
        maps by themselves.
    :param verbose: Display progress messages. Default is false.
    """

    # load config
    config = json.load(config_file)

    sal_generator = from_config_dict(
        config["GenerateObjectDetectorBlackboxSaliency"],
        GenerateObjectDetectorBlackboxSaliency.get_impls()
    )

    try:
        if "DetectImageObjects" in config:
            blackbox_detector = from_config_dict(config["DetectImageObjects"], DetectImageObjects.get_impls())

            sal_on_coco_dets(
                coco_file=coco_file,
                output_dir=output_dir,
                sal_generator=sal_generator,
                blackbox_detector=blackbox_detector,
                overlay_image=overlay_image,
                verbose=verbose
            )
        elif "ObjectDetector" in config:
            obj_detector = config["ObjectDetector"]

            # Config validation
            if "model_name" not in obj_detector:
                raise ValueError("model_name required for ObjectDetector")
            if "provider" not in obj_detector:
                raise ValueError("provider required for ObjectDetector")
            if "bbox_transform" not in obj_detector:
                raise ValueError("bbox_transform required for ObjectDetector")

            kwargs = {}
            if "kwargs" in obj_detector:
                kwargs = obj_detector["kwargs"]
            maite_detector = load_model(
                model_name=obj_detector["model_name"],
                provider=obj_detector["provider"],
                task="object-detection",
                **kwargs
            )

            maite_sal_on_coco_dets(
                coco_file=coco_file,
                output_dir=output_dir,
                sal_generator=sal_generator,
                maite_detector=maite_detector,
                bbox_transform=obj_detector["bbox_transform"],
                preprocessor=None,
                img_batch_size=obj_detector["img_batch_size"] if "img_batch_size" in obj_detector else 1,
                overlay_image=overlay_image,
                verbose=verbose
            )
        else:
            raise ValueError("Could not identify object detector in config file")
    except ImportError:
        print("This tool requires additional dependencies, please install 'xaitk-cdao[tools]'")
        exit(-1)
