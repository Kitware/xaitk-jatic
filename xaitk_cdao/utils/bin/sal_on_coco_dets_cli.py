import click  # type: ignore
import json
from typing import TextIO, Dict, Any, TYPE_CHECKING

from maite import load_model

from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects
from smqtk_core.configuration import from_config_dict, make_default_config

from xaitk_saliency import GenerateObjectDetectorBlackboxSaliency

from xaitk_cdao.interop.bbox_transformer import BBoxTransformer
from xaitk_cdao.interop.preprocessor import Preprocessor
from xaitk_cdao.utils.sal_on_coco_dets import maite_sal_on_coco_dets, sal_on_coco_dets


@click.command(context_settings={"help_option_names": ['-h', '--help']})
@click.argument('coco_file', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=False))
@click.argument('config_file', type=click.File(mode='r'))
@click.option('-g', '--generate-config-file', help='write default config to specified file', type=click.File(mode='w'))
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
    generate_config_file: TextIO,
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
    :param generate_config_file: File to write default config file, only written
        if specified. Only one of "DetectImageObjects" and "ObjectDetector" should
        be kept.
        This skips the normal operation of this tool and only outputs the file.
    :param verbose: Display progress messages. Default is false.
    """

    if generate_config_file:
        config: Dict[str, Any] = dict()

        config["DetectImageObjects"] = make_default_config(DetectImageObjects.get_impls())
        config["GenerateObjectDetectorBlackboxSaliency"] = make_default_config(
            GenerateObjectDetectorBlackboxSaliency.get_impls()
        )

        # MAITE detector config
        config["ObjectDetector"] = dict()
        config["ObjectDetector"]["model_name"] = "model_name"
        config["ObjectDetector"]["provider"] = "provider"
        config["ObjectDetector"]["model_name"] = "model_name"
        config["ObjectDetector"]["BBoxTransformer"] = make_default_config(BBoxTransformer.get_impls())
        config["ObjectDetector"]["Preprocessor"] = make_default_config(Preprocessor.get_impls())
        config["ObjectDetector"]["img_batch_size"] = 1

        json.dump(config, generate_config_file, indent=4)

        exit()

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
                raise ValueError("Missing required ObjectDetector configuration: model_name")
            if "provider" not in obj_detector:
                raise ValueError("Missing required ObjectDetector configuration: provider")
            if "BBoxTransformer" not in obj_detector:
                raise ValueError("Missing required ObjectDetector configuration: BBoxTransformer")

            kwargs = dict()
            if "kwargs" in obj_detector:
                if TYPE_CHECKING:
                    assert isinstance(obj_detector["kwargs"], Dict)
                kwargs = obj_detector["kwargs"]

            maite_detector = load_model(  # type: ignore
                model_name=obj_detector["model_name"],
                provider=obj_detector["provider"],
                task="object-detection",
                **kwargs
            )

            bbox_transform = from_config_dict(obj_detector["BBoxTransformer"], BBoxTransformer.get_impls())
            preprocessor = from_config_dict(obj_detector["Preprocessor"], Preprocessor.get_impls())

            maite_sal_on_coco_dets(
                coco_file=coco_file,
                output_dir=output_dir,
                sal_generator=sal_generator,
                detector=maite_detector,
                bbox_transform=bbox_transform,
                preprocessor=preprocessor,
                img_batch_size=obj_detector["img_batch_size"] if "img_batch_size" in obj_detector else 1,
                overlay_image=overlay_image,
                verbose=verbose
            )
        else:
            raise ValueError("Could not identify object detector in config file")
    except ImportError as e:
        print("This tool requires additional dependencies, please install 'xaitk-cdao[tools]'"
              " and confirm dependencies for selected model are installed")
        print(e)
        exit(-1)
