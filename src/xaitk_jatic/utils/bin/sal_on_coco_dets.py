import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, TextIO

import click  # type: ignore
import numpy as np
from PIL import Image  # type: ignore
from smqtk_core.configuration import from_config_dict, make_default_config
from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects
from xaitk_saliency import GenerateObjectDetectorBlackboxSaliency

from xaitk_jatic.utils.sal_on_dets import compute_sal_maps

try:
    import kwcoco  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.patches import Rectangle  # type: ignore

    from xaitk_jatic.interop.object_detection.dataset import (
        COCOJATICObjectDetectionDataset,
    )

    is_usable = True
except ImportError:
    is_usable = False


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("dataset_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(exists=False))
@click.argument("config_file", type=click.File(mode="r"))
@click.option(
    "-g",
    "--generate-config-file",
    help="write default config to specified file",
    type=click.File(mode="w"),
)
@click.option(
    "--overlay-image",
    is_flag=True,
    help="overlay saliency map on images with bounding boxes, RGB images are converted to grayscale",
)
@click.option("--verbose", "-v", count=True, help="print progress messages")
def sal_on_coco_dets(
    dataset_dir: str,
    output_dir: str,
    config_file: TextIO,
    overlay_image: bool,
    generate_config_file: TextIO,
    verbose: bool,
) -> None:
    """Generate saliency maps for detections in a COCO format file and write them to disk.

    Maps for each detection are written out in subdirectories named after their corresponding image file.

    DATASET_DIR - Root directory of dataset.
    OUTPUT_DIR - Directory to write the saliency maps to.
    CONFIG_FILE - Configuration file for the object detector and
    GenerateObjectDetectorBlackboxSaliency implementations to use.

    :param dataset_dir: Root directory of dataset.
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

        # TODO: Add MAITE config option(s) once it's possible to hydrate a MAITE model

        json.dump(config, generate_config_file, indent=4)

        exit()

    if verbose:
        logging.basicConfig(level=logging.INFO)

    if not is_usable:
        print("This tool requires additional dependencies, please install 'xaitk-jatic[tools]'.")
        exit(-1)

    # Load COCO dataset
    coco_file = Path(dataset_dir) / "annotations.json"
    if not coco_file.is_file():
        raise ValueError("Could not identify annotations file. Expected at '[dataset_dir]/annotations.json'.")
    logging.info(f"Loading kwcoco annotations from {coco_file}.")
    kwcoco_dataset = kwcoco.CocoDataset(coco_file)

    # Load metadata, if it exists
    metadata_file = Path(dataset_dir) / "image_metadata.json"
    if not metadata_file.is_file():
        logging.info(
            "Could not identify metadata file, assuming no metadata. " "Expected at '[dataset_dir]/image_metadata.json'"
        )
        metadata: Dict[int, Dict[str, Any]] = {img_id: dict() for img_id in kwcoco_dataset.imgs.keys()}
    else:
        logging.info(f"Loading metadata from {metadata_file}")
        with open(metadata_file) as f:
            metadata = json.load(f)
        metadata = {int(img_id): md for img_id, md in metadata.items()}

    # Initialize dataset object
    input_dataset = COCOJATICObjectDetectionDataset(
        kwcoco_dataset=kwcoco_dataset, image_metadata=metadata, skip_no_anns=True
    )

    cids = [cat["id"] for cat in kwcoco_dataset.cats.values()]
    min_cid = min(cids)
    num_classes = max(cids) - min_cid + 1

    # Load config
    config = json.load(config_file)

    # Instantiate objects from config
    sal_generator = from_config_dict(
        config["GenerateObjectDetectorBlackboxSaliency"],
        GenerateObjectDetectorBlackboxSaliency.get_impls(),
    )
    blackbox_detector = from_config_dict(config["DetectImageObjects"], DetectImageObjects.get_impls())
    # TODO: Add MAITE object handling once config options are added

    img_sal_maps, _ = compute_sal_maps(
        dataset=input_dataset,
        sal_generator=sal_generator,
        blackbox_detector=blackbox_detector,
        num_classes=num_classes,
    )

    # Save saliency maps
    for dset_idx in range(len(input_dataset)):
        ref_img, dets, md = input_dataset[dset_idx]
        img_file = md["image_info"]["file_name"]

        if overlay_image:
            # Convert to channel last
            ref_img = np.transpose(np.asarray(ref_img), axes=(1, 2, 0))
            ref_img = np.squeeze(ref_img)

        # split file from parent folder
        img_name = os.path.split(img_file)[1]
        # split off file extension
        img_name = os.path.splitext(img_name)[0]

        sub_dir = os.path.join(output_dir, img_name)

        os.makedirs(sub_dir, exist_ok=True)

        det_ids = md["det_ids"]
        bboxes = np.asarray(dets.boxes)
        for sal_idx, bbox in enumerate(bboxes):
            sal_map = img_sal_maps[dset_idx][sal_idx]
            det_id = det_ids[sal_idx]

            fig = plt.figure()
            plt.axis("off")
            if overlay_image:
                gray_img = np.asarray(Image.fromarray(np.asarray(ref_img)).convert("L"))
                plt.imshow(gray_img, alpha=0.7, cmap="gray")

                plt.gca().add_patch(
                    Rectangle(
                        (bbox[0], bbox[1]),
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1],
                        linewidth=1,
                        edgecolor="r",
                        facecolor="none",
                    )
                )
                plt.imshow(sal_map, cmap="jet", alpha=0.3)
                plt.colorbar()
            else:
                plt.imshow(sal_map, cmap="jet")
                plt.colorbar()
            plt.savefig(os.path.join(sub_dir, f"det_{det_id}.jpeg"), bbox_inches="tight")
            plt.close(fig)
