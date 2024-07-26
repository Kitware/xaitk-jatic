import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Hashable, TextIO

import click  # type: ignore
import maite.protocols.object_detection as od
import numpy as np
import torch  # type: ignore
from PIL import Image  # type: ignore
from smqtk_core.configuration import from_config_dict
from xaitk_saliency import GenerateObjectDetectorBlackboxSaliency

from xaitk_jatic.interop.object_detection.model import JATICDetector

try:
    import matplotlib.pyplot as plt  # type: ignore
    from torchvision.transforms.functional import get_image_size  # type: ignore
    from transformers import (
        AutoImageProcessor,  # type: ignore
        AutoModelForObjectDetection,  # type: ignore
    )

    is_usable = True
except ImportError:
    is_usable = False


@dataclass
class DetectionTarget:
    boxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor


class HuggingFaceDetector:
    def __init__(self, model_name: str, threshold: float, device: str):
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name)
        self.threshold = threshold
        self.device = device

        self.model.eval()
        self.model.to(device)

    def id2label(self) -> Dict[int, Hashable]:
        return self.model.config.id2label

    def __call__(self, batch: od.InputBatchType) -> od.TargetBatchType:
        # tensor bridging
        batch = torch.as_tensor(batch)
        assert batch.ndim == 4

        # save original image sizes for resizing boxes
        target_sizes = [get_image_size(img)[::-1] for img in batch]

        # preprocess
        hf_inputs = self.image_processor(batch, return_tensors="pt")

        # put on device
        hf_inputs = hf_inputs.to(self.device)

        # get predictions
        with torch.no_grad():
            hf_predictions = self.model(**hf_inputs)
        hf_results = self.image_processor.post_process_object_detection(
            hf_predictions, threshold=self.threshold, target_sizes=target_sizes
        )

        predictions: od.TargetBatchType = list()
        for result in hf_results:
            predictions.append(
                DetectionTarget(
                    result["boxes"].detach().cpu(),
                    result["labels"].detach().cpu(),
                    result["scores"].detach().cpu(),
                )
            )

        return predictions


def dets_to_mats(dets, jatic_detector):
    labels = [jatic_detector.id2label()[id_] for id_ in sorted(jatic_detector.id2label().keys())]  # type: ignore

    bboxes = np.empty((0, 4))
    scores = np.empty((0, len(labels)))
    for det in dets:
        bbox = det[0]

        bboxes = np.vstack(
            (
                bboxes,
                [
                    *bbox.min_vertex,
                    *bbox.max_vertex,
                ],
            )
        )

        score_dict = det[1]
        score_array = []
        for label in labels:
            score_array.append(score_dict[label])

        scores = np.vstack(
            (
                scores,
                score_array,
            )
        )

    return bboxes, scores


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("image_file", type=str)
@click.argument("output_dir", type=click.Path(exists=False))
@click.argument("config_file", type=click.File(mode="r"))
@click.argument("hugging_face_model_name", type=str)
@click.option("--verbose", "-v", count=True, help="print progress messages")
def generate_saliency_maps(
    image_file: str,
    output_dir: str,
    config_file: TextIO,
    hugging_face_model_name: str,
    verbose: bool,
) -> None:
    r"""Generate saliency maps for detections and write them to disk.

    \f
    :param image_file: Input image.
    :param output_dir: Directory to write the saliency maps to.
    :param config_file: Config file specifying the object detector and
        ``GenerateObjectDetectorBlackboxSaliency`` implementations to use.
    :param hugging_face_model_name: Name of model to load from HuggingFace.
    :param verbose: Display progress messages. Default is false.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    if not is_usable:
        print(
            "This tool requires additional dependencies, please install `xaitk-jatic[docker]'"
        )
        exit(-1)

    # Load config
    config = json.load(config_file)

    # Instantiate objects from config
    sal_generator = from_config_dict(
        config["GenerateObjectDetectorBlackboxSaliency"],
        GenerateObjectDetectorBlackboxSaliency.get_impls(),
    )

    fill = [95, 96, 93]
    sal_generator.fill = fill

    logging.info("Building detection model...")
    jatic_detector = HuggingFaceDetector(
        model_name=hugging_face_model_name, threshold=0.5, device="cpu"
    )

    blackbox_detector = JATICDetector(
        detector=jatic_detector, id_to_name=jatic_detector.id2label(), img_batch_size=5
    )

    img_path = Path(image_file)
    ref_img = np.asarray(Image.open(img_path))

    logging.info("Finding detections...")
    dets = list(blackbox_detector([ref_img]))[0]

    if len(dets) == 0:
        print("No detections found. Exiting...")
        return

    detector_bboxes, detector_scores = dets_to_mats(dets, jatic_detector)

    logging.info("Generating saliency maps...")
    img_sal_maps = sal_generator(
        ref_img, detector_bboxes, detector_scores, blackbox_detector
    )

    logging.info("Saving saliency maps...")
    for img_idx, sal_map in enumerate(img_sal_maps):
        fig = plt.figure()
        plt.axis("off")
        plt.imshow(sal_map, cmap="jet")
        plt.colorbar()
        plt.savefig(
            os.path.join(output_dir, f"det_{img_idx}.jpeg"), bbox_inches="tight"
        )
        plt.close(fig)

    logging.info("Saliency maps saved. Exiting...")


if __name__ == "__main__":
    generate_saliency_maps()
