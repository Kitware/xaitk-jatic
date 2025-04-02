"""
HuggingFace Object Detection and Saliency Map Generation Tool

This module implements an object detection workflow using HuggingFace pre-trained models
and generates corresponding saliency maps. It provides a command-line interface (CLI) to
process an input image, run object detection, generate saliency maps, and save them as
images for visualization.

Features:
    - Loads HuggingFace pre-trained object detection models.
    - Uses a configured saliency generator to compute saliency maps for detected objects.
    - Saves saliency maps as images for further inspection.
    - CLI tool for easy execution.

Classes:
    - DetectionTarget: Data container for detection results (boxes, labels, scores).
    - HuggingFaceDetector: Wrapper for HuggingFace object detection models.

Functions:
    - dets_to_mats: Converts detections into matrices of bounding boxes and scores.
    - generate_saliency_maps: CLI function to process input image and save saliency maps.

Dependencies:
    - torch
    - transformers
    - xaitk_saliency
    - matplotlib
    - numpy
    - smqtk_core
    - PIL (Pillow)
    - click

Usage:
    Run the CLI tool as follows:
        python <script_name>.py <image_file> <output_dir> <config_file> <hugging_face_model_name> [--verbose]

Example:
    >>> python generate_saliency_maps.py test.jpg output_dir config.json facebook/detr-resnet-50
"""

import json
import logging
import os
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

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
    """
    A data container for storing detection results.

    Attributes:
        boxes (torch.Tensor): Tensor containing bounding box coordinates.
        labels (torch.Tensor): Tensor containing class labels for detections.
        scores (torch.Tensor): Tensor containing confidence scores for detections.
    """

    boxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor


class HuggingFaceDetector:
    """
    Wrapper around HuggingFace's pre-trained object detection models.

    This class provides an interface to run inference on input batches of images
    and obtain detection results.

    Attributes:
        model_name (str): Name of the HuggingFace model to load.
        threshold (float): Confidence threshold for filtering predictions.
        device (str): Device to run the model on ("cpu" or "cuda").

    Methods:
        id2label(): Returns a dictionary mapping class indices to class labels.
        __call__(batch): Runs object detection on a batch of images and returns results.

    Args:
        model_name (str): Name of the HuggingFace model.
        threshold (float): Confidence threshold for predictions.
        device (str): Device for computation.
    """

    def __init__(self, model_name: str, threshold: float, device: str) -> None:
        """
        Initialize the HuggingFaceDetector with a pre-trained object detection model.

        This method sets up the image processor and model for object detection using
        the HuggingFace Transformers library. The model is loaded in evaluation mode
        and moved to the specified device.

        Args:
            model_name (str):
                Name or path of the pre-trained HuggingFace model to load (e.g., 'facebook/detr-resnet-50').
            threshold (float):
                Confidence threshold for filtering predictions.
                Detections with confidence below this value are discarded.
            device (str):
                Device to run the model on. Use "cpu" for CPU inference or "cuda" for GPU acceleration.

        Attributes:
            image_processor (AutoImageProcessor):
                Pre-processing pipeline for images, as provided by the HuggingFace model.
            model (AutoModelForObjectDetection):
                The pre-trained HuggingFace object detection model.
            threshold (float):
                The confidence threshold for detection filtering.
            device (str):
                The device where the model is loaded (e.g., "cpu" or "cuda").

        Example:
            >>> detector = HuggingFaceDetector("facebook/detr-resnet-50", threshold=0.5, device="cuda")
            >>> print(detector.threshold)
            0.5
        """
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name)
        self.threshold = threshold
        self.device = device

        self.model.eval()
        self.model.to(device)

    def id2label(self) -> dict[int, Hashable]:
        """Returns a dictionary mapping class indices to labels."""
        return self.model.config.id2label

    def __call__(self, batch: od.InputBatchType) -> od.TargetBatchType:
        """
        Run object detection on a batch of input images.

        Args:
            batch (od.InputBatchType): Batch of input images as tensors.

        Returns:
            od.TargetBatchType: List of DetectionTarget results containing boxes, labels, and scores.
        """
        # tensor bridging
        batch = torch.as_tensor(batch)
        if batch.ndim != 4:
            raise ValueError("Batch must have 4 dimensions")

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
            hf_predictions,
            threshold=self.threshold,
            target_sizes=target_sizes,
        )

        predictions: od.TargetBatchType = list()
        for result in hf_results:
            predictions.append(
                DetectionTarget(
                    result["boxes"].detach().cpu(),
                    result["labels"].detach().cpu(),
                    result["scores"].detach().cpu(),
                ),
            )

        return predictions


def dets_to_mats(dets: list, jatic_detector: HuggingFaceDetector) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts a list of detections into bounding box and score matrices.

    Args:
        dets (list): List of detections, where each detection contains a bounding box and scores.
        jatic_detector (HuggingFaceDetector): Detector instance to retrieve class labels.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Bounding boxes as a NumPy array of shape (N, 4).
            - Scores as a NumPy array of shape (N, num_classes).
    """
    ids = sorted(jatic_detector.id2label().keys())  # type: ignore

    bboxes = np.empty((0, 4))
    scores = np.empty((0, len(ids)))

    for det in dets:
        bbox = det[0]

        bboxes = np.vstack(
            (
                bboxes,
                [
                    *bbox.min_vertex,
                    *bbox.max_vertex,
                ],
            ),
        )

        score_dict = det[1]
        score_array = [score_dict[idx] for idx in ids]

        scores = np.vstack(
            (
                scores,
                score_array,
            ),
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
        print("This tool requires additional dependencies, please install `xaitk-jatic[docker]'")
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
    jatic_detector = HuggingFaceDetector(model_name=hugging_face_model_name, threshold=0.5, device="cpu")

    blackbox_detector = JATICDetector(
        detector=jatic_detector,
        ids=list(jatic_detector.id2label().keys()),
        img_batch_size=5,
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
    img_sal_maps = sal_generator(ref_img, detector_bboxes, detector_scores, blackbox_detector)

    logging.info("Saving saliency maps...")
    for img_idx, sal_map in enumerate(img_sal_maps):
        fig = plt.figure()
        plt.axis("off")
        plt.imshow(sal_map, cmap="jet")
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, f"det_{img_idx}.jpeg"), bbox_inches="tight")
        plt.close(fig)

    logging.info("Saliency maps saved. Exiting...")


if __name__ == "__main__":
    generate_saliency_maps()
