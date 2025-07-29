"""
Object Detection Using Torchvision Faster R-CNN with ResNet Backbone

This module implements an object detector using Faster R-CNN with a ResNet-50-FPN backbone
from the `torchvision` library. It adheres to the `DetectImageObjects` interface
for standardized object detection tasks. The model is pretrained on the COCO train2017 dataset.

Classes:
    - ResNetFRCNN: Implements the object detector for image input.

Functions:
    - _postprocess_detections: Modified bounding box post-processing to include class probabilities.

Constants:
    - COCO_INSTANCE_CATEGORY_NAMES: Class labels for COCO dataset detections.
    - COCO_INSTANCE_CATEGORY_NAMES_NA: Placeholder for ignored classes.

Dependencies:
    - torch
    - torchvision
    - numpy
    - smqtk-image-io
    - smqtk-detection

Usage Example:
    >>> from module_name import ResNetFRCNN
    >>> detector = ResNetFRCNN(box_thresh=0.1, num_dets=50, use_cuda=True)
    >>> image = np.random.rand(224, 224, 3)  # Simulated image input
    >>> detections = detector.detect_objects([image])
    >>> for img_dets in detections:
    >>>     for bbox, scores in img_dets:
    >>>         print(bbox, scores)

Key Features:
    - Lazy model loading for efficient resource utilization.
    - Configurable parameters such as box confidence thresholds, maximum detections, and batch size.
    - Optional GPU (CUDA) support for accelerated inference.
    - Enhanced bounding box post-processing to include class probabilities.

Module Components:
    1. **ResNetFRCNN Class**:
       - Implements Faster R-CNN detection with configurable options.
       - Supports detection on batches of images using a lazy-loaded model.

    2. **_postprocess_detections Function**:
       - Customizes the default Faster R-CNN postprocessing to include class probabilities
         instead of just the confidence score.

    3. **Constants**:
       - `COCO_INSTANCE_CATEGORY_NAMES`: Predefined class labels for COCO objects.
       - `COCO_INSTANCE_CATEGORY_NAMES_NA`: Placeholder for ignored labels.

Error Handling:
    - Raises `SystemExit` when required dependencies (`torch`, `torchvision`) are not installed.
    - Raises `RuntimeError` if CUDA is requested but not available.

Logging:
    - Logs the number of batches processed during inference for visibility.

Notes:
    - The module uses lazy loading of the detection model to optimize performance.
    - The provided model checkpoint (`carla_rgb_weights_eval5.pt`) must be available locally.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Hashable, Iterable
from types import MethodType
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

try:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore  # noqa: N812
    import torchvision.models as models  # type: ignore
    from torchvision import transforms  # type: ignore
    from torchvision.models.detection.roi_heads import RoIHeads  # type: ignore
    from torchvision.ops import boxes as box_ops  # type: ignore
except ModuleNotFoundError as err:
    raise SystemExit("One or more module(s) not found. Exiting...") from err

from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects

LOG = logging.getLogger(__name__)


class ResNetFRCNN(DetectImageObjects):
    """DetectImageObjects implementation using torchvision's Faster R-CNN with a ResNet-50-FPN backbone.

    Pretrained on COCO train2017.

    :param box_thresh: Confidence threshold for detections.
    :param num_dets: Maximum number of detections per image.
    :param img_batch_size: Batch size in images for inferences.
    :param use_cuda: Attempt to use a cuda device for inferences. If no
        device is found, CPU is used.
    :param cuda_device: When using CUDA use the device by the given ID. By
        default, this refers to GPU ID 0. This parameter is not used if
        `use_cuda` is false.
    """

    def __init__(
        self,
        box_thresh: float = 0.05,
        num_dets: int = 100,
        img_batch_size: int = 1,
        use_cuda: bool = False,
        cuda_device: int | str = "cuda:0",
    ) -> None:
        """
        Initialize the object detector with configurable parameters.

        Args:
            box_thresh (float, optional):
                Threshold for filtering detection boxes based on confidence scores.
                Defaults to 0.05.
            num_dets (int, optional):
                Maximum number of detections to retain per image.
                Defaults to 100.
            img_batch_size (int, optional):
                Number of images to process in a single batch.
                Defaults to 1.
            use_cuda (bool, optional):
                Whether to enable GPU acceleration using CUDA.
                Defaults to False.
            cuda_device (int | str, optional):
                CUDA device to use for computation (e.g., "cuda:0" or device index).
                Defaults to "cuda:0".

        Attributes:
            box_thresh (float): Confidence threshold for detection boxes.
            num_dets (int): Maximum number of detections per image.
            img_batch_size (int): Batch size for image processing.
            use_cuda (bool): Indicates if CUDA (GPU) is used.
            cuda_device (int | str): Specifies the CUDA device.
            checkpoint (str): Path to the pre-trained model weights.
            model (Optional[torch.nn.Module]): The detection model (lazy loaded).
            model_device (Optional[torch.device]): The device where the model resides.
            model_loader (transforms.Compose): Image transformation pipeline for model input.

        Example:
            >>> detector = MyObjectDetector(box_thresh=0.1, num_dets=50, use_cuda=True)
            >>> print(detector.box_thresh)
            0.1
        """
        self.box_thresh = box_thresh
        self.num_dets = num_dets
        self.img_batch_size = img_batch_size
        self.use_cuda = use_cuda
        self.cuda_device = cuda_device
        self.checkpoint = "./data/saved_models/carla_rgb_weights_eval5.pt"

        # Set to None for lazy loading later.
        self.model: torch.nn.Module = None  # type: ignore
        self.model_device: torch.device = None  # type: ignore

        # The model already has normalization and resizing baked into the
        # layers.
        self.model_loader = transforms.Compose(
            [
                transforms.ToTensor(),
            ],
        )

    def get_model(self) -> torch.nn.Module:
        """Lazy load the torch model in an idempotent manner.

        :raises RuntimeError: Use of CUDA was requested but is not available.
        """
        model = self.model
        if model is None:
            model = models.detection.fasterrcnn_resnet50_fpn(
                # pretrained=True,
                num_classes=4,
                progress=False,
                box_detections_per_img=self.num_dets,
                box_score_thresh=self.box_thresh,
            )

            model_device = torch.device("cpu")
            if self.use_cuda:
                if torch.cuda.is_available():
                    model_device = torch.device(device=self.cuda_device)
                    model = model.to(device=model_device)
                else:
                    raise RuntimeError("Use of CUDA requested, but not available.")

            ckpt = torch.load(self.checkpoint, map_location=model_device)
            model.load_state_dict(ckpt)
            model = model.eval()

            model.roi_heads.postprocess_detections = (  # type: ignore
                MethodType(_postprocess_detections, model.roi_heads)
            )
            # store the loaded model for later return.
            self.model = model
            self.model_device = model_device
        return model

    @override
    def detect_objects(  # noqa:C901
        self,
        img_iter: Iterable[np.ndarray],
    ) -> Iterable[Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]:
        model = self.get_model()

        # batch model passes
        all_img_dets = []  # type: list[dict]
        batch = []
        batch_idx = 0
        for img in img_iter:
            batch.append(img)

            if len(batch) is self.img_batch_size:
                batch_tensors = [self.model_loader(batch_img).to(device=self.model_device) for batch_img in batch]

                with torch.no_grad():
                    img_dets = model(batch_tensors)

                for det in img_dets:
                    det["boxes"] = det["boxes"].cpu().numpy()
                    det["scores"] = det["scores"].cpu().numpy()
                    all_img_dets.append(det)

                batch = []

                batch_idx += 1
                LOG.info(f"{batch_idx} batches computed")

        # compute leftover batch
        if len(batch) > 0:
            batch_tensors = [self.model_loader(batch_img).to(device=self.model_device) for batch_img in batch]

            with torch.no_grad():
                img_dets = model(batch_tensors)

            for det in img_dets:
                det["boxes"] = det["boxes"].cpu().numpy()
                det["scores"] = det["scores"].cpu().numpy()
                all_img_dets.append(det)

            batch_idx += 1
            LOG.info(f"{batch_idx} batches computed")

        formatted_dets = []  # AxisAlignedBoundingBox detections to return
        for img_dets in all_img_dets:
            bboxes = img_dets["boxes"]
            scores = img_dets["scores"]

            a_bboxes = [AxisAlignedBoundingBox([box[0], box[1]], [box[2], box[3]]) for box in bboxes]

            score_dicts = []

            for img_scores in scores:
                score_dict = {}  # type: dict[Hashable, float]
                # Scores returned start at COCO i.d. 1
                for i, n in enumerate(img_scores, start=1):
                    score_dict[COCO_INSTANCE_CATEGORY_NAMES[i]] = n
                # Don't bother publishing the clobbered "N/A" category.
                # del score_dict[COCO_INSTANCE_CATEGORY_NAMES_NA]
                score_dicts.append(score_dict)

            formatted_dets.append(list(zip(a_bboxes, score_dicts, strict=False)))

        return formatted_dets

    @override
    def get_config(self) -> dict[str, Any]:
        return {
            "box_thresh": self.box_thresh,
            "num_dets": self.num_dets,
            "img_batch_size": self.img_batch_size,
            "use_cuda": self.use_cuda,
            "cuda_device": self.cuda_device,
        }

    @classmethod
    @override
    def is_usable(cls) -> bool:
        """
        Check if the object detector is usable by verifying optional dependencies.

        Returns:
            bool: True if all required dependencies (`torch` and `torchvision`) are installed, False otherwise.

        Example:
            >>> if MyObjectDetector.is_usable():
            >>>     detector = MyObjectDetector()
            >>> else:
            >>>     print("Optional dependencies not installed.")
        """
        # check for optional dependencies
        torch_spec = importlib.util.find_spec("torch")
        torchvision_spec = importlib.util.find_spec("torchvision")
        return torch_spec is not None and torchvision_spec is not None


try:

    def _postprocess_detections(
        self: RoIHeads,  # type: ignore
        class_logits: torch.Tensor,  # type: ignore
        box_regression: torch.Tensor,  # type: ignore
        proposals: list[torch.Tensor],  # type: ignore
        image_shapes: list[tuple[int, int]],  # type: ignore
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:  # type: ignore
        """Modified bounding box postprocessing fcn that returns class probabilites instead of just a confidence score.

        Taken from https://github.com/XAITK/xaitk-saliency/blob/master/examples/DRISE.ipynb
        """
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)  # type: ignore

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes, strict=False):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)  # type: ignore
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            scores_orig = scores.clone()
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]  # type: ignore
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            inds = inds[keep]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)  # type: ignore
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]  # type: ignore
            inds = inds[keep]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # Find corresponding row of matrix
            inds = inds // (num_classes - 1)

            all_boxes.append(boxes)
            all_scores.append(scores_orig[inds, :])
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    # Labels for this pretrained model are detailed here
    # https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection
    COCO_INSTANCE_CATEGORY_NAMES = (
        "__background__",
        "person",
        "vehicle",
        "traffic light",
    )
    COCO_INSTANCE_CATEGORY_NAMES_NA = "N/A"
except NameError:
    pass
