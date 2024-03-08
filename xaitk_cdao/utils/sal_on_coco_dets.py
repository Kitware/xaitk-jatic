import logging
import numpy as np
import os
from PIL import Image  # type: ignore
from typing import Callable, Iterable, Literal, Optional, Union

from maite.protocols import ObjectDetector, SupportsArray

from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects
from smqtk_image_io import AxisAlignedBoundingBox

from xaitk_cdao.interop.object_detection import JATICDetector

from xaitk_saliency import GenerateObjectDetectorBlackboxSaliency

try:
    import kwcoco  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.patches import Rectangle  # type: ignore
    from xaitk_saliency.utils.coco import parse_coco_dset
    is_usable = True
except ImportError:
    is_usable = False


def sal_on_coco_dets(
    coco_file: str,
    output_dir: str,
    sal_generator: GenerateObjectDetectorBlackboxSaliency,
    blackbox_detector: DetectImageObjects,
    overlay_image: bool = False,
    verbose: bool = False
) -> None:
    """
    Generate saliency maps for detections in a COCO format file and write them
    to disk. Maps for each detection are written out in subdirectories named
    after their corresponding image file.

    \b
    COCO_FILE - COCO style annotation file with detections to compute saliency
        for.
    OUTPUT_DIR - Directory to write the saliency maps to.

    \f
    :param coco_file: COCO style annotation file with detections to compute
        saliency for.
    :param output_dir: Directory to write the saliency maps to.
    :param sal_generator: ``GenerateObjectDetectorBlackboxSaliency`` generator
    :param blackbox_detector: ``DetectImageObjects`` detector
    :param overlay_image: Overlay saliency maps on images with bounding boxes.
        RGB images are converted to grayscale. Default is to write out saliency
        maps by themselves.
    :param verbose: Display progress messages. Default is false.
    """

    if not is_usable:
        raise ImportError("This tool requires additional dependencies, please install 'xaitk-cdao[tools]'")

    # load dets
    dets_dset = kwcoco.CocoDataset(coco_file)

    if verbose:
        logging.basicConfig(level=logging.INFO)

    img_sal_maps = [
        sal_generator(
            ref_img,
            bboxes,
            scores,
            blackbox_detector
        ) for ref_img, bboxes, scores in parse_coco_dset(dets_dset)
    ]

    # The outputs of pase_coco_dset() are constructed using gid_to_aids, so we
    # can assume the order of image and annotation ids in gid_to_aids here
    # correspond correctly to that of the generated saliency maps.
    img_skip_counter = 0
    for img_idx, (img_id, det_ids) in enumerate(dets_dset.gid_to_aids.items()):
        # skip if there are no dets for this image
        if len(det_ids) == 0:
            img_skip_counter += 1
            continue  # pragma: no cover

        img_file = dets_dset.get_image_fpath(img_id)
        ref_img = np.asarray(Image.open(img_file))

        img_file = dets_dset.imgs[img_id]['file_name']

        # split file from parent folder
        img_name = os.path.split(img_file)[1]
        # split off file extension
        img_name = os.path.splitext(img_name)[0]

        sub_dir = os.path.join(output_dir, img_name)

        os.makedirs(sub_dir, exist_ok=True)

        sal_skip_counter = 0
        for sal_idx, det_id in enumerate(det_ids):
            ann = dets_dset.anns[det_id]
            if not ('score' in ann or 'prob' in ann):
                sal_skip_counter += 1
                continue

            sal_map = img_sal_maps[img_idx - img_skip_counter][sal_idx - sal_skip_counter]

            fig = plt.figure()
            plt.axis('off')
            if overlay_image:
                gray_img = np.asarray(Image.fromarray(ref_img).convert("L"))
                plt.imshow(gray_img, alpha=0.7, cmap='gray')

                bbox = dets_dset.anns[det_id]['bbox']
                plt.gca().add_patch(Rectangle(
                    (bbox[0], bbox[1]),
                    bbox[2],
                    bbox[3],
                    linewidth=1,
                    edgecolor='r',
                    facecolor='none'
                ))
                plt.imshow(sal_map, cmap='jet', alpha=0.3)
                plt.colorbar()
            else:
                plt.imshow(sal_map, cmap='jet')
                plt.colorbar()
            plt.savefig(os.path.join(sub_dir, f"det_{det_id}.jpeg"), bbox_inches='tight')
            plt.close(fig)


def maite_sal_on_coco_dets(
    coco_file: str,
    output_dir: str,
    sal_generator: GenerateObjectDetectorBlackboxSaliency,
    maite_detector: ObjectDetector,
    bbox_transform: Union[Literal["XYXY"], Callable[[np.ndarray], Iterable[Iterable[AxisAlignedBoundingBox]]]],
    preprocessor: Optional[Callable[[SupportsArray], SupportsArray]] = None,
    img_batch_size: int = 1,
    overlay_image: bool = False,
    verbose: bool = False
) -> None:
    """
    Generate saliency maps for detections in a COCO format file and write them
    to disk. Maps for each detection are written out in subdirectories named
    after their corresponding image file.

    \b
    COCO_FILE - COCO style annotation file with detections to compute saliency
        for.
    OUTPUT_DIR - Directory to write the saliency maps to.

    \f
    :param coco_file: COCO style annotation file with detections to compute
        saliency for.
    :param output_dir: Directory to write the saliency maps to.
    :param sal_generator: ``GenerateObjectDetectorBlackboxSaliency`` generator
    :param maite_detector: Detector the implements MAITE's ObjectDetector protocol
    :param bbox_transform: Predefined bounding box format literal or callable to transform
        the JATIC detector's bboxes to AxisAlignedBoundingBoxes.
    :param preprocessor: Callable that takes a batch of data and returns a batch of data
        for any preprocessing before model inference.
    :param img_batch_size: Image batch size for inference.
    :param overlay_image: Overlay saliency maps on images with bounding boxes.
        RGB images are converted to grayscale. Default is to write out saliency
        maps by themselves.
    :param verbose: Display progress messages. Default is false.
    """

    sal_on_coco_dets(
        coco_file=coco_file,
        output_dir=output_dir,
        sal_generator=sal_generator,
        blackbox_detector=JATICDetector(
            detector=maite_detector,
            bbox_transform=bbox_transform,
            preprocessor=preprocessor,
            img_batch_size=img_batch_size
        ),
        overlay_image=overlay_image,
        verbose=verbose
    )
