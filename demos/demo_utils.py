import os
import urllib.request
from typing import List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from matplotlib.patches import Rectangle
from PIL import Image


def dets_to_mats(dets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # use labels from first prediction to access all of them in a constant order
    labels = list(dets[0][1].keys())

    bboxes = np.empty((0, 4))
    scores = np.empty((0, 10))
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
        for la in labels:
            score_array.append(score_dict[la])

        scores = np.vstack(
            (
                scores,
                score_array,
            )
        )

    return bboxes, scores


def display_sal_maps(
    sal_maps: List[np.ndarray],
    bboxes: List[Tuple[int, int, int, int]],
    ref_img: np.ndarray,
    title: Optional[str] = None,
    fig_size: Tuple[int, int] = (8, 4),
) -> None:
    gray_img = np.asarray(Image.fromarray(ref_img).convert("L"))

    pad_perc = 0.4

    n = len(sal_maps)

    fig, axs = plt.subplots(1, n, fig_size=fig_size)

    for i, ax in enumerate(axs):
        try:
            sal_map = sal_maps[i]
        except IndexError:
            ax.axis("off")
            continue

        x_1, y_1, x_2, y_2 = bboxes[i]
        pad_x = pad_perc * (x_2 - x_1)
        pad_y = pad_perc * (y_2 - y_1)
        x_1 = max(int(x_1 - pad_x), 0)
        y_1 = max(int(y_1 - pad_y), 0)
        x_2 = int(x_2 + pad_x)
        y_2 = int(y_2 + pad_y)

        img_crop = gray_img[y_1: (y_2 + 1), x_1: (x_2 + 1)]
        sal_crop = sal_map[y_1: (y_2 + 1), x_1: (x_2 + 1)]

        ax.imshow(img_crop, alpha=0.7, cmap="gray")
        ax.imshow(sal_crop, alpha=0.3, cmap="jet")
        ax.axis("off")

    axs = fig.subplots()
    if title:
        axs.set_title(title, fontsize=15)
    _ = axs.axis("off")


def get_image(url: str, img_name: str, data_dir: str = "./data") -> Image.Image:
    os.makedirs(data_dir, exist_ok=True)

    img_path = os.path.join(data_dir, img_name)
    if not os.path.isfile(img_path):
        _ = urllib.request.urlretrieve(url, img_path)

    return Image.open(img_path)


def show_dets(
    ax: Axis, dets: np.ndarray, thresh: float = 0.5, show_labels: bool = False
) -> None:
    for _, det in enumerate(dets):
        score_dict = det[1]
        cls_name = max(score_dict, key=score_dict.get)
        conf = score_dict[cls_name]

        if conf >= thresh:
            bbox = det[0]
            x_1, y_1 = bbox.min_vertex
            x_2, y_2 = bbox.max_vertex
            ax.add_patch(
                Rectangle(
                    (x_1, y_1),
                    x_2 - x_1,
                    y_2 - y_1,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
            )

            if show_labels:
                label = f"{cls_name} {conf:.2f}"
                ax.text(x_1, y_1 - 2, label, color="b", fontsize=8)
