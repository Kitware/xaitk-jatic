import os
import urllib.request
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def dets_to_mats(dets):

    # use labels from first prediction to access all of them in a constant order
    labels = list(dets[0][1].keys())

    bboxes = np.empty((0, 4))
    scores = np.empty((0, 10))
    for det in dets:
        bbox = det[0]

        bboxes = np.vstack((
            bboxes,
            [*bbox.min_vertex, *bbox.max_vertex,]
        ))

        score_dict = det[1]
        score_array = []
        for la in labels:
            score_array.append(score_dict[la])

        scores = np.vstack((
            scores,
            score_array,
        ))

    return bboxes, scores


def display_sal_maps(sal_maps, bboxes, ref_img, title=None, figsize=(8, 4)):
    gray_img = np.asarray(Image.fromarray(ref_img).convert("L"))

    pad_perc = 0.4

    n = len(sal_maps)

    fig, axs = plt.subplots(1, n, figsize=figsize)

    for i, ax in enumerate(axs):
        try:
            sal_map = sal_maps[i]
        except IndexError:
            ax.axis('off')
            continue

        x1, y1, x2, y2 = bboxes[i]
        pad_x = pad_perc * (x2 - x1)
        pad_y = pad_perc * (y2 - y1)
        x1 = max(int(x1 - pad_x), 0)
        y1 = max(int(y1 - pad_y), 0)
        x2 = int(x2 + pad_x)
        y2 = int(y2 + pad_y)

        img_crop = gray_img[y1:(y2+1), x1:(x2+1)]
        sal_crop = sal_map[y1:(y2+1), x1:(x2+1)]

        ax.imshow(img_crop, alpha=0.7, cmap='gray')
        ax.imshow(sal_crop, alpha=0.3, cmap='jet')
        ax.axis('off')

    axs = fig.subplots()
    if title:
        axs.set_title(title, fontsize=15)
    _ = axs.axis('off')


def get_image(url, img_name, data_dir='./data'):
    os.makedirs(data_dir, exist_ok=True)

    img_path = os.path.join(data_dir, img_name)
    if not os.path.isfile(img_path):
        _ = urllib.request.urlretrieve(url, img_path)

    return Image.open(img_path)


def show_dets(ax, dets, thresh=0.5, show_labels=False):
    for i, det in enumerate(dets):
        score_dict = det[1]
        cls_name = max(score_dict, key=score_dict.get)
        conf = score_dict[cls_name]

        if conf >= thresh:
            bbox = det[0]
            x1, y1 = bbox.min_vertex
            x2, y2 = bbox.max_vertex
            ax.add_patch(Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            ))

            if show_labels:
                label = f'{cls_name} {conf:.2f}'
                ax.text(x1, y1 - 2, label, color='b', fontsize=8)
