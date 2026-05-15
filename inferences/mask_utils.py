import cv2
import numpy as np
from pycocotools import mask as mask_utils

def mask_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / (union + 1e-6)

def mask_nms(
    masks,
    scores,
    classes,
    iou_thresh=0.5,
):
    scores_np = scores.cpu().numpy()
    classes_np = classes.cpu().numpy()
    order = scores_np.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        remain = []
        for j in order[1:]:
            if classes_np[i] != classes_np[j]:
                remain.append(j)
                continue
            
            iou = mask_iou(masks[i], masks[j])

            if iou < iou_thresh:
                remain.append(j)

        order = np.array(remain)
    return keep

def mask_to_coco_segmentation(mask):
    """
    Convert binary mask to COCO RLE format
    """

    rle = mask_utils.encode(
        np.asfortranarray(mask.astype(np.uint8))
    )

    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def refine_single_mask(
    mask,
    kernel_size=3,
    dilate_iter=1,
):
    """
    Refine thin technical drawing masks.
    """

    mask_u8 = (mask.astype(np.uint8)) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS,
        (kernel_size, kernel_size),
    )

    # close small gaps
    refined = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_CLOSE,
        kernel,
    )

    # preserve thin lines
    refined = cv2.dilate(
        refined,
        kernel,
        iterations=dilate_iter,
    )

    return refined > 0


def masks_to_boxes(masks):
    boxes = []
    for mask in masks:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            boxes.append([0, 0, 0, 0])
        else:
            boxes.append(
                [
                    np.min(xs),
                    np.min(ys),
                    np.max(xs),
                    np.max(ys),
                ]
            )

    return np.array(boxes)