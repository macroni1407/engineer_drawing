import cv2
import torch
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
    # handle torch / numpy
    if isinstance(scores, torch.Tensor):
        scores_np = scores.cpu().numpy()
    else:
        scores_np = scores

    if isinstance(classes, torch.Tensor):
        classes_np = classes.cpu().numpy()
    else:
        classes_np = classes

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


def soft_mask_voting(cluster_masks, cluster_scores, temperature=1.0):
    """
    Soft voting (giữ xác suất thay vì bool)
    """
    masks = np.array(cluster_masks).astype(np.float32)
    scores = np.array(cluster_scores).astype(np.float32)

    # normalize score → weight
    weights = scores ** temperature
    weights = weights / (weights.sum() + 1e-6)

    weights = weights.reshape(-1, 1, 1)

    soft_mask = (masks * weights).sum(axis=0)

    return soft_mask  # float [0,1]


def adaptive_binarize(mask, base_thresh=0.4):
    """
    Threshold thích nghi theo density
    """
    mean_val = mask.mean()

    # mask mỏng → giảm threshold
    if mean_val < 0.1:
        thresh = base_thresh * 0.7
    elif mean_val > 0.5:
        thresh = base_thresh * 1.2
    else:
        thresh = base_thresh

    return mask > thresh


def boundary_refine(mask):
    """
    Preserve thin lines + clean noise
    """
    mask_u8 = (mask * 255).astype(np.uint8)

    # 1. edge detection
    edges = cv2.Canny(mask_u8, 50, 150)

    # 2. skeletonize-like (thin lines)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    thin = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    # 3. combine edge + mask
    combined = cv2.bitwise_or(thin, edges)

    # 4. remove noise (small components)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined)

    cleaned = np.zeros_like(combined)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 10:
            cleaned[labels == i] = 255

    return cleaned > 0