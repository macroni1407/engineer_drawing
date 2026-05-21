import cv2
import torch
import numpy as np
from detectron2.layers import batched_nms
from detectron2.structures import Instances, Boxes
from inferences import *

def tta_inference(
    predictor,
    image,
    scales=[1600],
    flip=True,
    score_thresh=0.5,
):
    h0, w0 = image.shape[:2]
    all_boxes = []
    all_scores = []
    all_classes = []
    all_masks = []

    for scale in scales:
        scale_factor = scale / min(h0, w0)
        new_h = int(h0 * scale_factor)
        new_w = int(w0 * scale_factor)

        resized = cv2.resize(image, (new_w, new_h))
        augmentations = [(resized, False)]

        if flip:
            augmentations.append(
                (cv2.flip(resized, 1), True)
            )

        for aug_img, is_flipped in augmentations:
            outputs = predictor(aug_img)
            inst = outputs["instances"].to("cpu")

            if len(inst) == 0:
                continue

            keep = inst.scores > score_thresh
            inst = inst[keep]

            if len(inst) == 0:
                continue

            # =====================================================
            # BOXES
            # =====================================================
            boxes = inst.pred_boxes.tensor.numpy()
            if is_flipped:
                x1 = boxes[:, 0].copy()
                x2 = boxes[:, 2].copy()
                boxes[:, 0] = new_w - x2
                boxes[:, 2] = new_w - x1

            boxes[:, [0, 2]] *= (w0 / new_w)
            boxes[:, [1, 3]] *= (h0 / new_h)

            # =====================================================
            # MASKS
            # =====================================================
            masks = inst.pred_masks.numpy()
            restored_masks = []
            for m in masks:
                if is_flipped:
                    m = np.fliplr(m)

                m = cv2.resize(
                    m.astype(np.uint8),
                    (w0, h0),
                    interpolation=cv2.INTER_NEAREST,
                    # interpolation=cv2.INTER_LINEAR
                )
                # m = m > 0.5

                restored_masks.append(m.astype(bool))
            all_boxes.extend(boxes)
            all_scores.extend(inst.scores.numpy())
            all_classes.extend(inst.pred_classes.numpy())
            all_masks.extend(restored_masks)

    return {
        "boxes": np.array(all_boxes),
        "scores": np.array(all_scores),
        "classes": np.array(all_classes),
        "masks": np.array(all_masks),
    }


def inference_with_tta(
    predictor,
    image,
    scales=[1600],
    flip=True,
    score_thresh=0.5,
    box_nms_thresh=0.4,
    mask_nms_thresh=0.4,
    max_detections=50,
):
    h, w = image.shape[:2]

    result = tta_inference(
        predictor,
        image,
        scales=scales,
        flip=flip,
        score_thresh=score_thresh,
    )

    if len(result["boxes"]) == 0:
        return Instances((h, w))
        # return empty_instances(h, w)

    boxes = torch.tensor(result["boxes"]).float()
    scores = torch.tensor(result["scores"])
    classes = torch.tensor(result["classes"])
    masks = result["masks"]

    # ================= BOX NMS =================
    keep = batched_nms(
        boxes,
        scores,
        classes,
        box_nms_thresh
    )

    keep = keep[:max_detections]

    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]
    masks = masks[keep.numpy()]

    # ================= MASK NMS =================
    keep_mask = mask_nms(
        masks,
        scores,
        classes,
        iou_thresh=mask_nms_thresh
    )

    keep_mask = keep_mask[:max_detections]

    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    classes = classes[keep_mask]
    masks = masks[keep_mask]

    # ================= FINAL INSTANCES =================
    instances = Instances((h, w))
    instances.pred_boxes = Boxes(boxes)
    instances.scores = scores
    instances.pred_classes = classes
    instances.pred_masks = torch.from_numpy(masks)

    return instances


