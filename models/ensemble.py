
import torch
import numpy as np
from detectron2.structures import Instances, Boxes
from detectron2.layers import batched_nms

from .load_models import MODELS
from inferences import *

def ensemble_inference(
    image,
    scales=[1600],
    flip=True,
    score_thresh=0.5,
    box_nms_thresh=0.4,
    mask_nms_thresh=0.4,
    max_detections=50,
):

    h0, w0 = image.shape[:2]

    all_boxes = []
    all_scores = []
    all_classes = []
    all_masks = []

    # =====================================================
    # RUN ALL MODELS
    # =====================================================
    for model_data in MODELS:
        predictor = model_data["predictor"]
        model_weight = model_data["weight"]

        result = tta_inference(
            predictor=predictor,
            image=image,
            scales=scales,
            flip=flip,
            score_thresh=score_thresh,
        )

        if len(result["boxes"]) == 0:
            continue

        all_boxes.extend(result["boxes"])

        weighted_scores = [
            min(s * model_weight, 1.0)
            for s in result["scores"]
        ]

        all_scores.extend(weighted_scores)
        all_classes.extend(result["classes"])
        all_masks.extend(result["masks"])

    # =====================================================
    # EMPTY
    # =====================================================
    if len(all_boxes) == 0:
        return Instances((h0, w0))

    # =====================================================
    # TO TENSOR
    # =====================================================
    all_boxes = torch.tensor(np.array(all_boxes))
    all_scores = torch.tensor(np.array(all_scores))
    all_classes = torch.tensor(np.array(all_classes))

    # =====================================================
    # BOX NMS
    # =====================================================
    keep_box = batched_nms(
        all_boxes,
        all_scores,
        all_classes,
        box_nms_thresh,
    )

    if len(keep_box) == 0:
        return Instances((h0, w0))

    kept_masks = [
        all_masks[i]
        for i in keep_box.numpy()
    ]

    kept_scores = all_scores[keep_box].numpy()

    # =====================================================
    # MASK NMS
    # =====================================================
    keep_mask = mask_nms(
        kept_masks,
        all_scores[keep_box],
        all_classes[keep_box],
        iou_thresh=mask_nms_thresh,
    )

    if len(keep_mask) == 0:
        return Instances((h0, w0))

    keep_mask = keep_mask[:max_detections]

    # =====================================================
    # FINAL SELECTED
    # =====================================================
    final_masks = [
        kept_masks[i]
        for i in keep_mask
    ]

    # final_scores = [
    #     kept_scores[i]
    #     for i in keep_mask
    # ]

    final_boxes = all_boxes[keep_box][keep_mask]
    final_scores_tensor = all_scores[keep_box][keep_mask]
    final_classes = all_classes[keep_box][keep_mask]

    # =====================================================
    # POST-PROCESSING
    # =====================================================

    # 1. Refine each mask
    refined_masks = []
    for mask in final_masks:
        # refined = refine_single_mask(
        #     mask,
        #     eps=15,
        #     min_samples=2,
        # )
        refined = refine_single_mask(mask)
        # refined = mask

        refined_masks.append(refined)
    refined_masks = np.array(refined_masks)
    resolved_masks = refined_masks

    # =====================================================
    # 3. Recompute boxes from refined masks
    # =====================================================
    refined_boxes = masks_to_boxes(
        resolved_masks
    )

    # =====================================================
    # FINAL INSTANCES
    # =====================================================
    final = Instances((h0, w0))
    final.pred_boxes = Boxes(
        torch.tensor(refined_boxes).float()
    )
    final.scores = final_scores_tensor
    final.pred_classes = final_classes
    final.pred_masks = torch.from_numpy(
        resolved_masks
    )
    return final
