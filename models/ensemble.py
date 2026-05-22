
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
    iou_thresh=0.5,
    box_nms_thresh=0.4,
    mask_nms_thresh=0.4,
    max_detections=100,
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

        result = inference_with_tta(
            predictor=predictor,
            image=image,
            scales=scales,
            flip=flip,
            score_thresh=score_thresh,
            box_nms_thresh=box_nms_thresh,
            mask_nms_thresh=mask_nms_thresh,
            max_detections=max_detections,
        )

        if len(result) == 0:
            continue

        masks = result.pred_masks.cpu().numpy()
        scores = result.scores.cpu().numpy() * model_weight
        classes = result.pred_classes.cpu().numpy()

        all_masks.extend(masks)
        all_scores.extend(scores)
        all_classes.extend(classes)

    # =====================================================
    # EMPTY CHECK
    # =====================================================
    if len(all_masks) == 0:
        return Instances((h0, w0))

    all_masks = list(all_masks)
    all_scores = list(all_scores)
    all_classes = list(all_classes)

    final_masks = []
    final_scores = []
    final_classes = []

    # =====================================================
    # PROCESS PER CLASS
    # =====================================================
    unique_classes = np.unique(all_classes)

    for cls in unique_classes:
        # lấy indices của class này
        idxs = [i for i in range(len(all_masks)) if all_classes[i] == cls]

        masks_cls = [all_masks[i] for i in idxs]
        scores_cls = [all_scores[i] for i in idxs]

        # sort theo score giảm dần
        order = np.argsort(scores_cls)[::-1]

        masks_cls = [masks_cls[i] for i in order]
        scores_cls = [scores_cls[i] for i in order]

        used = [False] * len(masks_cls)

        # =====================================================
        # CLUSTERING
        # =====================================================
        for i in range(len(masks_cls)):
            if used[i]:
                continue

            base_mask = masks_cls[i]

            cluster_masks = [base_mask]
            cluster_scores = [scores_cls[i]]
            used[i] = True

            for j in range(i + 1, len(masks_cls)):
                if used[j]:
                    continue

                iou = mask_iou(base_mask, masks_cls[j])
                # dice = soft_dice(base_mask, masks_cls[j])

                if iou > iou_thresh:
                # if dice > dice_thresh:
                    cluster_masks.append(masks_cls[j])
                    cluster_scores.append(scores_cls[j])
                    used[j] = True

            # vote
            soft_mask = soft_mask_voting(cluster_masks, cluster_scores)

            binary_mask = adaptive_binarize(soft_mask)
            
            refined = boundary_refine(binary_mask)
            refined = refine_single_mask(refined)

            final_masks.append(refined)
            final_scores.append(max(cluster_scores))
            final_classes.append(cls)

    # =====================================================
    # LIMIT
    # =====================================================
    if len(final_masks) == 0:
        return Instances((h0, w0))

    final_masks = np.array(final_masks)
    final_scores = np.array(final_scores)
    final_classes = np.array(final_classes)

    # sort final theo score
    order = np.argsort(final_scores)[::-1][:max_detections]

    final_masks = final_masks[order]
    final_scores = final_scores[order]
    final_classes = final_classes[order]

    keep = mask_nms(
        final_masks,
        final_scores,
        final_classes,
        iou_thresh=mask_nms_thresh
    )

    final_masks = final_masks[keep]
    final_scores = final_scores[keep]
    final_classes = final_classes[keep]

    # recompute boxes
    final_boxes = masks_to_boxes(final_masks)

    # =====================================================
    # FINAL INSTANCES
    # =====================================================
    instances = Instances((h0, w0))
    instances.pred_boxes = Boxes(torch.tensor(final_boxes).float())
    instances.scores = torch.tensor(final_scores)
    instances.pred_classes = torch.tensor(final_classes)
    instances.pred_masks = torch.from_numpy(final_masks)

    return instances
