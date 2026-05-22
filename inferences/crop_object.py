import cv2
import numpy as np

from configs import CLASS_PRIORITY, PRIORITY_WEIGHT
from .mask_utils import mask_to_coco_segmentation
from dataset import register_dataset

metadata = register_dataset()

def crop_objects_from_masks(
    image_rgb,
    instances,
    image_name="image.jpg",
    contain_threshold=0.8,
):
    """
    Smart crop objects from masks.

    Rules:
    - class priority
    - small area first
    - containment ignore
    - score tie-break
    - smooth edge
    - white bg OR transparent bg
    """

    if len(instances) == 0:
      return [], {
          "image": image_name,
          "objects": []
      }

    masks = instances.pred_masks.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()

    h, w = masks[0].shape

    # =====================================================
    # CLASS PRIORITY
    # Higher number = higher priority
    # =====================================================
    # CLASS_PRIORITY = {
    #     0: 3,  # note
    #     1: 1,  # partdrawing
    #     2: 2,  # table
    # }

    # =====================================================
    # BUILD MASK INFO
    # =====================================================
    mask_infos = []

    for idx in range(len(masks)):

        area = masks[idx].sum()

        priority = CLASS_PRIORITY.get(
            int(classes[idx]),
            0,
        )

        score = float(scores[idx])

        effective_score = score + PRIORITY_WEIGHT * priority

        mask_infos.append(
            {
                "idx": idx,
                "priority": priority,
                "score": score,
                "effective_score": effective_score,
                "area": area,
            }
        )

    # =====================================================
    # SORT RULES
    #
    # 1. higher class priority
    # 2. smaller area first
    # 3. higher score
    # =====================================================
    mask_infos = sorted(
        mask_infos,
        key=lambda x: (
            -x["effective_score"],
            x["area"],
            -x["score"],
        ),
    )

    # =====================================================
    # OVERLAP HANDLING
    # =====================================================
    occupied = np.zeros((h, w), dtype=bool)

    restored_masks = [None] * len(masks)

    for info in mask_infos:

        idx = info["idx"]

        current = masks[idx].copy()

        current_area = current.sum()

        # =================================================
        # CHECK CONTAINMENT
        # =================================================
        intersection = np.logical_and(
            current,
            occupied,
        ).sum()

        contain_ratio = (
            intersection / (current_area + 1e-6)
        )

        # =================================================
        # CONTAINMENT IGNORE
        #
        # If object mostly inside another object:
        # keep full object
        # =================================================
        if contain_ratio > contain_threshold:

            restored_masks[idx] = current

            continue

        # =================================================
        # NORMAL OVERLAP REMOVAL
        # =================================================
        current = np.logical_and(
            current,
            ~occupied,
        )

        # skip empty
        if current.sum() == 0:

            restored_masks[idx] = current

            continue

        restored_masks[idx] = current

        occupied = np.logical_or(
            occupied,
            current,
        )

    # =====================================================
    # CREATE CROPS
    # =====================================================
    metadata_json = {
        "image": image_name,
        "objects": []
    }
    
    crop_results = []

    for i, mask in enumerate(restored_masks):
        if mask is None:
            continue
        if mask.sum() == 0:
            continue

        ys, xs = np.where(mask)

        if len(xs) == 0 or len(ys) == 0:
            continue

        # =================================================
        # PADDED BOX
        # =================================================
        # object size
        obj_w = xs.max() - xs.min()
        obj_h = ys.max() - ys.min()

        # dynamic padding
        padding = max(
            10,
            int(max(obj_w, obj_h) * 0.05)
        )

        x1 = max(xs.min() - padding, 0)
        y1 = max(ys.min() - padding, 0)

        x2 = min(xs.max() + padding, w)
        y2 = min(ys.max() + padding, h)


        # =================================================
        # CROP
        # =================================================
        crop_rgb = image_rgb[
            y1:y2,
            x1:x2,
        ].copy()

        crop_mask = mask[
            y1:y2,
            x1:x2,
        ]

        # =================================================
        # MASK SMOOTHING
        # =================================================
        mask_u8 = (
            crop_mask.astype(np.uint8) * 255
        )

        # fill small holes
        mask_u8 = cv2.morphologyEx(
            mask_u8,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8),
        )

        # smooth edge
        mask_u8 = cv2.GaussianBlur(
            mask_u8,
            (5, 5),
            0,
        )


        # =================================================
        # WHITE BACKGROUND
        # =================================================
        # BLENDING
        alpha = mask_u8.astype(np.float32) / 255.0

        # giữ inside object full opacity
        alpha[crop_mask] = 1.0

        white_bg = np.ones_like(
            crop_rgb,
            dtype=np.float32
        ) * 255

        output = (
            crop_rgb.astype(np.float32)
            * alpha[..., None]
            + white_bg * (1 - alpha[..., None])
        ).astype(np.uint8)

        label_name = metadata.thing_classes[
            int(classes[i])
        ]

        scale_factor = 2.0
        new_w = int(output.shape[1] * scale_factor)
        new_h = int(output.shape[0] * scale_factor)

        output = cv2.resize(
            output,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC
        )

        crop_results.append(
            (
                output,
                label_name,
                round(
                    float(scores[i]),
                    4,
                ),
                i + 1
            )
        )

        # =====================================================
        # METADATA
        # =====================================================
        segmentation = mask_to_coco_segmentation(mask)

        metadata_json["objects"].append(
            {
                "id": i + 1,
                "class": label_name,
                "confidence": round(
                    float(scores[i]),
                    4,
                ),

                "bbox": {
                    "x1": int(xs.min()),
                    "y1": int(ys.min()),
                    "x2": int(xs.max()),
                    "y2": int(ys.max()),
                },

                "segmentation": segmentation,
            }
        )

    return crop_results, metadata_json