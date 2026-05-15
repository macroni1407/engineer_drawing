import cv2
import numpy as np

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
        "boxes": all_boxes,
        "scores": all_scores,
        "classes": all_classes,
        "masks": all_masks,
    }


