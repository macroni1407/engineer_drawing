import torch
import numpy as np

from detectron2.structures import Boxes
from detectron2.utils.visualizer import Visualizer
from inferences import masks_to_boxes
from dataset import register_dataset

metadata = register_dataset()

def visualize_predictions(
    image_rgb,
    instances,
):
    if len(instances) == 0:
        return image_rgb

    masks_np = instances.pred_masks.numpy()
    boxes_np = masks_to_boxes(masks_np)
    instances.pred_boxes = Boxes(
        torch.tensor(boxes_np)
    )

    labels = [
        f"{metadata.thing_classes[int(c)]} {float(s):.2f}"
        for c, s in zip(
            instances.pred_classes,
            instances.scores,
        )
    ]

    v = Visualizer(
        image_rgb,
        metadata=metadata,
        scale=1,
    )

    out = v.overlay_instances(
        boxes=instances.pred_boxes,
        masks=masks_np,
        labels=labels,
        alpha=0.35,
    )

    return out.get_image()