from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

from configs import CLASS_NAMES

DATASET_NAME = "bom_train"

def register_dataset():

    try:
        register_coco_instances(
            DATASET_NAME,
            {},
            "bom_set/instances_Train2.json",
            "bom_set/images",
        )
    except:
        pass

    metadata = MetadataCatalog.get(DATASET_NAME)

    metadata.thing_classes = CLASS_NAMES

    return metadata