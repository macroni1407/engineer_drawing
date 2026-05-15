from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config

from Mask2Former.mask2former import add_maskformer2_config

from configs import *

def build_predictor(config_path, weight_path):

    cfg = get_cfg()

    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    cfg.merge_from_file(config_path)

    cfg.MODEL.WEIGHTS = weight_path

    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = NUM_CLASSES

    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = INSTANCE_ON

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_TEST

    cfg.MODEL.DEVICE = DEVICE

    predictor = DefaultPredictor(cfg)

    return predictor