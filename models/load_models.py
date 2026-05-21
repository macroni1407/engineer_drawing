from .predictor import build_predictor

predictor_baseIN21k_20k = build_predictor(
    "Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_50ep.yaml",
    "model_weight/base_IN21k.pth",
)

predictor_base_20k = build_predictor(
    "Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_base_384_bs16_50ep.yaml",
    "model_weight/base_20k.pth",
)

MODELS = [
    {
        "predictor": predictor_baseIN21k_20k,
        "weight": 1,                                # weight of each model's predict
    },
    {
        "predictor": predictor_base_20k,
        "weight": 1,
    },
]