from .predictor import build_predictor

predictor_tiny_10k = build_predictor(
    "Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml",
    "model_weight/tiny_10k.pth",
)

predictor_tiny = build_predictor(
    "Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml",
    "model_weight/tiny_20k.pth",
)

predictor_small = build_predictor(
    "Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_small_bs16_50ep.yaml",
    "model_weight/small_20k.pth",
)

MODELS = [
    {
        "predictor": predictor_tiny_10k,
        "weight": 1,                                # weight of each model's predict
    },
    {
        "predictor": predictor_tiny,
        "weight": 1,
    },
    {
        "predictor": predictor_small,
        "weight": 1,
    },
]