import torch

torch.set_grad_enabled(False)

if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = "cpu"

# setting for ensemble_inference
SCORE_THRESH = 0.5
IOU_THRESH=0.5
BOX_NMS_THRESH = 0.4
MASK_NMS_THRESH = 0.4
MAX_DETECTIONS = 50

TTA_SCALES = [1600]
TTA_FLIP = True

# setting for init model 
NUM_CLASSES = 3
INSTANCE_ON = True
SCORE_THRESH_TEST=0.5

MIN_SIZE_TEST = 1000
MAX_SIZE_TEST = 2000
#setting for weight of model's predict
