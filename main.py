import os
from dataset.register_dataset import (
    register_dataset,
)

import down_dataset
import down_weights

import paddle
paddle.disable_static()

from ocr import get_ocr
from models import MODELS
from ui import create_demo
from pipelines import process_image

from Mask2Former.mask2former.modeling.pixel_decoder.ops.functions import ms_deform_attn_func
def apply_mask2former_cpu_patch():
    @staticmethod
    def patched_forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):

        output = ms_deform_attn_func.ms_deform_attn_core_pytorch(
            value,
            value_spatial_shapes,
            sampling_locations,
            attention_weights,
        )
        return output
    
    ms_deform_attn_func.MSDeformAttnFunction.forward = patched_forward
    print("CPU Patch Applied")
    
# down_dataset()

# required_files = [
#     "model_weight/tiny_10k.pth",
#     "model_weight/tiny_20k.pth",
#     "model_weight/small_20k.pth",
# ]

# if not all(os.path.exists(f) for f in required_files):
#     down_weights()

apply_mask2former_cpu_patch()
metadata = register_dataset()
models = MODELS
ocr = get_ocr()

demo = create_demo(
    lambda img: process_image(
        img
    )
)