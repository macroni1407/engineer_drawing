from unittest.mock import MagicMock
import sys
sys.modules["MultiScaleDeformableAttention"] = MagicMock()

from main import demo
from configs import *

demo.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        share=True,
      )