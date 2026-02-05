"""Person segmentation model wrapper (CPU-friendly)."""

from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)


class PersonSegmenter:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.model = deeplabv3_mobilenet_v3_large(weights=weights)
        self.model.eval()

    def predict_mask(self, image: Image.Image) -> np.ndarray:
        """Return HxW mask in [0,1]."""
        with torch.inference_mode():
            input_tensor = self.preprocess(image).unsqueeze(0)
            output = self.model(input_tensor)["out"][0]
            # COCO class index for person is 15
            person_mask = output.argmax(0).eq(15).float().cpu().numpy()
            return person_mask.astype(np.float32)
