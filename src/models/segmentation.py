from typing import Optional
import numpy as np
import torch
from PIL import Image
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)


class PersonSegmenter:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.device = _resolve_device()
        self.model = deeplabv3_mobilenet_v3_large(weights=weights).to(self.device)
        self.model.eval()

    def predict_mask(self, image: Image.Image) -> np.ndarray:
        with torch.inference_mode():
            input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            output = self.model(input_tensor)["out"][0]
            person_mask = output.argmax(0).eq(15).float().cpu().numpy()
            return person_mask.astype(np.float32)


def _resolve_device() -> torch.device:
    from src.config.settings import settings
    prefer = settings.PIXELART_DEVICE.lower()
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
