"""Pixel-art stylization model wrapper (TorchScript)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import os
import torch
from PIL import Image
from torchvision import transforms


@dataclass
class PixelArtModelConfig:
    model_path: str
    input_size: Optional[int] = None
    normalize: bool = False


class PixelArtStylizer:
    def __init__(self, config: PixelArtModelConfig):
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"pixel-art model not found: {config.model_path}")
        self.config = config
        self.device = _resolve_device()
        self.model = torch.jit.load(config.model_path, map_location=self.device)
        self.model.eval()

        self.to_tensor = transforms.ToTensor()
        if config.normalize:
            self.normalize = transforms.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            )
        else:
            self.normalize = None

    def apply(self, image: Image.Image) -> Image.Image:
        # Convert to tensor
        img = image.convert("RGB")
        original_size = img.size

        if self.config.input_size:
            img = img.resize((self.config.input_size, self.config.input_size), Image.BILINEAR)

        x = self.to_tensor(img).unsqueeze(0)
        if self.normalize:
            x = self.normalize(x)

        with torch.inference_mode():
            y = self.model(x.to(self.device))

        # Support models that return dict or tuple
        if isinstance(y, (list, tuple)):
            y = y[0]
        if isinstance(y, dict):
            y = y.get("out", y)

        y = y.squeeze(0).clamp(0, 1)
        out = transforms.ToPILImage()(y.cpu())

        if self.config.input_size:
            out = out.resize(original_size, Image.NEAREST)
        return out


def default_model_config() -> PixelArtModelConfig:
    from src.config.settings import settings

    model_path = settings.PIXELART_MODEL_PATH
    input_size = settings.PIXELART_MODEL_INPUT
    normalize = settings.PIXELART_MODEL_NORMALIZE
    return PixelArtModelConfig(
        model_path=model_path,
        input_size=int(input_size) if input_size else None,
        normalize=normalize,
    )


def _resolve_device() -> torch.device:
    from src.config.settings import settings
    prefer = settings.PIXELART_DEVICE.lower()
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
