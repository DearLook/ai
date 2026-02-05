"""Inference pipeline: person segmentation -> pixelation -> composite."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image


@dataclass
class PixelateConfig:
    block_size: int = 12


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_png(image: Image.Image, path: str) -> None:
    image.save(path, format="PNG")


def resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    # mask: HxW (0..1)
    pil = Image.fromarray((mask * 255).astype(np.uint8))
    pil = pil.resize(size, Image.NEAREST)
    return np.array(pil) / 255.0


def pixelate(image: Image.Image, block_size: int) -> Image.Image:
    w, h = image.size
    small_w = max(1, w // block_size)
    small_h = max(1, h // block_size)
    return image.resize((small_w, small_h), Image.BILINEAR).resize((w, h), Image.NEAREST)


def composite(original: Image.Image, pixelated: Image.Image, mask: np.ndarray) -> Image.Image:
    # mask: HxW in [0,1]
    orig = np.array(original).astype(np.float32)
    pix = np.array(pixelated).astype(np.float32)
    m = np.expand_dims(mask, axis=2)
    out = pix * m + orig * (1.0 - m)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def run_pipeline(input_path: str, output_path: str, mask: np.ndarray, config: PixelateConfig) -> None:
    image = load_image(input_path)
    mask_resized = resize_mask(mask, image.size)
    pix = pixelate(image, config.block_size)
    out = composite(image, pix, mask_resized)
    save_png(out, output_path)
