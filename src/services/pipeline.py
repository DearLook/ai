"""Inference pipeline: person segmentation -> pixelation -> composite."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from src.models.pixelart_model import PixelArtStylizer
from src.models.pixelart_diffusion import PixelArtDiffusionStylizer


@dataclass
class PixelateConfig:
    block_size: int = 12


@dataclass
class PixelArtConfig:
    target_long_edge: int = 160
    palette_size: int = 48
    dither: bool = False
    outline: bool = False
    edge_threshold: float = 0.12
    pre_smooth: int = 3
    color_boost: float = 1.15
    contrast_boost: float = 1.1


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


def _edge_map(rgb: np.ndarray, threshold: float) -> np.ndarray:
    # rgb: HxWx3 uint8
    lum = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    ) / 255.0
    gx = np.zeros_like(lum)
    gy = np.zeros_like(lum)
    gx[:, 1:] = np.abs(lum[:, 1:] - lum[:, :-1])
    gy[1:, :] = np.abs(lum[1:, :] - lum[:-1, :])
    grad = np.maximum(gx, gy)
    return grad > threshold


def pixel_art(image: Image.Image, config: PixelArtConfig) -> Image.Image:
    w, h = image.size
    if w >= h:
        new_w = config.target_long_edge
        new_h = max(1, int(h * (config.target_long_edge / w)))
    else:
        new_h = config.target_long_edge
        new_w = max(1, int(w * (config.target_long_edge / h)))

    small = image.resize((new_w, new_h), Image.BOX)

    if config.pre_smooth and config.pre_smooth > 1:
        small = small.filter(ImageFilter.MedianFilter(config.pre_smooth))
    if config.color_boost and config.color_boost != 1.0:
        small = ImageEnhance.Color(small).enhance(config.color_boost)
    if config.contrast_boost and config.contrast_boost != 1.0:
        small = ImageEnhance.Contrast(small).enhance(config.contrast_boost)
    dither = Image.Dither.FLOYDSTEINBERG if config.dither else Image.Dither.NONE
    quant = small.convert("P", palette=Image.ADAPTIVE, colors=config.palette_size, dither=dither)
    rgb = quant.convert("RGB")

    if config.outline:
        arr = np.array(rgb)
        edges = _edge_map(arr, config.edge_threshold)
        arr[edges] = np.array([0, 0, 0], dtype=np.uint8)
        rgb = Image.fromarray(arr)

    pixel = rgb.resize((w, h), Image.NEAREST)
    return pixel.convert("RGBA")


def composite(original: Image.Image, pixelated: Image.Image, mask: np.ndarray) -> Image.Image:
    # mask: HxW in [0,1]
    orig = np.array(original).astype(np.float32)
    pix = np.array(pixelated).astype(np.float32)
    m = np.expand_dims(mask, axis=2)
    out = pix * m + orig * (1.0 - m)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def apply_alpha(image: Image.Image, mask: np.ndarray) -> Image.Image:
    # mask: HxW in [0,1]
    rgba = image.convert("RGBA")
    alpha = (mask * 255).astype(np.uint8)
    rgba.putalpha(Image.fromarray(alpha))
    return rgba


def _mask_to_bbox(mask: np.ndarray, threshold: float = 0.5, pad: int = 6) -> Tuple[int, int, int, int]:
    h, w = mask.shape
    ys, xs = np.where(mask >= threshold)
    if xs.size == 0 or ys.size == 0:
        return 0, 0, w, h
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, w)
    y1 = min(int(ys.max()) + pad + 1, h)
    return x0, y0, x1, y1


def _median_color(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # arr: HxWx3, mask: HxW bool
    if mask.sum() == 0:
        return np.array([128, 128, 128], dtype=np.uint8)
    return np.median(arr[mask], axis=0).astype(np.uint8)


def pixel_art_person(image: Image.Image, mask: np.ndarray, config: PixelArtConfig) -> Image.Image:
    """Generate pixel-art for person region only, returning RGBA with transparent background."""
    w, h = image.size
    x0, y0, x1, y1 = _mask_to_bbox(mask)
    arr = np.array(image)
    crop = arr[y0:y1, x0:x1]
    m_crop = mask[y0:y1, x0:x1] >= 0.5

    # Fill background inside crop with median person color to stabilize palette
    bg = _median_color(crop, m_crop)
    filled = crop.copy()
    filled[~m_crop] = bg
    crop_img = Image.fromarray(filled)

    pix = pixel_art(crop_img, config)
    pix = apply_alpha(pix.convert("RGB"), m_crop.astype(np.float32))

    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas.paste(pix, (x0, y0), pix)
    return canvas


def pixel_art_person_model(
    image: Image.Image, mask: np.ndarray, stylizer: PixelArtStylizer
) -> Image.Image:
    """Generate pixel-art via model for person region only, returning RGBA with transparent background."""
    w, h = image.size
    x0, y0, x1, y1 = _mask_to_bbox(mask)
    arr = np.array(image)
    crop = arr[y0:y1, x0:x1]
    m_crop = mask[y0:y1, x0:x1] >= 0.5

    # Fill background inside crop with median person color to stabilize model input
    bg = _median_color(crop, m_crop)
    filled = crop.copy()
    filled[~m_crop] = bg
    crop_img = Image.fromarray(filled)

    styled = stylizer.apply(crop_img)
    styled = apply_alpha(styled.convert("RGB"), m_crop.astype(np.float32))

    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas.paste(styled, (x0, y0), styled)
    return canvas


def pixel_art_person_diffusion(
    image: Image.Image, mask: np.ndarray, stylizer: PixelArtDiffusionStylizer
) -> Image.Image:
    """Generate pixel-art via diffusion for person region only, returning RGBA with transparent background."""
    w, h = image.size
    x0, y0, x1, y1 = _mask_to_bbox(mask)
    arr = np.array(image)
    crop = arr[y0:y1, x0:x1]
    m_crop = mask[y0:y1, x0:x1] >= 0.5

    # Fill background inside crop with median person color to stabilize model input
    bg = _median_color(crop, m_crop)
    filled = crop.copy()
    filled[~m_crop] = bg
    crop_img = Image.fromarray(filled)

    styled = stylizer.apply(crop_img)
    styled = apply_alpha(styled.convert("RGB"), m_crop.astype(np.float32))

    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas.paste(styled, (x0, y0), styled)
    return canvas


def run_pipeline(input_path: str, output_path: str, mask: np.ndarray, config: PixelateConfig) -> None:
    image = load_image(input_path)
    mask_resized = resize_mask(mask, image.size)
    pix = pixelate(image, config.block_size)
    out = composite(image, pix, mask_resized)
    save_png(out, output_path)
