from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage import binary_dilation

if TYPE_CHECKING:
    from src.models.cartoon_stylizer import CartoonStylizer
    from src.models.anime_stylizer import AnimeStylizer


@dataclass
class PixelArtConfig:
    target_long_edge: int = 256
    palette_size: int = 64
    dither: bool = False
    outline: bool = False
    edge_threshold: float = 0.12
    pre_smooth: int = 1
    color_boost: float = 1.1
    contrast_boost: float = 1.05
    mask_threshold: float = 0.5


ANIME_PIXELART_DEFAULTS = PixelArtConfig(
    target_long_edge=112,
    palette_size=20,
    outline=True,
    edge_threshold=0.07,
    color_boost=1.3,
    contrast_boost=1.15,
)

def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    safe_mask = np.clip(mask, 0.0, 1.0)
    pil = Image.fromarray((safe_mask * 255).astype(np.uint8))
    pil = pil.resize(size, Image.NEAREST)
    return np.array(pil, dtype=np.float32) / 255.0


def apply_alpha(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
    rgba.putalpha(Image.fromarray(alpha))
    return rgba


def _edge_map(rgb: np.ndarray, threshold: float) -> np.ndarray:
    lum = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    ) / 255.0
    gx = np.zeros_like(lum)
    gy = np.zeros_like(lum)
    gx[:, 1:] = np.abs(lum[:, 1:] - lum[:, :-1])
    gy[1:, :] = np.abs(lum[1:, :] - lum[:-1, :])
    return np.maximum(gx, gy) > threshold


def _mask_to_bbox(
    mask: np.ndarray,
    threshold: float = 0.5,
    pad: int = 6,
    bottom_pad: int = 20,
) -> tuple[int, int, int, int]:
    """bbox 추출. bottom_pad를 더 크게 주어 신발 등 하단 악세사리를 포함."""
    h, w = mask.shape
    ys, xs = np.where(mask >= threshold)
    if xs.size == 0 or ys.size == 0:
        return 0, 0, w, h
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, w)
    y1 = min(int(ys.max()) + bottom_pad + 1, h)
    return x0, y0, x1, y1


def _pixel_art(image: Image.Image, config: PixelArtConfig) -> Image.Image:
    w, h = image.size
    if w >= h:
        new_w = config.target_long_edge
        new_h = max(1, int(h * (config.target_long_edge / w)))
    else:
        new_h = config.target_long_edge
        new_w = max(1, int(w * (config.target_long_edge / h)))

    small = image.resize((new_w, new_h), Image.BOX)

    if config.color_boost != 1.0:
        small = ImageEnhance.Color(small).enhance(config.color_boost)
    if config.contrast_boost != 1.0:
        small = ImageEnhance.Contrast(small).enhance(config.contrast_boost)

    dither = Image.Dither.FLOYDSTEINBERG if config.dither else Image.Dither.NONE
    quant = small.convert("P", palette=Image.ADAPTIVE, colors=config.palette_size, dither=dither)
    rgb = quant.convert("RGB")

    if config.outline:
        arr = np.array(rgb)
        edges = _edge_map(arr, config.edge_threshold)
        arr[edges] = np.array([0, 0, 0], dtype=np.uint8)
        rgb = Image.fromarray(arr)

    return rgb.resize((w, h), Image.NEAREST).convert("RGBA")


def _stylizer_to_pixel_art(
    image: Image.Image,
    mask: np.ndarray,
    stylizer,
    config: PixelArtConfig,
    background: str,
    mask_dilate_px: int,
    fill_color: tuple[int, int, int],
) -> Image.Image:
    """공통 파이프라인: 마스크 dilation → crop → stylizer → 픽셀아트 → 합성."""
    binary = mask >= config.mask_threshold
    if mask_dilate_px > 0:
        struct = np.ones((mask_dilate_px * 2 + 1, mask_dilate_px * 2 + 1), dtype=bool)
        binary = binary_dilation(binary, structure=struct)
    dilated_mask = binary

    w, h = image.size
    x0, y0, x1, y1 = _mask_to_bbox(dilated_mask, threshold=config.mask_threshold)
    crop_img = image.crop((x0, y0, x1, y1)).convert("RGB")
    m_crop = dilated_mask[y0:y1, x0:x1]

    bg = Image.new("RGB", crop_img.size, fill_color)
    mask_pil = Image.fromarray((m_crop * 255).astype(np.uint8), mode="L")
    bg.paste(crop_img, mask=mask_pil)
    crop_img = bg

    styled = stylizer.apply(crop_img)
    pixeled = _pixel_art(styled, config)
    styled_masked = apply_alpha(pixeled, m_crop.astype(np.float32))

    if background == "original":
        canvas = image.convert("RGBA")
        canvas.paste(styled_masked, (x0, y0), styled_masked)
    else:
        canvas_color = (255, 255, 255, 255) if background == "white" else (0, 0, 0, 0)
        canvas = Image.new("RGBA", (w, h), canvas_color)
        canvas.paste(styled_masked, (x0, y0), styled_masked)

    return canvas


def pixel_art_person_cartoon(
    image: Image.Image,
    mask: np.ndarray,
    stylizer: CartoonStylizer,
    config: PixelArtConfig | None = None,
    background: str = "white",
    mask_dilate_px: int = 12,
) -> Image.Image:
    """mask_dilate_px: 마스크 팽창 반경(픽셀). 핸드폰·가방·신발 등 인물 인접 악세사리 포함."""
    if config is None:
        config = PixelArtConfig()
    return _stylizer_to_pixel_art(
        image, mask, stylizer, config, background, mask_dilate_px,
        fill_color=(220, 220, 220),
    )


def pixel_art_person_anime(
    image: Image.Image,
    mask: np.ndarray,
    stylizer: AnimeStylizer,
    config: PixelArtConfig | None = None,
    background: str = "white",
    mask_dilate_px: int = 12,
) -> Image.Image:
    """AnimeGAN2 일러스트 변환 → 픽셀아트."""
    if config is None:
        config = ANIME_PIXELART_DEFAULTS
    return _stylizer_to_pixel_art(
        image, mask, stylizer, config, background, mask_dilate_px,
        fill_color=(255, 255, 255),
    )
