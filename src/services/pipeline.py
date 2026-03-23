from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage import binary_dilation

import torch
from pixeloe.torch.pixelize import pixelize

if TYPE_CHECKING:
    from src.models.cartoon_stylizer import CartoonStylizer
    from src.models.anime_stylizer import AnimeStylizer
    from src.models.controlnet_stylizer import ControlNetStylizer


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
    target_long_edge=128,
    palette_size=48,
    outline=False,
    color_boost=1.2,
    contrast_boost=1.1,
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


def _pixel_art_pixeloe(image: Image.Image, config: PixelArtConfig) -> Image.Image:
    """pixeloe 기반 픽셀아트 변환 — 엣지 인식, 깔끔한 캐릭터 스프라이트 스타일."""
    original_size = image.size
    w, h = original_size
    pixel_size = max(4, max(w, h) // config.target_long_edge)

    # pixeloe 내부 다운스케일(pixel_size) 후 wavelet radius(32) 보장
    min_w = pixel_size * 64
    min_h = pixel_size * 64
    if w < min_w or h < min_h:
        image = image.resize((max(w, min_w), max(h, min_h)), Image.BILINEAR)

    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    result_t = pixelize(
        img_t,
        pixel_size=pixel_size,
        thickness=2,
        mode="contrast",
        do_color_match=False,
        do_quant=True,
        num_colors=config.palette_size,
    )
    result_arr = (result_t.squeeze(0).permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(result_arr).resize(original_size, Image.NEAREST).convert("RGBA")


def _person_pipeline(
    image: Image.Image,
    mask: np.ndarray,
    mask_threshold: float,
    mask_dilate_px: int,
    fill_color: tuple[int, int, int],
    background: str,
    stylize_fn: Callable[[Image.Image], Image.Image],
    pixelize_fn: Callable[[Image.Image], Image.Image],
) -> Image.Image:
    """공통 파이프라인: dilation → bbox crop → 흰 배경 합성 → stylize → pixelize → alpha 합성 → canvas."""
    binary = mask >= mask_threshold
    if mask_dilate_px > 0:
        struct = np.ones((mask_dilate_px * 2 + 1, mask_dilate_px * 2 + 1), dtype=bool)
        binary = binary_dilation(binary, structure=struct)

    w, h = image.size
    x0, y0, x1, y1 = _mask_to_bbox(binary, threshold=mask_threshold)
    crop_img = image.crop((x0, y0, x1, y1)).convert("RGB")
    m_crop = binary[y0:y1, x0:x1]

    bg = Image.new("RGB", crop_img.size, fill_color)
    mask_pil = Image.fromarray((m_crop * 255).astype(np.uint8), mode="L")
    bg.paste(crop_img, mask=mask_pil)

    styled = stylize_fn(bg)
    pixeled = pixelize_fn(styled)
    styled_masked = apply_alpha(pixeled, m_crop.astype(np.float32))

    if background == "original":
        canvas = image.convert("RGBA")
        canvas.paste(styled_masked, (x0, y0), styled_masked)
    else:
        canvas_color = (255, 255, 255, 255) if background == "white" else (0, 0, 0, 0)
        canvas = Image.new("RGBA", (w, h), canvas_color)
        canvas.paste(styled_masked, (x0, y0), styled_masked)

    return canvas


def _color_transfer(source: Image.Image, target: Image.Image) -> Image.Image:
    """원본(source) 색감을 타겟(target)에 이식 (LAB 통계 전이)."""
    src = np.array(source.convert("RGB")).astype(np.float32)
    tgt = np.array(target.convert("RGB")).astype(np.float32)

    def rgb_to_lab(img):
        img = img / 255.0
        mask = img > 0.04045
        img = np.where(mask, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)
        m = np.array([[0.4124, 0.3576, 0.1805],
                    [0.2126, 0.7152, 0.0722],
                    [0.0193, 0.1192, 0.9505]])
        xyz = img @ m.T
        xyz /= [0.95047, 1.00000, 1.08883]
        eps = 0.008856
        xyz = np.where(xyz > eps, xyz ** (1/3), 7.787 * xyz + 16/116)
        L = 116 * xyz[..., 1] - 16
        a = 500 * (xyz[..., 0] - xyz[..., 1])
        b = 200 * (xyz[..., 1] - xyz[..., 2])
        return np.stack([L, a, b], axis=-1)

    def lab_to_rgb(lab):
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200
        xyz = np.stack([fx, fy, fz], axis=-1)
        eps = 0.008856
        xyz = np.where(xyz ** 3 > eps, xyz ** 3, (xyz - 16/116) / 7.787)
        xyz *= [0.95047, 1.00000, 1.08883]
        m_inv = np.array([[ 3.2406, -1.5372, -0.4986],
                        [-0.9689,  1.8758,  0.0415],
                        [ 0.0557, -0.2040,  1.0570]])
        rgb = xyz @ m_inv.T
        rgb = np.clip(rgb, 0, 1)
        mask = rgb > 0.0031308
        rgb = np.where(mask, 1.055 * rgb ** (1/2.4) - 0.055, 12.92 * rgb)
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    src_lab = rgb_to_lab(src)
    tgt_lab = rgb_to_lab(tgt)

    for ch in range(3):
        s_mean, s_std = src_lab[..., ch].mean(), src_lab[..., ch].std() + 1e-6
        t_mean, t_std = tgt_lab[..., ch].mean(), tgt_lab[..., ch].std() + 1e-6
        tgt_lab[..., ch] = (tgt_lab[..., ch] - t_mean) * (s_std / t_std) + s_mean

    result_rgb = lab_to_rgb(tgt_lab)
    result = Image.fromarray(result_rgb)
    return result.resize(target.size, Image.BILINEAR)


def pixel_art_person_controlnet(
    image: Image.Image,
    mask: np.ndarray,
    stylizer: "ControlNetStylizer",
    background: str = "white",
    mask_dilate_px: int = 4,
    mask_threshold: float = 0.5,
    pixel_target: int = 128,
    palette_size: int = 32,
) -> Image.Image:
    """ControlNet + SD1.5 + LoRA 픽셀아트 변환 후 pixeloe 마무리."""
    config = PixelArtConfig(target_long_edge=pixel_target, palette_size=palette_size)

    captured_original: list[Image.Image] = []

    def stylize_and_capture(img: Image.Image) -> Image.Image:
        captured_original.append(img.copy())
        return stylizer.apply(img)

    def pixelize_with_orig_palette(styled: Image.Image) -> Image.Image:
        orig = captured_original[0] if captured_original else styled
        orig_resized = orig.resize(styled.size, Image.BILINEAR).convert("RGB")

        # 원본 이미지에서 팔레트 추출
        q = orig_resized.quantize(colors=config.palette_size, method=Image.Quantize.MEDIANCUT)
        raw = q.getpalette()
        orig_palette = np.array(raw[:config.palette_size * 3]).reshape(-1, 3).astype(np.float32)

        # HSV 기반 팔레트 보정: 색조 유지 + 채도 강화 + 어두운 색만 밝기 보정
        import colorsys
        boosted = []
        for color in orig_palette:
            r, g, b = color / 255.0
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            s = min(1.0, s * 2.2)       # 채도 강화
            v = min(1.0, v * 1.4 + 0.08) if v < 0.5 else min(1.0, v * 1.1)  # 어두운 색만 밝기 보정
            r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
            boosted.append([r2 * 255, g2 * 255, b2 * 255])
        orig_palette = np.array(boosted, dtype=np.float32)

        # styled 이미지를 pixeloe로 픽셀화
        pixelized = _pixel_art_pixeloe(styled, config).convert("RGB")

        pix_arr = np.array(pixelized).astype(np.float32)
        h, w = pix_arr.shape[:2]
        flat = pix_arr.reshape(-1, 3)

        # 각 픽셀 → 원본 팔레트 최근접 색 매핑
        dists = np.sum((flat[:, None, :] - orig_palette[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(dists, axis=1)
        remapped = orig_palette[nearest].reshape(h, w, 3).clip(0, 255).astype(np.uint8)

        result = Image.fromarray(remapped)
        result = ImageEnhance.Color(result).enhance(1.2)
        return result.convert("RGBA")

    return _person_pipeline(
        image, mask,
        mask_threshold=mask_threshold,
        mask_dilate_px=mask_dilate_px,
        fill_color=(255, 255, 255),
        background=background,
        stylize_fn=stylize_and_capture,
        pixelize_fn=pixelize_with_orig_palette,
    )


def pixel_art_person_anime(
    image: Image.Image,
    mask: np.ndarray,
    stylizer: "AnimeStylizer",
    config: PixelArtConfig | None = None,
    background: str = "white",
    mask_dilate_px: int = 4,
) -> Image.Image:
    """AnimeGAN2 일러스트 변환 → pixeloe 픽셀아트."""
    if config is None:
        config = ANIME_PIXELART_DEFAULTS

    def _stylize(img: Image.Image) -> Image.Image:
        if config.color_boost != 1.0:
            img = ImageEnhance.Color(img).enhance(config.color_boost)
        if config.contrast_boost != 1.0:
            img = ImageEnhance.Contrast(img).enhance(config.contrast_boost)
        styled = stylizer.apply(img)
        styled = ImageEnhance.Color(styled).enhance(1.5)
        styled = ImageEnhance.Brightness(styled).enhance(1.2)
        styled = ImageEnhance.Contrast(styled).enhance(1.1)
        return styled

    return _person_pipeline(
        image, mask,
        mask_threshold=config.mask_threshold,
        mask_dilate_px=mask_dilate_px,
        fill_color=(255, 255, 255),
        background=background,
        stylize_fn=_stylize,
        pixelize_fn=lambda img: _pixel_art_pixeloe(img, config),
    )


def pixel_art_person_cartoon(
    image: Image.Image,
    mask: np.ndarray,
    stylizer: "CartoonStylizer",
    config: PixelArtConfig | None = None,
    background: str = "white",
    mask_dilate_px: int = 12,
) -> Image.Image:
    """mask_dilate_px: 마스크 팽창 반경(픽셀). 핸드폰·가방·신발 등 인물 인접 악세사리 포함."""
    if config is None:
        config = PixelArtConfig()
    return _person_pipeline(
        image, mask,
        mask_threshold=config.mask_threshold,
        mask_dilate_px=mask_dilate_px,
        fill_color=(220, 220, 220),
        background=background,
        stylize_fn=stylizer.apply,
        pixelize_fn=lambda img: _pixel_art(img, config),
    )
