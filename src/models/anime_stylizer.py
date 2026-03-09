"""
AnimeStylizer — AnimeGAN2 기반 사진 → 애니 스타일 변환.
bryandlee/animegan2-pytorch (MIT License) torch.hub으로 로드.
모델 크기: ~9MB, MPS/CPU 동작.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import torch
import numpy as np
from PIL import Image, ImageEnhance

REPO = "bryandlee/animegan2-pytorch:main"

def __init__(self, config: AnimeConfig | None = None) -> None:
    self.config = config or AnimeConfig()
    self.device = self.resolve_device(self.config.device)
    self.model = torch.hub.load(
        self.REPO,
        "generator",
        pretrained=self.config.pretrained,
        trust_repo=True,
    )
    self.model.to(self._device).eval()


@dataclass
class AnimeConfig:
    pretrained: str = "face_paint_512_v2"  # face_paint_512_v1/v2, celeba_distill, paprika
    device: str = "cpu"
    pre_brightness: float = 1.3
    pre_saturation: float = 1.4
    post_saturation: float = 1.1
    post_contrast: float = 1.05
    # AnimeGAN2 스타일 vs 원본 색상 블렌드 비율 (0.0=원본, 1.0=완전 애니)
    anime_blend: float = 0.75
    # bilateral 평탄화 (카툰 색 영역 강화)
    bilateral_passes: int = 7
    # AnimeGAN2 출력 후 추가 bilateral (색 평탄화 강화)
    post_bilateral_passes: int = 4


class AnimeStylizer:
    """사진을 AnimeGAN2로 애니 일러스트 스타일로 변환."""

    REPO = "bryandlee/animegan2-pytorch:main"

    def __init__(self, config: AnimeConfig | None = None) -> None:
        self.config = config or AnimeConfig()
        self._device = self._resolve_device(self.config.device)
        self.model = torch.hub.load(
            self.REPO,
            "generator",
            pretrained=self.config.pretrained,
            trust_repo=True,
        )
        self.model.to(self._device).eval()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "mps" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def apply(self, image: Image.Image) -> Image.Image:
        cfg = self.config
        img = image.convert("RGB")

        # Pre-processing
        if cfg.pre_brightness != 1.0:
            img = ImageEnhance.Brightness(img).enhance(cfg.pre_brightness)
        if cfg.pre_saturation != 1.0:
            img = ImageEnhance.Color(img).enhance(cfg.pre_saturation)

        # bilateral로 원본 색상 평탄화 (카툰 베이스)
        flat = np.array(img)
        for _ in range(cfg.bilateral_passes):
            flat = cv2.bilateralFilter(flat, 9, 75, 75)

        # PIL → tensor [-1, 1]
        arr = flat.astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self._device)

        with torch.inference_mode():
            out = self.model(tensor)

        # tensor [-1,1] → numpy
        out_arr = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_arr = np.clip((out_arr + 1.0) * 127.5, 0, 255).astype(np.uint8)

        # AnimeGAN2 스타일 + bilateral 평탄화 색상 블렌드
        blended = (out_arr * cfg.anime_blend + flat * (1.0 - cfg.anime_blend)).astype(np.uint8)

        # 블렌드 후 추가 bilateral — 카툰 색 영역 완전 평탄화
        for _ in range(cfg.post_bilateral_passes):
            blended = cv2.bilateralFilter(blended, 9, 75, 75)

        result = Image.fromarray(blended)

        # Post-processing
        if cfg.post_saturation != 1.0:
            result = ImageEnhance.Color(result).enhance(cfg.post_saturation)
        if cfg.post_contrast != 1.0:
            result = ImageEnhance.Contrast(result).enhance(cfg.post_contrast)

        return result
