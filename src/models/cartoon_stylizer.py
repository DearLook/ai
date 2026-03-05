"""
CartoonStylizer — OpenCV bilateralFilter 기반 일러스트/카툰 변환.
별도 학습/모델 다운로드 없이 로컬에서 무료로 동작합니다.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class CartoonConfig:
    # 전처리: 밝기/채도 보정 (어두운 사진 보정)
    brightness_boost: float = 1.4   # 밝기 강화
    saturation_boost: float = 1.8   # 채도 강화 (일러스트는 채도가 높음)

    # bilateral filter: 색상 평탄화
    bilateral_d: int = 9
    bilateral_sigma_color: int = 300
    bilateral_sigma_space: int = 300
    bilateral_passes: int = 7       # 많을수록 평탄화 → 일러스트 느낌


class CartoonStylizer:
    """사진을 일러스트/카툰 스타일로 변환."""

    def __init__(self, config: CartoonConfig | None = None) -> None:
        self.config = config or CartoonConfig()

    def apply(self, image: Image.Image) -> Image.Image:
        cfg = self.config
        img = np.array(image.convert("RGB"))

        # 1. 밝기/채도 선(先) 보정 — 어두운 옷 사진이 검게 뭉개지는 문제 방지
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * cfg.brightness_boost, 0, 255)  # V (밝기)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * cfg.saturation_boost, 0, 255)  # S (채도)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # 2. 색상 평탄화 (bilateral filter 반복)
        smooth = img
        for _ in range(cfg.bilateral_passes):
            smooth = cv2.bilateralFilter(
                smooth,
                cfg.bilateral_d,
                cfg.bilateral_sigma_color,
                cfg.bilateral_sigma_space,
            )

        return Image.fromarray(smooth)
