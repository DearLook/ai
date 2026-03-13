"""
ControlNet + SD 1.5 + Pixel Art LoRA 스타일라이저.

파이프라인:
  1. SoftEdge 전처리기로 원본 사진의 윤곽선(포즈·악세사리) 추출
  2. ControlNet SD 1.5 img2img로 픽셀아트 캐릭터 생성
  3. Pixel Art LoRA(zenafey/pixel_f2)로 스프라이트 스타일 적용
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")


@dataclass
class ControlNetConfig:
    sd15_path: str = "models/base/anything-v3"
    sd15_model_id: str = "Linaqruf/anything-v3-1"
    controlnet_path: str = "models/controlnet/softedge"
    controlnet_model_id: str = "lllyasviel/control_v11p_sd15_softedge"
    lora_path: str = "models/character_lora/pixel_f2.safetensors"
    prompt: str = (
        "pixel art character sprite, full body, cute anime face, "
        "highly detailed face, sharp clear eyes, nose, mouth, distinct facial features, "
        "clean pixel outlines, flat colors, cel shading, indie game sprite, "
        "white background, standing pose, pixel art style, warm natural colors"
    )
    negative_prompt: str = (
        "photorealistic, blurry face, smeared face, noisy, realistic skin, text, logo, "
        "watermark, extra limbs, deformed, bad anatomy, missing face, "
        "faceless, no face, blob face, melted face, "
        "blue tones, purple, oversaturated, neon colors"
    )
    num_inference_steps: int = 35
    guidance_scale: float = 8.5
    controlnet_conditioning_scale: float = 1.0
    strength: float = 0.75
    max_size: int = 768
    lora_scale: float = 0.6
    seed: int = 1234
    device: str = "cpu"
    torch_dtype: torch.dtype = field(default=torch.float32)


class ControlNetStylizer:
    """SoftEdge ControlNet + SD1.5 + pixel_f2 LoRA로 픽셀아트 캐릭터 변환."""

    def __init__(self, config: ControlNetConfig) -> None:
        from controlnet_aux import PidiNetDetector
        from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

        self.config = config
        self._device = self._resolve_device(config.device)

        sd15_path = config.sd15_path if os.path.isdir(config.sd15_path) else config.sd15_model_id
        cn_path = config.controlnet_path if os.path.isdir(config.controlnet_path) else config.controlnet_model_id

        controlnet = ControlNetModel.from_pretrained(cn_path, torch_dtype=config.torch_dtype)
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            sd15_path,
            controlnet=controlnet,
            torch_dtype=config.torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        if os.path.exists(config.lora_path):
            self.pipe.load_lora_weights(config.lora_path)
            self.pipe.fuse_lora(lora_scale=config.lora_scale)

        self.pipe.to(self._device)
        self.pipe.enable_attention_slicing()

        self.preprocessor = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
        self.preprocessor.to(self._device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "mps" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resize(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        scale = min(1.0, self.config.max_size / max(w, h))
        nw = int(w * scale)
        nh = int(h * scale)
        # SD 1.5 는 64의 배수 필요
        nw = max(64, (nw // 64) * 64)
        nh = max(64, (nh // 64) * 64)
        return image.resize((nw, nh), Image.BILINEAR)

    def apply(self, image: Image.Image) -> Image.Image:
        img = self._resize(image.convert("RGB"))

        # SoftEdge 윤곽선 추출 — 반드시 img와 동일한 크기로
        control_image = self.preprocessor(img, detect_resolution=min(img.size), image_resolution=min(img.size))
        control_image = control_image.resize(img.size, Image.BILINEAR)

        generator_device = "cpu" if self._device == "mps" else self._device
        generator = torch.Generator(device=generator_device).manual_seed(self.config.seed)

        with torch.inference_mode():
            result = self.pipe(
                prompt=self.config.prompt,
                negative_prompt=self.config.negative_prompt,
                image=img,
                control_image=control_image,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                controlnet_conditioning_scale=self.config.controlnet_conditioning_scale,
                strength=self.config.strength,
                generator=generator,
            ).images[0]

        return result.resize(image.size, Image.BILINEAR)
