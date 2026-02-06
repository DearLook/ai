"""Pixel-art stylization via Stable Diffusion img2img + LoRA."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
# Prevent transformers from importing TF/JAX stacks in this CPU-only setup
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image


@dataclass
class PixelArtDiffusionConfig:
    model_id: str
    lora_path: str
    prompt: str = "pixel art, 16-bit, clean outlines, simple shading"
    negative_prompt: str = "blurry, photorealistic, noisy, artifacts"
    num_inference_steps: int = 20
    strength: float = 0.6
    guidance_scale: float = 5.0
    max_size: int = 512
    torch_dtype: torch.dtype = torch.float32
    device: str = "cpu"
    lora_scale: float = 1.0
    seed: int = 1234
    post_grid: int = 128
    post_palette: int = 32
    lora_repo: str | None = None
    lora_weight_name: str | None = None


def default_diffusion_config() -> PixelArtDiffusionConfig:
    from src.config.settings import settings

    model_id = settings.PIXELART_DIFFUSER_MODEL_ID
    lora_path = settings.PIXELART_LORA_PATH
    lora_repo = settings.PIXELART_LORA_REPO
    lora_weight_name = settings.PIXELART_LORA_WEIGHT_NAME
    prompt = settings.PIXELART_PROMPT
    negative_prompt = settings.PIXELART_NEGATIVE_PROMPT
    steps = settings.PIXELART_STEPS
    strength = settings.PIXELART_STRENGTH
    guidance = settings.PIXELART_GUIDANCE
    max_size = settings.PIXELART_MAX_SIZE
    lora_scale = settings.PIXELART_LORA_SCALE
    seed = settings.PIXELART_SEED
    post_grid = settings.PIXELART_POST_GRID
    post_palette = settings.PIXELART_POST_PALETTE

    device = settings.PIXELART_DEVICE.lower()
    dtype_env = settings.PIXELART_DTYPE.lower()
    if dtype_env == "fp16":
        torch_dtype = torch.float16
    elif dtype_env == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    return PixelArtDiffusionConfig(
        model_id=model_id,
        lora_path=lora_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        strength=strength,
        guidance_scale=guidance,
        max_size=max_size,
        torch_dtype=torch_dtype,
        device=device,
        lora_scale=lora_scale,
        seed=seed,
        post_grid=post_grid,
        post_palette=post_palette,
        lora_repo=lora_repo or None,
        lora_weight_name=lora_weight_name or None,
    )


class PixelArtDiffusionStylizer:
    def __init__(self, config: PixelArtDiffusionConfig):
        if not os.path.exists(config.lora_path) and not config.lora_repo:
            raise FileNotFoundError(f"pixel-art LoRA not found: {config.lora_path}")
        self.config = config
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            config.model_id, torch_dtype=config.torch_dtype
        )
        if os.path.exists(config.lora_path):
            self.pipe.load_lora_weights(config.lora_path)
        elif config.lora_repo:
            if not config.lora_weight_name:
                raise FileNotFoundError(
                    "pixel-art LoRA repo set but weight name missing: set PIXELART_LORA_WEIGHT_NAME"
                )
            self.pipe.load_lora_weights(config.lora_repo, weight_name=config.lora_weight_name)
        else:
            raise FileNotFoundError(f"pixel-art LoRA not found: {config.lora_path}")
        self.pipe.fuse_lora(lora_scale=config.lora_scale)
        device = "mps" if (config.device == "mps" and torch.backends.mps.is_available()) else "cpu"
        self.pipe.to(device)
        # Memory helpers
        self.pipe.enable_attention_slicing()
        self.pipe.vae.enable_slicing()
        # Avoid NaNs on MPS by keeping VAE in fp32
        self.pipe.vae.to(torch.float32)

    def _resize_for_pipe(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        scale = min(1.0, float(self.config.max_size) / float(max(w, h)))
        if scale < 1.0:
            nw = max(1, int(w * scale))
            nh = max(1, int(h * scale))
            return image.resize((nw, nh), Image.BILINEAR)
        return image

    def apply(self, image: Image.Image) -> Image.Image:
        img = image.convert("RGB")
        img = self._resize_for_pipe(img)
        generator = torch.Generator("cpu").manual_seed(self.config.seed)
        with torch.inference_mode():
            out = self.pipe(
                prompt=self.config.prompt,
                negative_prompt=self.config.negative_prompt,
                image=img,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                strength=self.config.strength,
                generator=generator,
            ).images[0]
        # Match original size (pixel-ish)
        out = self._post_pixelize(out, image.size)
        return out

    def _post_pixelize(self, image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
        w, h = target_size
        # Downscale to grid, quantize palette, then upscale to crisp pixels
        if self.config.post_grid and self.config.post_grid > 1:
            if w >= h:
                new_w = self.config.post_grid
                new_h = max(1, int(h * (self.config.post_grid / w)))
            else:
                new_h = self.config.post_grid
                new_w = max(1, int(w * (self.config.post_grid / h)))
            image = image.resize((new_w, new_h), Image.BOX)
        if self.config.post_palette and self.config.post_palette > 2:
            image = image.convert("P", palette=Image.ADAPTIVE, colors=self.config.post_palette).convert("RGB")
        return image.resize((w, h), Image.NEAREST)
