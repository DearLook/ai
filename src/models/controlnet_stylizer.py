from __future__ import annotations

import os
from dataclasses import dataclass, field
import torch
from PIL import Image, ImageEnhance
from controlnet_aux import PidiNetDetector
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")


@dataclass
class ControlNetConfig:
    sd15_path: str = "models/base/dreamshaper_8.safetensors"
    sd15_model_id: str = "Lykon/dreamshaper-8"
    controlnet_path: str = "models/controlnet/softedge"
    controlnet_model_id: str = "lllyasviel/control_v11p_sd15_softedge"
    lora_path: str = "models/character_lora/pixelartV3.safetensors"
    prompt: str = (
        "pixel, pixel art character sprite, full body, cute anime face, "
        "highly detailed face, sharp clear eyes, nose, mouth, distinct facial features, "
        "clean pixel outlines, flat colors, cel shading, indie game sprite, "
        "white background, standing pose, pixel art style"
    )
    negative_prompt: str = (
        "photorealistic, blurry face, smeared face, noisy, realistic skin, text, logo, "
        "watermark, extra limbs, deformed, bad anatomy, missing face, "
        "faceless, no face, blob face, melted face, "
        "wrong colors, color bleeding, extra colors, unnatural colors"
    )
    num_inference_steps: int = 36
    guidance_scale: float = 6.0
    controlnet_conditioning_scale: float = 0.75
    strength: float = 0.65
    max_size: int = 768
    lora_scale: float = 0.9
    seed: int = 1234
    device: str = "cpu"
    torch_dtype: torch.dtype = field(default=torch.float32)
    brightness_boost: float = 1.3
    color_boost: float = 1.3


class ControlNetStylizer:

    def __init__(self, config: ControlNetConfig) -> None:

        self.config = config
        self._device = self._resolve_device(config.device)

        cn_path = config.controlnet_path if os.path.isdir(config.controlnet_path) else config.controlnet_model_id
        controlnet = ControlNetModel.from_pretrained(cn_path, torch_dtype=config.torch_dtype)

        sd15_path = config.sd15_path
        if os.path.isfile(sd15_path) and sd15_path.endswith(".safetensors"):
            self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
                sd15_path,
                controlnet=controlnet,
                torch_dtype=config.torch_dtype,
                safety_checker=None,
            )
        else:
            hub_id = sd15_path if os.path.isdir(sd15_path) else config.sd15_model_id
            self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                hub_id,
                controlnet=controlnet,
                torch_dtype=config.torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
        if not os.path.exists(config.lora_path):
            raise FileNotFoundError("Required LoRA weights are missing")
        self._apply_kohya_lora(self.pipe, config.lora_path, config.lora_scale)
        self._lora_scale = config.lora_scale

        self.pipe.to(self._device)
        self.pipe.enable_attention_slicing()

        self.preprocessor = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
        self.preprocessor.to(self._device)

    @staticmethod
    def _apply_kohya_lora(pipe, lora_path: str, scale: float) -> None:
        from safetensors.torch import load_file

        state_dict = load_file(lora_path)

        # kohya key → diffusers module path 변환
        def kohya_key_to_module(key: str):
            key = key.replace("lora_unet_", "").replace("lora_te_", "te.")
            key = key.replace("_", ".", 1) if key.startswith("te.") else key
            parts = key.split(".")
            # lora_down / lora_up / alpha 분리
            suffix = None
            for marker in ("lora_down", "lora_up", "alpha"):
                if marker in parts:
                    idx = parts.index(marker)
                    suffix = ".".join(parts[idx:])
                    parts = parts[:idx]
                    break
            module_path = "_".join(parts)
            return module_path, suffix

        # 레이어별 down/up/alpha 수집
        lora_layers: dict[str, dict] = {}
        for full_key, tensor in state_dict.items():
            if "lora_unet" not in full_key:
                continue
            base = full_key.replace("lora_unet_", "")
            if ".lora_down.weight" in full_key:
                layer = base.replace(".lora_down.weight", "")
                lora_layers.setdefault(layer, {})["down"] = tensor
            elif ".lora_up.weight" in full_key:
                layer = base.replace(".lora_up.weight", "")
                lora_layers.setdefault(layer, {})["up"] = tensor
            elif ".alpha" in full_key:
                layer = base.replace(".alpha", "")
                lora_layers.setdefault(layer, {})["alpha"] = tensor

        def get_module(model, path: str):
            parts = path.split("_")
            # 언더스코어 구분 경로를 순차 탐색으로 해결
            node = model
            i = 0
            segments = path.split("_")
            current = ""
            for seg in segments:
                current = current + ("_" if current else "") + seg
                if hasattr(node, current):
                    node = getattr(node, current)
                    current = ""
            return node if not current else None

        unet = pipe.unet
        applied = 0
        for layer_path, weights in lora_layers.items():
            if "down" not in weights or "up" not in weights:
                continue
            down = weights["down"].float()
            up = weights["up"].float()
            alpha = weights.get("alpha", torch.tensor(down.shape[0])).float()
            rank = down.shape[0]
            # conv(4D) vs linear(2D) 처리
            if down.dim() == 4:
                delta = (up.reshape(up.shape[0], -1) @ down.reshape(down.shape[0], -1)).reshape(up.shape[0], down.shape[1], *down.shape[2:]) * (alpha / rank) * scale
            else:
                delta = (up @ down) * (alpha / rank) * scale

            # 경로를 점 구분자로 변환해 모듈 탐색
            path_parts = layer_path.split("_")
            node = unet
            consumed = []
            remaining = list(path_parts)
            while remaining:
                found = False
                for length in range(len(remaining), 0, -1):
                    candidate = "_".join(remaining[:length])
                    if hasattr(node, candidate):
                        node = getattr(node, candidate)
                        consumed.extend(remaining[:length])
                        remaining = remaining[length:]
                        found = True
                        break
                if not found:
                    node = None
                    break

            if node is None or not hasattr(node, "weight"):
                continue

            with torch.no_grad():
                if node.weight.shape == delta.shape:
                    node.weight.add_(delta.to(node.weight.dtype))
                    applied += 1
                elif delta.squeeze().shape == node.weight.squeeze().shape:
                    node.weight.add_(delta.squeeze().reshape(node.weight.shape).to(node.weight.dtype))
                    applied += 1

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
        nw = max(64, (nw // 64) * 64)
        nh = max(64, (nh // 64) * 64)
        return image.resize((nw, nh), Image.BILINEAR)

    def apply(self, image: Image.Image) -> Image.Image:
        img = self._resize(image.convert("RGB"))
        # 입력 전 밝기/채도 보정
        img = ImageEnhance.Brightness(img).enhance(self.config.brightness_boost)
        img = ImageEnhance.Color(img).enhance(self.config.color_boost)

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
