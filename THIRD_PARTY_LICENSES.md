# Third-Party Models and Licenses

Last updated: 2026-03-11

## Purpose

This document tracks model provenance, license, version, and commercial-use status for deployment review.

## Python Library Dependencies

### PyTorch & torchvision
- **Packages**: `torch`, `torchvision`
- **License**: BSD 3-Clause
- **Source**: https://github.com/pytorch/pytorch

### Diffusers (Hugging Face)
- **Package**: `diffusers`
- **License**: Apache 2.0
- **Source**: https://github.com/huggingface/diffusers

### Transformers (Hugging Face)
- **Package**: `transformers`
- **License**: Apache 2.0
- **Source**: https://github.com/huggingface/transformers

### Accelerate (Hugging Face)
- **Package**: `accelerate`
- **License**: Apache 2.0
- **Source**: https://github.com/huggingface/accelerate

### PEFT (Hugging Face)
- **Package**: `peft`
- **License**: Apache 2.0
- **Source**: https://github.com/huggingface/peft

### Safetensors (Hugging Face)
- **Package**: `safetensors`
- **License**: Apache 2.0
- **Source**: https://github.com/huggingface/safetensors

### Hugging Face Hub
- **Package**: `huggingface_hub`
- **License**: Apache 2.0
- **Source**: https://github.com/huggingface/huggingface_hub

### Pillow
- **Package**: `pillow`
- **License**: HPND (Historical Permission Notice and Disclaimer)
- **Source**: https://github.com/python-pillow/Pillow
- **Used in**: `pipeline.py` — RGB 변환, paste() 합성

### OpenCV
- **Package**: `opencv-python` (`cv2`)
- **License**: Apache 2.0
- **Source**: https://github.com/opencv/opencv
- **Used in**: `anime_stylizer.py`, `cartoon_stylizer.py` — bilateralFilter

### NumPy
- **Package**: `numpy`
- **License**: BSD 3-Clause
- **Source**: https://github.com/numpy/numpy

### SciPy
- **Package**: `scipy`
- **License**: BSD 3-Clause
- **Source**: https://github.com/scipy/scipy
- **Used in**: `pipeline.py` — binary_dilation (마스크 팽창)

### FastAPI
- **Package**: `fastapi`
- **License**: MIT
- **Source**: https://github.com/tiangolo/fastapi

### Uvicorn
- **Package**: `uvicorn`
- **License**: BSD 3-Clause
- **Source**: https://github.com/encode/uvicorn

### python-multipart
- **Package**: `python-multipart`
- **License**: Apache 2.0
- **Source**: https://github.com/andrew-d/python-multipart

### pydantic-settings
- **Package**: `pydantic-settings`
- **License**: MIT
- **Source**: https://github.com/pydantic/pydantic-settings

---

## Model Inventory

### 1) Stable Diffusion XL Base 1.0

- Component: `SDXL base (img2img backbone)`
- Local path: `models/base/sdxl-base-1.0`
- Upstream: `stabilityai/stable-diffusion-xl-base-1.0`
- Version: `1.0`
- License file: `models/base/sdxl-base-1.0/LICENSE.md`
- Commercial-use status: `Allowed with Stability AI terms`
- Notes:
  - Follow Stability AI license and usage policy.
  - Keep policy-compliance checks in product/legal review.

### 2) AnimeGAN2 (bryandlee/animegan2-pytorch)

- Component: `AnimeGAN2 generator`
- Load method: `torch.hub.load("bryandlee/animegan2-pytorch:main", "generator")`
- Upstream: https://github.com/bryandlee/animegan2-pytorch
- License: MIT
- Commercial-use status: Allowed (MIT)
- Notes:
  - `face_paint_512_v2` pretrained weight 사용 (기본값)
  - `src/models/anime_stylizer.py`에서 로드

### 3) DeepLabV3 + MobileNetV3 Large (torchvision)

- Component: `Person segmentation backbone`
- Load method: `torchvision.models.segmentation.deeplabv3_mobilenet_v3_large`
- Upstream: https://github.com/pytorch/vision
- License: BSD 3-Clause
- Weights: COCO-pretrained (`DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT`)
- Commercial-use status: Allowed (BSD 3-Clause)
- Notes:
  - `src/models/segmentation.py`에서 로드

### 4) PixelArtRedmond LoRA (artificialguybr/PixelArtRedmond)

- Component: `Pixel Art LoRA (PixelArtRedmond-Lite64)`
- Local path: `models/character_lora/PixelArtRedmond-Lite64.safetensors`
- Upstream: `artificialguybr/PixelArtRedmond`
- Upstream URL: https://huggingface.co/artificialguybr/PixelArtRedmond
- Version: `Lite64`
- License: `creativeml-openrail-m`
- License URL: https://huggingface.co/spaces/CompVis/stable-diffusion-license
- Commercial-use status: `Allowed (OpenRAIL-M permits commercial use with responsible use guidelines)`
- Trigger word: `Pixel Art, PixArFK`
- SHA256: `(run scripts/download_lora.py to obtain)`
- Notes:
  - 픽셀아트 전용 대규모 데이터셋 학습, SDXL 1.0 호환
  - 다운로드: `python scripts/download_lora.py`
  - OpenRAIL-M 금지 용도: 불법 콘텐츠, 허위 정보, 개인 정보 침해 등 — 서비스 이용 약관에 반영 필요

## Runtime License Guard

- Setting: `LICENSE_STRICT_MODE` (default: `true`)
- Behavior:
  - If strict mode is enabled and LoRA metadata/approval/commercial checks fail, API runs `base-only` mode (LoRA off).
  - In base-only mode, quality is typically lower than dedicated pixel-character LoRA output.
  - API headers expose runtime state:
    - `X-Lora-Enabled`
    - `X-Lora-Block-Reason`
    - `X-Lora-Approved`
    - `X-Lora-Sha256`
    - `X-License-Strict-Mode`
    - `X-Lora-Commercial-Allowed`

## Deployment Checklist

- [ ] Confirm every model source URL is recorded.
- [ ] Confirm exact version/hash for each model file.
- [ ] Confirm license text archived in repo or legal docs.
- [ ] Confirm commercial-use approval for deployed models.
- [ ] Keep this file updated with every model change.
