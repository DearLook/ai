# Third-Party Models and Licenses

Last updated: 2026-02-24

## Purpose
This document tracks model provenance, license, version, and commercial-use status for deployment review.

## Model Inventory

### 1) Base diffusion model
- Component: `SDXL base (img2img backbone)`
- Local path: `models/base/sdxl-base-1.0`
- Upstream: `stabilityai/stable-diffusion-xl-base-1.0`
- Version: `1.0`
- License file: `models/base/sdxl-base-1.0/LICENSE.md`
- Commercial-use status: `Allowed with Stability AI terms`
- Notes:
  - Follow Stability AI license and usage policy.
  - Keep policy-compliance checks in product/legal review.

### 2) Pixel style LoRA (optional)
- Component: `Pixel-art LoRA`
- Local path: `models/pixelart/pixel-art-xl-v1.1.safetensors`
- Upstream/source: `CivitAI model 120096 (Pixel Art XL), creator: NeriJS`
- Version: `v1.1 (CivitAI modelVersionId=135931)`
- License: `CivitAI creator-defined license (commercial status unverified)`
- License URL: `https://civitai.com/models/120096?modelVersionId=135931`
- Commercial-use status: `Not allowed in strict mode`
- Approval status: `Not approved for production`
- SHA256: `bbf3d8defbfb3fb71331545225c0cf50c74a748d2525f7c19ebb8f74445de274`
- Notes:
  - This LoRA is disabled automatically when strict-mode metadata conditions are not fully satisfied.
  - Enable LoRA in production only after license/provenance is verified and recorded.
  - Internal safetensors metadata indicates a training output name (`pixelbuildings128-v2`) and does not include an explicit commercial redistribution license in-file.

## Runtime License Guard
- Setting: `LICENSE_STRICT_MODE` (default: `true`)
- Behavior:
  - If strict mode is enabled and LoRA metadata/approval/commercial checks fail, API runs `base-only` mode (LoRA off).
  - In base-only mode, quality is typically lower than dedicated pixel-character LoRA output.
  - API headers expose runtime state:
    - `X-Lora-Enabled`
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
