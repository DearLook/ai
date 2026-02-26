# Character LoRA Training Scaffold

This project now assumes `character_v1` style should come from a dedicated person-character LoRA.

## 1) Prepare dataset

```bash
python scripts/prepare_character_dataset.py \
  --input-dir data/train_raw \
  --output-dir data/train_character \
  --token dearlook_char
```

Output format:
- `00001.png` (transparent cutout crop)
- `00001.txt` (caption)

## 2) Train LoRA (example with `sd-scripts`)

Use your preferred trainer. Example command shape:

```bash
accelerate launch train_network.py \
  --pretrained_model_name_or_path=models/base/sdxl-base-1.0 \
  --train_data_dir=data/train_character \
  --resolution=1024,1024 \
  --output_dir=models/character_lora \
  --output_name=character_v1 \
  --network_module=networks.lora \
  --network_dim=32 \
  --network_alpha=32 \
  --train_batch_size=1 \
  --max_train_steps=3000 \
  --learning_rate=1e-4 \
  --mixed_precision=bf16 \
  --save_model_as=safetensors
```

## 3) Connect trained LoRA

Copy resulting file:
- `models/character_lora/character_v1.safetensors`

Then set `.env`:
- `PIXELART_LORA_PATH=models/character_lora/character_v1.safetensors`
- `PIXELART_LORA_SOURCE=<official source or internal training record>`
- `PIXELART_LORA_VERSION=character_v1`
- `PIXELART_LORA_LICENSE=<license>`
- `PIXELART_LORA_LICENSE_URL=<license url>`
- `PIXELART_LORA_UPSTREAM_URL=<source url>`
- `PIXELART_LORA_SHA256=<sha256>`
- `PIXELART_LORA_APPROVED=true`
- `PIXELART_LORA_COMMERCIAL_ALLOWED=true`

With strict mode on, missing metadata will automatically disable LoRA.
