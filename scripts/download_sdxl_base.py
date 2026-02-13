"""Download SDXL base model into models/base/sdxl-base-1.0."""

import os
from huggingface_hub import snapshot_download

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGET_DIR = os.path.join(BASE_DIR, "models", "base", "sdxl-base-1.0")
MODEL_ID = os.environ.get("PIXELART_DIFFUSER_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
HF_TOKEN = os.environ.get("HF_TOKEN")


def main() -> None:
    os.makedirs(TARGET_DIR, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=TARGET_DIR,
        token=HF_TOKEN,
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "text_encoder_2/config.json",
            "text_encoder_2/model.safetensors",
            "tokenizer/*",
            "tokenizer_2/*",
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "LICENSE.md",
            "README.md",
        ],
    )
    print(f"Downloaded {MODEL_ID} to {TARGET_DIR}")


if __name__ == "__main__":
    main()
