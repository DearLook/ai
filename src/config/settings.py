from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PIXELART_DEVICE: str = "cpu"
    PIXELART_DTYPE: str = "fp32"

    PIXELART_DIFFUSER_MODEL_ID: str = "stabilityai/stable-diffusion-xl-base-1.0"
    PIXELART_LORA_PATH: str = "models/pixelart/pixel-art-xl-v1.1.safetensors"
    PIXELART_LORA_SCALE: float = 0.6
    PIXELART_PROMPT: str = (
        "pixel art character, clean outlines, flat shading, 16-bit sprite, simple shapes, limited palette"
    )
    PIXELART_NEGATIVE_PROMPT: str = (
        "glitch, neon, cyberpunk, text, logo, letters, symbols, high contrast, photorealistic, blurry, noise, artifacts"
    )
    PIXELART_STEPS: int = 20
    PIXELART_STRENGTH: float = 0.4
    PIXELART_GUIDANCE: float = 4.5
    PIXELART_MAX_SIZE: int = 512
    PIXELART_SEED: int = 1234
    PIXELART_POST_GRID: int = 128
    PIXELART_POST_PALETTE: int = 32

    PIXELART_MODEL_PATH: str = "models/pixelart/pixelart.ts"
    PIXELART_MODEL_INPUT: int | None = None
    PIXELART_MODEL_NORMALIZE: bool = False


settings = Settings()
