from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PIXELART_DEVICE: str = "cpu"
    PIXELART_DTYPE: str = "fp32"
    PIXELART_DIFFUSER_MODEL_ID: str = "stabilityai/stable-diffusion-xl-base-1.0"
    PIXELART_DIFFUSER_PATH: str | None = "models/base/sdxl-base-1.0"
    PIXELART_USE_LORA: bool = True
    PIXELART_LORA_PATH: str = "models/pixelart/pixel-art-xl-v1.1.safetensors"
    PIXELART_LORA_SCALE: float = 0.38
    PIXELART_PROMPT: str = (
        "full-body pixel art character sprite, cute style, clean line art, clear facial features, natural colors, simple shading, plain outfit details"
    )
    PIXELART_NEGATIVE_PROMPT: str = (
        "glitch, neon, cyberpunk, text, logo, letters, symbols, photorealistic, blurry, noise, artifacts, dark, underexposed, muddy colors, low saturation, distorted body, malformed face, armor, military uniform, red-blue hair"
    )
    PIXELART_STEPS: int = 24
    PIXELART_STRENGTH: float = 0.34
    PIXELART_GUIDANCE: float = 5.6
    PIXELART_MAX_SIZE: int = 512
    PIXELART_SEED: int = 1234
    PIXELART_POST_GRID: int = 224
    PIXELART_POST_PALETTE: int = 72
    PIXELART_POST_SATURATION: float = 1.08
    PIXELART_POST_CONTRAST: float = 1.04
    PIXELART_POST_BRIGHTNESS: float = 1.03

settings = Settings()
