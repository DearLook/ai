from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LICENSE_STRICT_MODE: bool = False
    CHARACTER_STYLE_NAME: str = "character_v1"
    ENABLE_LEGACY_STYLES: bool = False

    PIXELART_DEVICE: str = "cpu"
    PIXELART_DTYPE: str = "fp32"
    PIXELART_DIFFUSER_MODEL_ID: str = "stabilityai/stable-diffusion-xl-base-1.0"
    PIXELART_DIFFUSER_PATH: str | None = "models/base/sdxl-base-1.0"
    PIXELART_USE_LORA: bool = True
    PIXELART_LORA_PATH: str = "models/character_lora/character_v1.safetensors"
    PIXELART_LORA_SCALE: float = 1.15
    PIXELART_PROMPT: str = (
        "dearlook_char, full-body pixel-art character illustration, cute anime sprite, "
        "clear face, clean outline, cel shading, game character design, simplified clothing folds"
    )
    PIXELART_NEGATIVE_PROMPT: str = (
        "photorealistic, realistic skin, camera photo, blurry, noise, artifacts, "
        "distorted body, malformed face, neon glitch, text, logo, armor, military, "
        "over-detailed texture"
    )
    PIXELART_STEPS: int = 36
    PIXELART_STRENGTH: float = 0.78
    PIXELART_GUIDANCE: float = 6.5
    PIXELART_MAX_SIZE: int = 512
    PIXELART_SEED: int = 1234
    PIXELART_POST_GRID: int = 96
    PIXELART_POST_PALETTE: int = 24
    PIXELART_POST_SATURATION: float = 1.12
    PIXELART_POST_CONTRAST: float = 1.08
    PIXELART_POST_BRIGHTNESS: float = 1.08
    PIXELART_PRE_BRIGHTNESS: float = 1.0
    PIXELART_PRE_CONTRAST: float = 1.0
    PIXELART_PRE_SATURATION: float = 1.0
    PIXELART_BASE_ONLY_PROMPT: str = (
        "full-body pixel-art character sprite, clean silhouette, readable face, "
        "clean color blocks, simple cel-shading, game sprite style, white background"
    )

    PIXELART_LORA_SOURCE: str = ""
    PIXELART_LORA_VERSION: str = ""
    PIXELART_LORA_LICENSE: str = "UNSPECIFIED"
    PIXELART_LORA_LICENSE_URL: str = ""
    PIXELART_LORA_UPSTREAM_URL: str = ""
    PIXELART_LORA_SHA256: str = ""
    PIXELART_LORA_APPROVED: bool = False
    PIXELART_LORA_COMMERCIAL_ALLOWED: bool = False
    PIXELART_REQUIRE_LORA_FOR_MODEL: bool = True

settings = Settings()

def get_settings() -> Settings:
    return settings