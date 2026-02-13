import io, os, sys, time
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.segmentation import PersonSegmenter
from src.models.pixelart_diffusion import (
    PixelArtDiffusionStylizer,
    default_diffusion_config,
)
from src.config.settings import settings
from src.services.pipeline import (
    PixelArtConfig,
    composite,
    pixel_art_person,
    pixel_art_person_diffusion,
    pixelate,
    resize_mask,
)

app = FastAPI(title="DearLook AI")

_segmenter: Optional[PersonSegmenter] = None
_diffusion_stylizer: Optional[PixelArtDiffusionStylizer] = None


def get_segmenter() -> PersonSegmenter:
    global _segmenter
    if _segmenter is None:
        _segmenter = PersonSegmenter()
    return _segmenter

def get_diffusion_stylizer() -> PixelArtDiffusionStylizer:
    global _diffusion_stylizer
    if _diffusion_stylizer is None:
        _diffusion_stylizer = PixelArtDiffusionStylizer(default_diffusion_config())
    return _diffusion_stylizer


@app.post("/pixelate")
async def pixelate_person(
    file: UploadFile = File(...),
    style: str = Form("pixelart"),
    block: int = Form(12),
    long_edge: int = Form(160),
    palette: int = Form(48),
    dither: bool = Form(False),
    outline: bool = Form(False),
    edge_threshold: float = Form(0.12),
    smooth: int = Form(3),
    color: float = Form(1.15),
    contrast: float = Form(1.1),
    background: str = Form("transparent"),
):
    start = time.time()
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    segmenter = get_segmenter()
    mask = segmenter.predict_mask(image)
    mask = resize_mask(mask, image.size)
    style = style.lower().strip()
    mode_used = "mosaic"

    if style == "mosaic":
        pix = pixelate(image, block)
        out = composite(image, pix, mask)
    elif style in ("pixelart", "pixel-art"):
        mode_used = "pixelart"
        pix = pixel_art_person(
            image,
            mask,
            PixelArtConfig(
                target_long_edge=long_edge,
                palette_size=palette,
                dither=dither,
                outline=outline,
                edge_threshold=edge_threshold,
                pre_smooth=smooth,
                color_boost=color,
                contrast_boost=contrast,
            ),
        )
        if background == "original":
            base = image.convert("RGBA")
            base.paste(pix, (0, 0), pix)
            out = base
        else:
            out = pix
    
    elif style in ("model", "pixelart-model", "stylized"):
        mode_used = "model"
        try:
            stylizer = get_diffusion_stylizer()
            pix = pixel_art_person_diffusion(image, mask, stylizer)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"diffusion_failed: {exc}") from exc
        if background == "original":
            base = image.convert("RGBA")
            base.paste(pix, (0, 0), pix)
            out = base
        else:
            out = pix
    
    else:
        raise HTTPException(status_code=400, detail="style must be one of: mosaic, pixelart, model")

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    elapsed_ms = int((time.time() - start) * 1000)
    headers = {
        "X-Style-Input": style,
        "X-Mode-Used": mode_used,
        "X-Device": settings.PIXELART_DEVICE,
        "X-Dtype": settings.PIXELART_DTYPE,
        "X-Model-Id": settings.PIXELART_DIFFUSER_MODEL_ID,
        "X-Lora-Path": settings.PIXELART_LORA_PATH,
        "X-Lora-Scale": str(getattr(settings, "PIXELART_LORA_SCALE", 1.0)),
        "X-Elapsed-Ms": str(elapsed_ms),
    }
    
    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)
