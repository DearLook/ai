"""FastAPI service for person-only pixelation."""

import io
import os
import sys
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

# Allow running as a module from repo root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.src.models.segmentation import PersonSegmenter  # noqa: E402
from ai.src.services.pipeline import (  # noqa: E402
    PixelArtConfig,
    PixelateConfig,
    apply_alpha,
    composite,
    pixel_art_person,
    pixel_art,
    pixelate,
    resize_mask,
)

app = FastAPI(title="DearLook AI", version="0.1.0")

_segmenter: Optional[PersonSegmenter] = None


def get_segmenter() -> PersonSegmenter:
    global _segmenter
    if _segmenter is None:
        _segmenter = PersonSegmenter()
    return _segmenter


@app.post("/pixelate")
async def pixelate_person(
    file: UploadFile = File(...),
    style: str = "pixelart",
    block: int = 12,
    long_edge: int = 160,
    palette: int = 48,
    dither: bool = False,
    outline: bool = False,
    edge_threshold: float = 0.12,
    smooth: int = 3,
    color: float = 1.15,
    contrast: float = 1.1,
    background: str = "transparent",
):
    # Read image
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")

    # Predict person mask
    segmenter = get_segmenter()
    mask = segmenter.predict_mask(image)
    mask = resize_mask(mask, image.size)

    style = style.lower().strip()
    if style == "mosaic":
        pix = pixelate(image, block)
        out = composite(image, pix, mask)
    elif style in ("pixelart", "pixel-art"):
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
    else:
        raise HTTPException(status_code=400, detail="style must be one of: mosaic, pixelart")

    # Return PNG
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
