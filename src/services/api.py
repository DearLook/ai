import io, os, sys, time
from typing import Optional, Dict, Any, Tuple

import asyncio, uuid
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
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

_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = asyncio.Lock()
_TASKS: set[asyncio.Task] = set()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _job_public(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "created_ms": job["created_ms"],
        "updated_ms": job["updated_ms"],
        "error": job.get("error"),
        "meta": job.get("meta", {}),
    }


async def _run_blocking(fn, *args, **kwargs):
    if hasattr(asyncio, "to_thread"):
        return await asyncio.to_thread(fn, *args, **kwargs)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


def _pixelate_sync(content: bytes, params: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
    image = Image.open(io.BytesIO(content)).convert("RGB")
    segmenter = get_segmenter()
    mask = segmenter.predict_mask(image)
    mask = resize_mask(mask, image.size)

    style = str(params.get("style", "pixelart")).lower().strip()
    block = int(params.get("block", 12))
    long_edge = int(params.get("long_edge", 160))
    palette = int(params.get("palette", 48))
    dither = bool(params.get("dither", False))
    outline = bool(params.get("outline", False))
    edge_threshold = float(params.get("edge_threshold", 0.12))
    smooth = int(params.get("smooth", 3))
    color = float(params.get("color", 1.15))
    contrast = float(params.get("contrast", 1.1))
    background = str(params.get("background", "transparent"))

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
        stylizer = get_diffusion_stylizer()
        pix = pixel_art_person_diffusion(image, mask, stylizer)
        if background == "original":
            base = image.convert("RGBA")
            base.paste(pix, (0, 0), pix)
            out = base
        else:
            out = pix

    else:
        raise ValueError("style must be one of: mosaic, pixelart, model")

    buf = io.BytesIO()
    out.save(buf, format="PNG")

    meta = {
        "style_input": style,
        "mode_used": mode_used,
        "device": settings.PIXELART_DEVICE,
        "dtype": settings.PIXELART_DTYPE,
        "model_id": settings.PIXELART_DIFFUSER_MODEL_ID,
        "lora_path": settings.PIXELART_LORA_PATH,
        "lora_scale": str(getattr(settings, "PIXELART_LORA_SCALE", 1.0)),
    }
    return buf.getvalue(), meta


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


async def _run_pixelate_job(job_id: str, content: bytes, params: Dict[str, Any]) -> None:
    async with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        job["status"] = "RUNNING"
        job["updated_ms"] = _now_ms()

    async def _is_cancelled_locked() -> bool:
        async with _JOBS_LOCK:
            j = _JOBS.get(job_id)
            return bool(j and j.get("cancelled"))

    try:
        if await _is_cancelled_locked():
            async with _JOBS_LOCK:
                j = _JOBS.get(job_id)
                if j:
                    j["status"] = "CANCELLED"
                    j["updated_ms"] = _now_ms()
            return

        png_bytes, meta = await _run_blocking(_pixelate_sync, content, params)

        if await _is_cancelled_locked():
            async with _JOBS_LOCK:
                j = _JOBS.get(job_id)
                if j:
                    j["status"] = "CANCELLED"
                    j["updated_ms"] = _now_ms()
            return

        async with _JOBS_LOCK:
            j = _JOBS.get(job_id)
            if j:
                if j.get("cancelled"):
                    j["status"] = "CANCELLED"
                    j["updated_ms"] = _now_ms()
                    return
                j["status"] = "SUCCEEDED"
                j["updated_ms"] = _now_ms()
                j["result_png"] = png_bytes
                j["meta"] = meta

    except Exception as exc:
        async with _JOBS_LOCK:
            j = _JOBS.get(job_id)
            if j:
                if j.get("cancelled"):
                    j["status"] = "CANCELLED"
                else:
                    j["status"] = "FAILED"
                    j["error"] = f"{type(exc).__name__}: {exc}"
                j["updated_ms"] = _now_ms()
        return


@app.post("/pixelate/async")
async def pixelate_person_async(
    request: Request,
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
    if await request.is_disconnected():
        raise HTTPException(status_code=499, detail="client_disconnected")

    style_norm = style.lower().strip()
    if style_norm not in ("mosaic", "pixelart", "pixel-art", "model", "pixelart-model", "stylized"):
        raise HTTPException(status_code=400, detail="style must be one of: mosaic, pixelart, model")

    content = await file.read()
    job_id = uuid.uuid4().hex

    job = {
        "job_id": job_id,
        "status": "QUEUED",
        "created_ms": _now_ms(),
        "updated_ms": _now_ms(),
        "cancelled": False,
        "result_png": None,
        "error": None,
        "meta": {},
    }

    async with _JOBS_LOCK:
        _JOBS[job_id] = job

    params = {
        "style": style_norm,
        "block": block,
        "long_edge": long_edge,
        "palette": palette,
        "dither": dither,
        "outline": outline,
        "edge_threshold": edge_threshold,
        "smooth": smooth,
        "color": color,
        "contrast": contrast,
        "background": background,
    }

    task = asyncio.create_task(_run_pixelate_job(job_id, content, params))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)

    async with _JOBS_LOCK:
        j = _JOBS.get(job_id)
        if j is not None:
            j["task"] = task

    return {"job_id": job_id, "status": "QUEUED"}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")
    return _job_public(job)


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    async with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job_not_found")

        if job["status"] in ("QUEUED", "RUNNING"):
            job["cancelled"] = True
            job["status"] = "CANCELLED"
            job["updated_ms"] = _now_ms()
        else:
            raise HTTPException(status_code=409, detail="job_already_finished")

        return _job_public(job)


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")

    status = job.get("status")

    if status == "SUCCEEDED" and job.get("result_png"):
        headers = {
            "X-Job-Id": job_id,
            "X-Job-Status": status,
            "X-Mode-Used": str(job.get("meta", {}).get("mode_used", "")),
        }
        return Response(content=job["result_png"], media_type="image/png", headers=headers)

    if status == "FAILED":
        raise HTTPException(status_code=500, detail=job.get("error") or "job_failed")

    if status == "CANCELLED":
        raise HTTPException(status_code=409, detail="job_cancelled")

    raise HTTPException(status_code=202, detail=f"job_{status.lower()}")


@app.post("/pixelate")
async def pixelate_person(
    request: Request,
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

    if await request.is_disconnected():
        raise HTTPException(status_code=499, detail="client_disconnected")

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