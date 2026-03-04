from __future__ import annotations

import io, os, sys, time
from typing import Any

import asyncio, uuid
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import Response
from PIL import Image
import threading

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.segmentation import PersonSegmenter
from src.models.cartoon_stylizer import CartoonStylizer, CartoonConfig
from src.services.pipeline import PixelArtConfig, pixel_art_person_cartoon, resize_mask

app = FastAPI(title="DearLook AI")

_segmenter: PersonSegmenter | None = None
_cartoon_stylizer: CartoonStylizer | None = None

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = asyncio.Lock()
_TASKS: set[asyncio.Task] = set()
_stylizer_lock = threading.Lock()

_VALID_BACKGROUNDS = ("transparent", "white", "original")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _normalize_background(bg: str) -> str:
    b = bg.lower().strip()
    if b not in _VALID_BACKGROUNDS:
        raise ValueError(f"background must be one of: {', '.join(_VALID_BACKGROUNDS)}")
    return b


def _job_public(job: dict[str, Any]) -> dict[str, Any]:
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


def get_segmenter() -> PersonSegmenter:
    global _segmenter
    if _segmenter is None:
        _segmenter = PersonSegmenter()
    return _segmenter


def get_cartoon_stylizer() -> CartoonStylizer:
    global _cartoon_stylizer
    with _stylizer_lock:
        if _cartoon_stylizer is None:
            _cartoon_stylizer = CartoonStylizer(CartoonConfig())
    return _cartoon_stylizer


def _pixelate_sync(content: bytes, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    image = Image.open(io.BytesIO(content)).convert("RGB")
    segmenter = get_segmenter()
    mask = segmenter.predict_mask(image)
    mask = resize_mask(mask, image.size)

    background = _normalize_background(str(params.get("background", "white")))
    long_edge = int(params.get("long_edge", 160))
    palette = int(params.get("palette", 48))

    stylizer = get_cartoon_stylizer()
    config = PixelArtConfig(
        target_long_edge=long_edge,
        palette_size=palette,
    )
    out = pixel_art_person_cartoon(image, mask, stylizer, config, background=background)

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue(), {"mode": "cartoon_pixelart"}


async def _run_pixelate_job(job_id: str, content: bytes, params: dict[str, Any]) -> None:
    async with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        job["status"] = "RUNNING"
        job["updated_ms"] = _now_ms()

    async def _is_cancelled() -> bool:
        async with _JOBS_LOCK:
            j = _JOBS.get(job_id)
            return bool(j and j.get("cancelled"))

    try:
        if await _is_cancelled():
            async with _JOBS_LOCK:
                j = _JOBS.get(job_id)
                if j:
                    j["status"] = "CANCELLED"
                    j["updated_ms"] = _now_ms()
            return

        png_bytes, meta = await _run_blocking(_pixelate_sync, content, params)

        if await _is_cancelled():
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
                j["status"] = "CANCELLED" if j.get("cancelled") else "FAILED"
                if not j.get("cancelled"):
                    j["error"] = f"{type(exc).__name__}: {exc}"
                j["updated_ms"] = _now_ms()


@app.post("/pixelate/async")
async def pixelate_person_async(
    request: Request,
    file: UploadFile = File(...),
    background: str = Form("white"),
    long_edge: int = Form(160),
    palette: int = Form(48),
):
    if await request.is_disconnected():
        raise HTTPException(status_code=499, detail="client_disconnected")

    try:
        background = _normalize_background(background)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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

    params = {"background": background, "long_edge": long_edge, "palette": palette}
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
    async with _JOBS_LOCK:
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
    async with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job_not_found")
        status = job.get("status")
        result_png = job.get("result_png")
        error = job.get("error")

    if status == "SUCCEEDED" and result_png:
        return Response(
            content=result_png,
            media_type="image/png",
            headers={"X-Job-Id": job_id, "X-Job-Status": status},
        )
    if status == "FAILED":
        raise HTTPException(status_code=500, detail=error or "job_failed")
    if status == "CANCELLED":
        raise HTTPException(status_code=409, detail="job_cancelled")
    raise HTTPException(status_code=202, detail=f"job_{status.lower()}")


@app.post("/pixelate")
async def pixelate_person(
    request: Request,
    file: UploadFile = File(...),
    background: str = Form("white"),
    long_edge: int = Form(160),
    palette: int = Form(48),
):
    start = time.time()

    if await request.is_disconnected():
        raise HTTPException(status_code=499, detail="client_disconnected")

    try:
        background = _normalize_background(background)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    content = await file.read()
    try:
        png_bytes, _ = _pixelate_sync(
            content, {"background": background, "long_edge": long_edge, "palette": palette}
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"pixelate_failed: {exc}") from exc

    elapsed_ms = int((time.time() - start) * 1000)
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"X-Elapsed-Ms": str(elapsed_ms)},
    )
