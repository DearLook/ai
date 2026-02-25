import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.segmentation import PersonSegmenter
from src.services.pipeline import resize_mask


def _iter_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _mask_to_bbox(mask: np.ndarray, threshold: float = 0.5, pad: int = 8):
    h, w = mask.shape
    ys, xs = np.where(mask >= threshold)
    if xs.size == 0 or ys.size == 0:
        return 0, 0, w, h
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, w)
    y1 = min(int(ys.max()) + pad + 1, h)
    return x0, y0, x1, y1


def _save_sample(
    image: Image.Image, mask: np.ndarray, out_png: Path, out_txt: Path, token: str, caption_suffix: str
):
    rgba = image.convert("RGBA")
    alpha = (mask * 255).astype(np.uint8)
    rgba.putalpha(Image.fromarray(alpha))

    x0, y0, x1, y1 = _mask_to_bbox(mask)
    crop = rgba.crop((x0, y0, x1, y1))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_png, format="PNG")
    out_txt.write_text(f"{token}, {caption_suffix}\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Prepare character LoRA training dataset from full-body photos.")
    parser.add_argument("--input-dir", required=True, help="Raw photo directory")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory")
    parser.add_argument("--token", default="dearlook_char", help="Training token")
    parser.add_argument(
        "--caption-suffix",
        default="full body, standing, pixel-art character reference, plain background",
        help="Shared caption suffix",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    seg = PersonSegmenter()

    count = 0
    for idx, path in enumerate(_iter_images(in_dir), start=1):
        img = Image.open(path).convert("RGB")
        mask = seg.predict_mask(img)
        mask = resize_mask(mask, img.size)

        stem = f"{idx:05d}"
        out_png = out_dir / f"{stem}.png"
        out_txt = out_dir / f"{stem}.txt"
        _save_sample(img, mask, out_png, out_txt, args.token, args.caption_suffix)
        count += 1

    print(f"Prepared {count} samples -> {out_dir}")


if __name__ == "__main__":
    main()
