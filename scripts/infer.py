"""CLI inference: image -> person mask -> pixelated PNG."""

import argparse
import os
import sys

from PIL import Image

# Allow running as a script from repo root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.segmentation import PersonSegmenter  # noqa: E402
from src.services.pipeline import PixelateConfig, run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--block", type=int, default=12)
    args = parser.parse_args()

    image = Image.open(args.input).convert("RGB")
    segmenter = PersonSegmenter(model_path=args.model)
    mask = segmenter.predict_mask(image)

    run_pipeline(args.input, args.output, mask, PixelateConfig(block_size=args.block))


if __name__ == "__main__":
    main()
