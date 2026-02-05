"""CLI inference: image -> person mask -> pixelated PNG."""

import argparse

from PIL import Image

from src.models.segmentation import PersonSegmenter
from src.services.pipeline import PixelateConfig, run_pipeline


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
