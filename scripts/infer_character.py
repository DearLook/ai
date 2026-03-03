"""
로컬 CLI 테스트 스크립트 — character_v1 (SDXL + LoRA) 파이프라인

사용법:
    python scripts/infer_character.py --input data/사진/test1.png --output out.png
    python scripts/infer_character.py --input data/사진/test1.png --output out.png --background white
    python scripts/infer_character.py --input data/사진/test1.png --output out.png --background transparent

옵션:
    --input       입력 이미지 경로 (필수)
    --output      출력 PNG 경로 (필수)
    --background  white | transparent | original  (기본: white)
    --seed        랜덤 시드 (기본: settings 값)
    --strength    img2img 강도 0.0~1.0 (기본: settings 값)
    --lora-scale  LoRA 적용 강도 (기본: settings 값)
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PIL import Image

from src.config.settings import get_settings
from src.models.pixelart_diffusion import PixelArtDiffusionStylizer, default_diffusion_config
from src.models.segmentation import PersonSegmenter
from src.services.pipeline import pixel_art_person_diffusion, resize_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="character_v1 pixel art pipeline")
    parser.add_argument("--input", required=True, help="입력 이미지 경로")
    parser.add_argument("--output", required=True, help="출력 PNG 경로")
    parser.add_argument(
        "--background",
        default="white",
        choices=["white", "transparent", "original"],
        help="배경 옵션 (기본: white)",
    )
    parser.add_argument("--seed", type=int, default=None, help="랜덤 시드")
    parser.add_argument("--strength", type=float, default=None, help="img2img 강도 (0~1)")
    parser.add_argument("--lora-scale", type=float, default=None, dest="lora_scale", help="LoRA 강도")
    args = parser.parse_args()

    settings = get_settings()
    config = default_diffusion_config(settings)

    # CLI 인자로 설정 오버라이드
    if args.seed is not None:
        config.seed = args.seed
    if args.strength is not None:
        config.strength = args.strength
    if args.lora_scale is not None:
        config.lora_scale = args.lora_scale

    print(f"[설정]")
    print(f"  device      : {config.device}")
    print(f"  model       : {config.model_id}")
    print(f"  lora        : {config.lora_path} (enabled={config.use_lora}, scale={config.lora_scale})")
    print(f"  steps       : {config.num_inference_steps}")
    print(f"  strength    : {config.strength}")
    print(f"  guidance    : {config.guidance_scale}")
    print(f"  post_grid   : {config.post_grid}")
    print(f"  post_palette: {config.post_palette}")
    print(f"  background  : {args.background}")
    if config.lora_block_reason:
        print(f"  [경고] LoRA 비활성: {config.lora_block_reason}")

    print(f"\n[1/3] 이미지 로딩: {args.input}")
    image = Image.open(args.input).convert("RGB")
    print(f"      크기: {image.size}")

    print("[2/3] 인물 세그멘테이션...")
    segmenter = PersonSegmenter()
    mask = segmenter.predict_mask(image)
    mask = resize_mask(mask, image.size)
    person_ratio = float(mask.mean()) * 100
    print(f"      인물 영역: {person_ratio:.1f}%")

    print("[3/3] 픽셀아트 변환 (SDXL img2img)...")
    stylizer = PixelArtDiffusionStylizer(config)
    out = pixel_art_person_diffusion(image, mask, stylizer, background=args.background)

    out.save(args.output, format="PNG")
    print(f"\n[완료] 저장됨: {args.output}  (크기: {out.size})")


if __name__ == "__main__":
    main()
