# AI (PyTorch)

사용자 입력 이미지에서 **인물만 픽셀화**하여 PNG로 반환하는 추론 파이프라인을 구성합니다.
CPU-only 환경을 전제로 하고, 배치 처리(비동기) 기반을 권장합니다.

## 목표 파이프라인

1. 입력 이미지 로드
2. 인물 영역 추출 (segmentation)
3. 인물 영역 픽셀화 (pixelation)
4. 배경과 합성
5. PNG로 저장

## 권장 구조 (현재 폴더 기준)

```
.
  data/           # 샘플/테스트 데이터
  models/         # 가중치 저장 (pixelart 모델 포함)
  scripts/        # 실행 스크립트 (infer)
  src/
    models/       # 모델 정의/래퍼
    services/     # 추론 서비스 로직
```

## 기본 설계안 (CPU-only)

- 인물 분리: 경량 인물 세그멘테이션 모델 (ex. MobileNet 기반)
- 픽셀화: 인물 마스크 영역만 블록 단위 평균/모자이크 처리
- 합성: 원본 + 픽셀화된 인물 영역 합성
- 처리 방식: 요청 큐 -> 배치 처리 -> 결과 PNG 저장

## 실행 예시

```
python scripts/infer.py --input data/sample.jpg --output outputs/sample.png --block 12
```

## API 실행 (Postman 테스트용)

```
uvicorn src.services.api:app --host 0.0.0.0 --port 8000
```

### Postman 요청

- Method: `POST`
- URL: `http://localhost:8000/pixelate`
- Body: `form-data`
  - key: `file` (type: File)
  - key: `style` (type: Text, optional) -> `pixelart` (기본), `mosaic`, `model`
  - key: `block` (type: Text, optional) -> mosaic용
  - key: `long_edge` (type: Text, optional) -> pixelart용 (기본 160)
  - key: `palette` (type: Text, optional) -> pixelart용 (기본 48)
  - key: `dither` (type: Text, optional) -> pixelart용 (기본 false)
  - key: `outline` (type: Text, optional) -> pixelart용 (기본 false)
  - key: `edge_threshold` (type: Text, optional) -> pixelart용 (기본 0.12)
  - key: `smooth` (type: Text, optional) -> pixelart용 (기본 3)
  - key: `color` (type: Text, optional) -> pixelart용 (기본 1.15)
  - key: `contrast` (type: Text, optional) -> pixelart용 (기본 1.1)
  - key: `background` (type: Text, optional) -> `transparent` (기본), `original`

## 설정 파일

- `.env` (배포/운영용 추천)
- `src/config/settings.py` (Pydantic Settings)

`.env` 값이 있으면 `src/config/settings.py`에서 자동으로 읽어 설정됩니다.

## 전용 픽셀아트 모델 (권장 경로)

- diffusion 모델 (권장, 고품질)
  - 기본 베이스: `stabilityai/stable-diffusion-xl-base-1.0`
  - LoRA 파일: `models/pixelart/pixel-art-xl-v1.1.safetensors` (파일명은 자유, 경로만 맞추면 됨)
  - 환경변수:
    - `PIXELART_DIFFUSER_MODEL_ID`
    - `PIXELART_LORA_PATH`
    - `PIXELART_PROMPT`
    - `PIXELART_NEGATIVE_PROMPT`
    - `PIXELART_STEPS`
    - `PIXELART_STRENGTH`
    - `PIXELART_GUIDANCE`
    - `PIXELART_MAX_SIZE`
    - `PIXELART_LORA_SCALE`
    - `PIXELART_DEVICE` (`cpu`/`mps`)
    - `PIXELART_DTYPE` (`fp16`/`bf16`/`fp32`)

- TorchScript 모델 (대체용, 가벼움)
  - 기본 경로: `models/pixelart/pixelart.ts`
  - 환경변수:
    - `PIXELART_MODEL_PATH`
    - `PIXELART_MODEL_INPUT`
    - `PIXELART_MODEL_NORMALIZE`
    - `PIXELART_DEVICE` (`cpu`/`mps`)

## 참고
- diffusion 기반은 CPU에서 매우 느릴 수 있음 (수십 초 ~ 수 분)

## LoRA 다운로드 후보 (SDXL)
- Pixel Art (LoRA) v1.0 (SDXL 1.0 기반)  
  [CivitAI](https://civitai.work/models/266711/sdxlpixel-art-lora)
- Soft Pixel Art XL (SDXL 1.0 기반)  
  [CivitAI](https://civitai.green/models/230035/soft-pixel-art-xl)
