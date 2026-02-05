# AI (PyTorch)

사용자 입력 이미지에서 **인물만 픽셀화**하여 PNG로 반환하는 추론 파이프라인을 구성합니다.
CPU-only 환경을 전제로 하고, 배치 처리(비동기) 기반을 권장합니다.

## 목표 파이프라인

1. 입력 이미지 로드
2. 인물 영역 추출 (segmentation)
3. 인물 영역 픽셀화 (pixelation)
4. 배경과 합성
5. PNG로 저장

## 권장 구조

```
ai/
  configs/        # 설정 파일 (YAML 등)
  data/           # 샘플/테스트 데이터
  models/         # 가중치 저장
  outputs/        # 결과 출력
  scripts/        # 실행 스크립트 (train/infer)
  src/
    datasets/     # 데이터셋 로더
    models/       # 모델 정의/래퍼
    preprocess/   # 전처리
    postprocess/  # 후처리
    services/     # 추론 서비스 로직
    utils/        # 공용 유틸
  tests/
```

## 기본 설계안 (CPU-only)

- 인물 분리: 경량 인물 세그멘테이션 모델 (ex. MobileNet 기반)
- 픽셀화: 인물 마스크 영역만 블록 단위 평균/모자이크 처리
- 합성: 원본 + 픽셀화된 인물 영역 합성
- 처리 방식: 요청 큐 -> 배치 처리 -> 결과 PNG 저장

## 실행 예시

```
python ai/scripts/infer.py --input data/sample.jpg --output outputs/sample.png --block 12
```

## API 실행 (Postman 테스트용)

```
uvicorn ai.src.services.api:app --host 0.0.0.0 --port 8000
```

### Postman 요청

- Method: `POST`
- URL: `http://localhost:8000/pixelate`
- Body: `form-data`
  - key: `file` (type: File)
  - key: `style` (type: Text, optional) -> `pixelart` (기본), `mosaic`
  - key: `block` (type: Text, optional) -> mosaic용
  - key: `long_edge` (type: Text, optional) -> pixelart용 (기본 96)
  - key: `palette` (type: Text, optional) -> pixelart용 (기본 24)
  - key: `dither` (type: Text, optional) -> pixelart용 (기본 false)
  - key: `outline` (type: Text, optional) -> pixelart용 (기본 true)
  - key: `edge_threshold` (type: Text, optional) -> pixelart용 (기본 0.15)
  - key: `background` (type: Text, optional) -> `transparent` (기본), `original`
